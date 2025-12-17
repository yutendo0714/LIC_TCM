import argparse
import copy
import math
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim

from models import TCM
import os
import wandb

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)


def compute_vq_weight(current_epoch, warmup_epochs, start_weight, end_weight):
    if warmup_epochs <= 0:
        return end_weight
    progress = min(max(float(current_epoch), 0.0) / float(warmup_epochs), 1.0)
    return start_weight + (end_weight - start_weight) * progress

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type
        self.vq_weight = 1.0

    def set_vq_weight(self, weight):
        self.vq_weight = weight

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        bpp_terms = []
        for likelihoods in output["likelihoods"].values():
            safe_likelihood = torch.clamp(likelihoods, min=1e-9)
            bpp_terms.append(torch.log(safe_likelihood).sum() / (-math.log(2) * num_pixels))
        out["bpp_loss"] = sum(bpp_terms)
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        if "vq_loss" in output:
            vq_loss = output["vq_loss"]
            out["vq_loss"] = vq_loss
            out["loss"] = out["loss"] + self.vq_weight * vq_loss
            out["vq_weight"] = self.vq_weight

        return out


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            if not param.requires_grad:
                continue
            ema_param.mul_(decay).add_(param, alpha=1 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = {n: p for n, p in net.named_parameters() if p.requires_grad}
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    clip_max_norm,
    vq_weight,
    accumulate_steps,
    ema_model=None,
    ema_decay=None,
    type='mse',
):
    model.train()
    device = next(model.parameters()).device
    total_batches = len(train_dataloader)
    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        out_net = model(d)
        codebook_util = out_net.get("codebook_utilization")

        out_criterion = criterion(out_net, d)
        if not torch.isfinite(out_criterion["loss"]):
            print(f"Warning: non-finite loss at epoch {epoch}, step {i}. Skipping update.")
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            continue
        loss = out_criterion["loss"] / accumulate_steps
        loss.backward()

        aux_loss = model.aux_loss()
        aux_loss_value = aux_loss.item()
        if not math.isfinite(aux_loss_value):
            print(f"Warning: non-finite aux loss at epoch {epoch}, step {i}. Skipping update.")
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            continue
        (aux_loss / accumulate_steps).backward()

        should_step = ((i + 1) % accumulate_steps == 0) or (i + 1 == total_batches)
        if should_step:
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()
            optimizer.zero_grad()
            aux_optimizer.step()
            aux_optimizer.zero_grad()
            if ema_model is not None and ema_decay is not None:
                update_ema(model, ema_model, ema_decay)

        vq_loss = out_criterion.get("vq_loss")
        global_step = epoch * total_batches + i
        log_payload = {
            "train_loss": out_criterion["loss"].item(),
            "train_bpp_loss": out_criterion["bpp_loss"].item(),
            "train_aux_loss": aux_loss_value,
            "train_vq_weight": vq_weight,
        }
        if type == 'mse':
            log_payload["train_mse_loss"] = out_criterion["mse_loss"].item()
        else:
            log_payload["train_ms_ssim_loss"] = out_criterion["ms_ssim_loss"].item()
        if vq_loss is not None:
            log_payload["train_vq_loss"] = vq_loss.item()
        if codebook_util is not None:
            log_payload["train_codebook_util"] = codebook_util.item()
        wandb.log(log_payload, step=global_step)

        if i % 1000 == 0:
            vq_str = f'\tVQ loss: {vq_loss.item():.3f} |' if vq_loss is not None else ""
            weight_str = f'\tVQ weight: {vq_weight:.3f} |'
            util_str = f'\tCodebook util: {codebook_util.item():.3f} |' if codebook_util is not None else ""
            if type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"{vq_str}"
                    f"{weight_str}"
                    f"{util_str}"
                    f"\tAux loss: {aux_loss_value:.2f}"
                )
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"{vq_str}"
                    f"{weight_str}"
                    f"{util_str}"
                    f"\tAux loss: {aux_loss_value:.2f}"
                )


def test_epoch(epoch, test_dataloader, model, criterion, vq_weight, type='mse'):
    model.eval()
    device = next(model.parameters()).device
    if type == 'mse':
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()
        vq_loss = AverageMeter()
        has_vq = False
        codebook_util = AverageMeter()
        has_codebook = False

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])
                if "vq_loss" in out_criterion:
                    vq_loss.update(out_criterion["vq_loss"])
                    has_vq = True
                if "codebook_utilization" in out_net and out_net["codebook_utilization"] is not None:
                    codebook_util.update(out_net["codebook_utilization"])
                    has_codebook = True

        vq_str = f"\tVQ loss: {vq_loss.avg:.3f} |" if has_vq else ""
        weight_str = f"\tVQ weight: {vq_weight:.3f} |"
        codebook_str = f"\tCodebook util: {codebook_util.avg:.3f} |" if has_codebook else ""
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"{vq_str}"
            f"{weight_str}"
            f"{codebook_str}"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

        return loss.avg, (vq_loss.avg if has_vq else None), (codebook_util.avg if has_codebook else None)

    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()
        vq_loss = AverageMeter()
        has_vq = False
        codebook_util = AverageMeter()
        has_codebook = False

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])
                if "vq_loss" in out_criterion:
                    vq_loss.update(out_criterion["vq_loss"])
                    has_vq = True
                if "codebook_utilization" in out_net and out_net["codebook_utilization"] is not None:
                    codebook_util.update(out_net["codebook_utilization"])
                    has_codebook = True

        vq_str = f"\tVQ loss: {vq_loss.avg:.3f} |" if has_vq else ""
        weight_str = f"\tVQ weight: {vq_weight:.3f} |"
        codebook_str = f"\tCodebook util: {codebook_util.avg:.3f} |" if has_codebook else ""
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"{vq_str}"
            f"{weight_str}"
            f"{codebook_str}"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

        return loss.avg, (vq_loss.avg if has_vq else None), (codebook_util.avg if has_codebook else None)


def save_checkpoint(state, is_best, epoch, save_path, filename):
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=20,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    parser.add_argument("--use_simvq", dest="use_simvq", action="store_true")
    parser.add_argument("--no_simvq", dest="use_simvq", action="store_false")
    parser.set_defaults(use_simvq=True)
    parser.add_argument("--vq_codebook_size", type=int, default=512)
    parser.add_argument("--vq_beta", type=float, default=0.25)
    parser.add_argument("--vq_commit_weight", type=float, default=1.0)
    parser.add_argument(
        "--vq_weight_start",
        type=float,
        default=0.0,
        help="Initial multiplier for vq_loss contribution",
    )
    parser.add_argument(
        "--vq_weight_end",
        type=float,
        default=1.0,
        help="Final multiplier for vq_loss contribution",
    )
    parser.add_argument(
        "--vq_warmup_epochs",
        type=int,
        default=10,
        help="Number of epochs to linearly warm up vq_loss weight",
    )
    parser.add_argument(
        "--vq_input_norm",
        action="store_true",
        help="Apply per-sample normalization before SimVQ to stabilize training",
    )
    parser.add_argument(
        "--accumulate_steps",
        type=int,
        default=1,
        help="Accumulate gradients over this many steps before optimizer update",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Track an exponential moving average of model weights",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="Decay factor for EMA updates",
    )
    parser.add_argument(
        "--ema_eval",
        action="store_true",
        help="Use EMA weights for validation/evaluation when EMA is enabled",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    accumulate_steps = max(1, args.accumulate_steps)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    os.makedirs(save_path, exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    wandb_project = os.environ.get("WANDB_PROJECT", "LIC_TCM")
    wandb_run = wandb.init(
        project=wandb_project,
        config=vars(args),
        name=f"lambda_{args.lmbda}_N_{args.N}",
    )

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )


    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    device = 'cuda'

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    train_batches = len(train_dataloader)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = TCM(
        config=[2,2,2,2,2,2],
        head_dim=[8, 16, 32, 32, 16, 8],
        drop_path_rate=0.0,
        N=args.N,
        M=320,
        use_simvq=args.use_simvq,
        vq_codebook_size=args.vq_codebook_size,
        vq_beta=args.vq_beta,
        vq_commit_weight=args.vq_commit_weight,
        vq_input_norm=args.vq_input_norm,
    )
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(net)
        ema_model = ema_model.to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False
    wandb.watch(net, log="all", log_freq=100)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.use_ema:
            if ema_model is None:
                ema_model = copy.deepcopy(net).to(device)
                ema_model.eval()
                for p in ema_model.parameters():
                    p.requires_grad = False
            if "ema_state_dict" in checkpoint:
                ema_model.load_state_dict(checkpoint["ema_state_dict"])
            else:
                ema_model.load_state_dict(net.state_dict())
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        vq_weight = compute_vq_weight(
            epoch, args.vq_warmup_epochs, args.vq_weight_start, args.vq_weight_end
        )
        criterion.set_vq_weight(vq_weight)
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            vq_weight,
            accumulate_steps,
            ema_model if args.use_ema else None,
            args.ema_decay if args.use_ema else None,
            type
        )
        eval_model = ema_model if (args.use_ema and args.ema_eval and ema_model is not None) else net
        loss, test_vq_loss, test_codebook_util = test_epoch(epoch, test_dataloader, eval_model, criterion, vq_weight, type)
        log_payload = {"test_loss": loss, "test_vq_weight": vq_weight}
        if test_vq_loss is not None:
            log_payload["test_vq_loss"] = test_vq_loss
        if test_codebook_util is not None:
            log_payload["test_codebook_util"] = test_codebook_util
        wandb.log(log_payload, step=(epoch + 1) * train_batches)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    **({"ema_state_dict": ema_model.state_dict()} if args.use_ema and ema_model is not None else {}),
                },
                is_best,
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
