import argparse
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

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse', vq_weight=0.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type
        self.vq_weight = vq_weight

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]
        if "vq_loss" in output:
            out["vq_loss"] = output["vq_loss"]
            if self.vq_weight > 0:
                out["loss"] = out["loss"] + self.vq_weight * out["vq_loss"]

        return out


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
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters
    trainable_params = {
        n for n, p in net.named_parameters() if p.requires_grad
    }

    assert len(inter_params) == 0
    assert union_params == trainable_params

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
    type='mse',
    log_interval=200,
    global_step=0,
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        log_payload = {
            "train/loss": out_criterion["loss"].item(),
            "train/bpp_loss": out_criterion["bpp_loss"].item(),
            "train/aux_loss": aux_loss.item(),
            "train/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        }
        if type == 'mse':
            log_payload["train/mse_loss"] = out_criterion["mse_loss"].item()
        else:
            log_payload["train/ms_ssim"] = out_criterion["ms_ssim_loss"].item()
        if "vq_loss" in out_criterion:
            log_payload["train/vq_loss"] = out_criterion["vq_loss"].item()
        wandb.log(log_payload, step=global_step)
        global_step += 1

        if i % log_interval == 0:
            extra_vq = (
                f'\tVQ loss: {out_criterion["vq_loss"].item():.4f}'
                if "vq_loss" in out_criterion
                else ""
            )
            if type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                    f"{extra_vq}"
                )
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                    f"{extra_vq}"
                )

    return global_step


def test_epoch(epoch, test_dataloader, model, criterion, type='mse', global_step=None):
    model.eval()
    device = next(model.parameters()).device
    log_step = global_step if global_step is not None else epoch
    track_vq = getattr(model, "use_vq", False)
    if type == 'mse':
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()
        vq_loss = AverageMeter() if track_vq else None

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])
                if vq_loss is not None and "vq_loss" in out_criterion:
                    vq_loss.update(out_criterion["vq_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}"
            f"{f' |\\tVQ loss: {vq_loss.avg:.4f}' if vq_loss is not None else ''}\n"
        )
        wandb.log(
            {
                "test/loss": loss.avg,
                "test/mse_loss": mse_loss.avg,
                "test/bpp_loss": bpp_loss.avg,
                "test/aux_loss": aux_loss.avg,
                "epoch": epoch,
                **({"test/vq_loss": vq_loss.avg} if vq_loss is not None else {}),
            },
            step=log_step,
        )

    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()
        vq_loss = AverageMeter() if track_vq else None

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])
                if vq_loss is not None and "vq_loss" in out_criterion:
                    vq_loss.update(out_criterion["vq_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}"
            f"{f' |\\tVQ loss: {vq_loss.avg:.4f}' if vq_loss is not None else ''}\n"
        )
        wandb.log(
            {
                "test/loss": loss.avg,
                "test/ms_ssim": ms_ssim_loss.avg,
                "test/bpp_loss": bpp_loss.avg,
                "test/aux_loss": aux_loss.avg,
                "epoch": epoch,
                **({"test/vq_loss": vq_loss.avg} if vq_loss is not None else {}),
            },
            step=log_step,
        )

    return loss.avg


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
    parser.add_argument("--use_vq", action="store_true", help="Enable SimVQ-based latent quantization")
    parser.add_argument("--vq_codebook_size", type=int, default=512, help="Number of entries in the VQ codebook")
    parser.add_argument("--vq_beta", type=float, default=0.25, help="Commitment weight used inside SimVQ")
    parser.add_argument("--vq_proj_depth", type=int, default=2, help="Depth of the shared post-codebook projection (0 disables it)")
    parser.add_argument("--vq_proj_hidden_dim", type=int, help="Hidden dimension of the projection layers (defaults to slice dim)")
    parser.add_argument("--vq_proj_dropout", type=float, default=0.0, help="Dropout probability for projection layers")
    parser.add_argument(
        "--vq_proj_type",
        type=str,
        default="conv",
        choices=["conv", "mlp"],
        help="Layer type for the shared post-codebook projection",
    )
    parser.add_argument(
        "--disable_vq_proj_residual",
        action="store_true",
        help="Disable residual skip connections inside the projection head",
    )
    parser.add_argument(
        "--vq_loss_weight",
        type=float,
        default=0.0,
        help="Weight applied to the VQ commitment loss when optimizing",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
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
        use_vq=args.use_vq,
        vq_codebook_size=args.vq_codebook_size,
        vq_beta=args.vq_beta,
        vq_proj_depth=args.vq_proj_depth,
        vq_proj_hidden_dim=args.vq_proj_hidden_dim,
        vq_proj_dropout=args.vq_proj_dropout,
        vq_proj_use_residual=not args.disable_vq_proj_residual,
        vq_proj_type=args.vq_proj_type,
    )
    net = net.to(device)
    wandb.watch(net, log="all", log_freq=100)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type, vq_weight=args.vq_loss_weight)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    global_step = 0
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        global_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            type,
            global_step=global_step
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, type, global_step=global_step)
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
                },
                is_best,
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )


if __name__ == "__main__":
    main(sys.argv[1:])
