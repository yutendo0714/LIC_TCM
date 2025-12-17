from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from einops import rearrange 
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import math

from .simvq import SimVQ
import bisect

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

class ArithmeticEncoder:
    def __init__(self, precision=32):
        self.precision = precision
        self.low = 0
        self.high = (1 << precision) - 1
        self.pending_bits = 0
        self.buffer = bytearray()
        self.current_byte = 0
        self.bits_filled = 0

    def _output_bit(self, bit):
        self.current_byte = ((self.current_byte << 1) | bit) & 0xFF
        self.bits_filled += 1
        if self.bits_filled == 8:
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bits_filled = 0

    def _bit_plus_follow(self, bit):
        self._output_bit(bit)
        while self.pending_bits > 0:
            self._output_bit(1 - bit)
            self.pending_bits -= 1

    def encode_symbol(self, cdf, symbol, total_freq):
        range_ = self.high - self.low + 1
        self.high = self.low + (range_ * cdf[symbol + 1] // total_freq) - 1
        self.low = self.low + (range_ * cdf[symbol] // total_freq)
        half = 1 << (self.precision - 1)
        quarter = half >> 1
        three_quarter = quarter * 3
        while True:
            if self.high < half:
                self._bit_plus_follow(0)
            elif self.low >= half:
                self._bit_plus_follow(1)
                self.low -= half
                self.high -= half
            elif self.low >= quarter and self.high < three_quarter:
                self.pending_bits += 1
                self.low -= quarter
                self.high -= quarter
            else:
                break
            self.low = (self.low << 1) & ((1 << self.precision) - 1)
            self.high = ((self.high << 1) & ((1 << self.precision) - 1)) | 1

    def finalize(self):
        half = 1 << (self.precision - 1)
        if self.low < half:
            self._bit_plus_follow(0)
        else:
            self._bit_plus_follow(1)
        if self.bits_filled > 0:
            self.buffer.append((self.current_byte << (8 - self.bits_filled)) & 0xFF)
            self.current_byte = 0
            self.bits_filled = 0
        return bytes(self.buffer)


class ArithmeticDecoder:
    def __init__(self, data, precision=32):
        self.precision = precision
        self.low = 0
        self.high = (1 << precision) - 1
        self.data = data
        self.byte_index = 0
        self.current_byte = 0
        self.bits_left = 0
        self.value = 0
        for _ in range(precision):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self):
        if self.bits_left == 0:
            if self.byte_index < len(self.data):
                self.current_byte = self.data[self.byte_index]
                self.byte_index += 1
            else:
                self.current_byte = 0
            self.bits_left = 8
        bit = (self.current_byte >> 7) & 1
        self.current_byte = (self.current_byte << 1) & 0xFF
        self.bits_left -= 1
        return bit

    def decode_symbol(self, cdf, total_freq):
        range_ = self.high - self.low + 1
        cum = ((self.value - self.low + 1) * total_freq - 1) // range_
        symbol = bisect.bisect_right(cdf, cum) - 1
        self.high = self.low + (range_ * cdf[symbol + 1] // total_freq) - 1
        self.low = self.low + (range_ * cdf[symbol] // total_freq)
        half = 1 << (self.precision - 1)
        quarter = half >> 1
        three_quarter = quarter * 3
        while True:
            if self.high < half:
                pass
            elif self.low >= half:
                self.low -= half
                self.high -= half
                self.value -= half
            elif self.low >= quarter and self.high < three_quarter:
                self.low -= quarter
                self.high -= quarter
                self.value -= quarter
            else:
                break
            self.low = (self.low << 1) & ((1 << self.precision) - 1)
            self.high = ((self.high << 1) & ((1 << self.precision) - 1)) | 1
            self.value = ((self.value << 1) | self._read_bit()) & ((1 << self.precision) - 1)
        return symbol


def build_cdf_from_logits(logits, precision):
    probs = torch.softmax(logits, dim=0)
    probs = probs.cpu()
    num_symbols = probs.numel()
    total = 1 << precision
    base = torch.ones_like(probs, dtype=torch.long)
    leftover = total - num_symbols
    scaled = torch.floor(probs * leftover).long()
    freq = base + scaled
    diff = total - int(freq.sum().item())
    if diff > 0:
        order = torch.argsort(probs, descending=True)
        freq[order[:diff]] += 1
    elif diff < 0:
        order = torch.argsort(probs)
        idx = 0
        while diff < 0 and idx < num_symbols:
            pos = order[idx]
            if freq[pos] > 1:
                freq[pos] -= 1
                diff += 1
            else:
                idx += 1
    freq = torch.clamp(freq, min=1)
    cdf = torch.cumsum(freq, dim=0)
    cdf[-1] = total
    cdf = torch.cat([torch.zeros(1, dtype=torch.long), cdf], dim=0)
    return cdf.tolist()

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}", 
            state_dict,
            policy,
            dtype,
        )

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W'):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x

class SWAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192) -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(input_dim, input_dim, head_dim, window_size, drop_path)
        if inter_dim is not None:
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path) -> None:
        super().__init__()
        self.block_1 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='W')
        self.block_2 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='SW')
        self.window_size = window_size

    def forward(self, x):
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col+1, padding_row, padding_row+1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.block_1(trans_x)
        trans_x =  self.block_2(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        if resize:
            x = F.pad(x, (-padding_col, -padding_col-1, -padding_row, -padding_row-1))
        return trans_x

class TCM(CompressionModel):
    def __init__(
        self,
        config=[2, 2, 2, 2, 2, 2],
        head_dim=[8, 16, 32, 32, 16, 8],
        drop_path_rate=0,
        N=128,
        M=320,
        num_slices=5,
        max_support_slices=5,
        use_simvq=True,
        vq_codebook_size=512,
        vq_beta=0.25,
        vq_commit_weight=1.0,
        vq_input_norm=False,
        **kwargs,
    ):
        super().__init__(entropy_bottleneck_channels=N)
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        self.M = M
        self.use_simvq = use_simvq
        self.vq_codebook_size = vq_codebook_size
        self.vq_commit_weight = vq_commit_weight
        self.vq_beta = vq_beta
        self.vq_input_norm = vq_input_norm
        self.vq_cdf_precision = kwargs.get("vq_cdf_precision", 16)
        self.vq_cdf_total = 1 << self.vq_cdf_precision
        self.vq_range_precision = kwargs.get("vq_range_precision", 32)
        self.register_buffer("codebook_util", torch.tensor(0.0), persistent=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        self.m_down1 = [ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[1])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down3 = [ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[2])] + \
                      [conv3x3(2*N, M, stride=2)]

        self.m_up1 = [ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up2 = [ConvTransBlock(dim, dim, self.head_dim[4], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[4])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up3 = [ConvTransBlock(dim, dim, self.head_dim[5], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[5])] + \
                      [subpel_conv3x3(2*N, 3, 2)]
        
        self.g_a = nn.Sequential(*[ResidualBlockWithStride(3, 2*N, 2)] + self.m_down1 + self.m_down2 + self.m_down3)
        

        self.g_s = nn.Sequential(*[ResidualBlockUpsample(M, 2*N, 2)] + self.m_up1 + self.m_up2 + self.m_up3)

        self.ha_down1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [conv3x3(2*N, 192, stride=2)]

        self.h_a = nn.Sequential(
            *[ResidualBlockWithStride(320, 2*N, 2)] + \
            self.ha_down1
        )

        self.hs_up1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, 320, 2)]

        self.h_mean_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2*N, 2)] + \
            self.hs_up1
        )

        self.hs_up2 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, 320, 2)]


        self.h_scale_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2*N, 2)] + \
            self.hs_up2
        )


        self.atten_mean = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320//self.num_slices)*min(i, 5)), (320 + (320//self.num_slices)*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )
        self.atten_scale = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320//self.num_slices)*min(i, 5)), (320 + (320//self.num_slices)*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )
        if self.use_simvq:
            if self.M % self.num_slices != 0:
                raise ValueError("M must be divisible by num_slices when using SimVQ.")
            self.slice_dim = self.M // self.num_slices
            self.simvq_layers = nn.ModuleList(
                SimVQ(self.vq_codebook_size, self.slice_dim, beta=self.vq_beta, sane_index_shape=True)
                for _ in range(self.num_slices)
            )
            self.cc_prob_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(2 * (320 + (320//self.num_slices)*min(i, 5)), 256, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(256, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    nn.Conv2d(128, self.vq_codebook_size, kernel_size=1, stride=1),
                ) for i in range(self.num_slices)
            )
        else:
            self.cc_mean_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, (320//self.num_slices), stride=1, kernel_size=3),
                ) for i in range(self.num_slices)
            )
            self.cc_scale_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, (320//self.num_slices), stride=1, kernel_size=3),
                ) for i in range(self.num_slices)
            )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def _encode_indices(self, encoders, logits, indices):
        B, _, H, W = logits.size()
        logits = logits.detach()
        for b in range(B):
            encoder = encoders[b]
            logit_b = logits[b]
            idx_b = indices[b]
            for h in range(H):
                for w in range(W):
                    cdf = build_cdf_from_logits(logit_b[:, h, w], self.vq_cdf_precision)
                    symbol = int(idx_b[h, w].item())
                    encoder.encode_symbol(cdf, symbol, self.vq_cdf_total)

    def _decode_indices(self, decoders, logits):
        B, _, H, W = logits.size()
        logits = logits.detach()
        decoded = torch.empty((B, H, W), dtype=torch.long, device=logits.device)
        for b in range(B):
            decoder = decoders[b]
            logit_b = logits[b]
            for h in range(H):
                for w in range(W):
                    cdf = build_cdf_from_logits(logit_b[:, h, w], self.vq_cdf_precision)
                    symbol = decoder.decode_symbol(cdf, self.vq_cdf_total)
                    decoded[b, h, w] = symbol
        return decoded

    def _normalize_for_vq(self, tensor, eps=1e-5):
        mean = tensor.mean(dim=(1, 2, 3), keepdim=True)
        std = tensor.std(dim=(1, 2, 3), keepdim=True)
        return (tensor - mean) / (std + eps)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        vq_commit_loss = y.new_tensor(0.) if self.use_simvq else None
        codebook_utils = [] if self.use_simvq else None
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_features = self.atten_mean[slice_index](mean_support)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_features = self.atten_scale[slice_index](scale_support)

            if self.use_simvq:
                context = torch.cat([mean_features, scale_features], dim=1)
                logits = self.cc_prob_transforms[slice_index](context)
                logits = logits[:, :, :y_shape[0], :y_shape[1]]
                vq_input = self._normalize_for_vq(y_slice) if self.vq_input_norm else y_slice
                (quantized_slice, _, indices), loss_breakdown = self.simvq_layers[slice_index](vq_input)
                indices = indices.long()
                flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, self.vq_codebook_size)
                flat_indices = indices.reshape(-1)
                log_prob = -F.cross_entropy(flat_logits, flat_indices, reduction="none").view(
                    indices.size(0), 1, indices.size(1), indices.size(2)
                )
                probs = torch.exp(log_prob).clamp(min=1e-9)
                y_likelihood.append(probs)
                y_hat_slice = quantized_slice
                vq_commit_loss = vq_commit_loss + loss_breakdown.commitment
                if codebook_utils is not None:
                    utilized = torch.unique(indices)
                    util_ratio = utilized.numel() / float(self.vq_codebook_size)
                    codebook_utils.append(indices.new_tensor(util_ratio, dtype=torch.float32))
            else:
                mu = self.cc_mean_transforms[slice_index](mean_features)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]
                mu_list.append(mu)
                scale = self.cc_scale_transforms[slice_index](scale_features)
                scale = scale[:, :, :y_shape[0], :y_shape[1]]
                scale_list.append(scale)
                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
                y_likelihood.append(torch.clamp(y_slice_likelihood, min=1e-9))
                y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_features, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        if self.use_simvq:
            means = latent_means
            scales = latent_scales
            y_likelihoods = torch.cat(y_likelihood, dim=1)
            vq_term = self.vq_commit_weight * vq_commit_loss
            if codebook_utils:
                avg_util = torch.stack(codebook_utils).mean()
            else:
                avg_util = y.new_tensor(0.0)
            self.codebook_util = avg_util.detach()
        else:
            means = torch.cat(mu_list, dim=1)
            scales = torch.cat(scale_list, dim=1)
            y_likelihoods = torch.cat(y_likelihood, dim=1)
            vq_term = None
        x_hat = self.g_s(y_hat)

        output = {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para":{"means": means, "scales":scales, "y":y}
        }
        if self.use_simvq:
            output["vq_loss"] = vq_term
            output["codebook_utilization"] = self.codebook_util
        return output

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    @torch.no_grad()
    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        if self.use_simvq:
            encoders = [ArithmeticEncoder(self.vq_range_precision) for _ in range(y.size(0))]
        else:
            y_scales = []
            y_means = []
            cdf = self.gaussian_conditional.quantized_cdf.tolist()
            cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
            offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

            encoder = BufferedRansEncoder()
            symbols_list = []
            indexes_list = []
            y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_features = self.atten_mean[slice_index](mean_support)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_features = self.atten_scale[slice_index](scale_support)

            if self.use_simvq:
                context = torch.cat([mean_features, scale_features], dim=1)
                logits = self.cc_prob_transforms[slice_index](context)
                logits = logits[:, :, :y_shape[0], :y_shape[1]]
                (quantized_slice, _, indices), _ = self.simvq_layers[slice_index](y_slice)
                indices = indices.long()
                self._encode_indices(encoders, logits, indices)
                y_hat_slice = quantized_slice
            else:
                mu = self.cc_mean_transforms[slice_index](mean_features)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale = self.cc_scale_transforms[slice_index](scale_features)
                scale = scale[:, :, :y_shape[0], :y_shape[1]]

                index = self.gaussian_conditional.build_indexes(scale)
                y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
                y_hat_slice = y_q_slice + mu

                symbols_list.extend(y_q_slice.reshape(-1).tolist())
                indexes_list.extend(index.reshape(-1).tolist())
                y_scales.append(scale)
                y_means.append(mu)

            lrp_support = torch.cat([mean_features, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
        if self.use_simvq:
            y_strings = [encoder.finalize() for encoder in encoders]
        else:
            encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
            y_string = encoder.flush()
            y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @torch.no_grad()
    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_strings = strings[0]

        y_hat_slices = []
        if self.use_simvq:
            decoders = [ArithmeticDecoder(s, self.vq_range_precision) for s in y_strings]
        else:
            cdf = self.gaussian_conditional.quantized_cdf.tolist()
            cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
            offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

            decoder = RansDecoder()
            decoder.set_stream(y_strings[0])

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_features = self.atten_mean[slice_index](mean_support)

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_features = self.atten_scale[slice_index](scale_support)

            if self.use_simvq:
                context = torch.cat([mean_features, scale_features], dim=1)
                logits = self.cc_prob_transforms[slice_index](context)
                logits = logits[:, :, :y_shape[0], :y_shape[1]]
                indices = self._decode_indices(decoders, logits)
                y_hat_slice = self.simvq_layers[slice_index].decode_indices(indices)
            else:
                mu = self.cc_mean_transforms[slice_index](mean_features)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale = self.cc_scale_transforms[slice_index](scale_features)
                scale = scale[:, :, :y_shape[0], :y_shape[1]]

                index = self.gaussian_conditional.build_indexes(scale)

                rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
                y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
