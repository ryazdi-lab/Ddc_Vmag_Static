"""
Attention U-Net for reactive dependency matrix (DQ) estimation from DC dependency matrix (Ddc).

Upgrades over the plain U-Net (unet_model.py):
  1. CBAM (Convolutional Block Attention Module) on every skip connection
     — re-weights channel and spatial importance before concatenation,
       so the decoder pays attention to the *relevant* parts of each skip
       feature map instead of using it raw.
  2. Self-attention block at the bottleneck
     — at 1x1 spatial resolution, this is nearly free, and lets the model
       relate any two bus-pair positions directly, compensating for the
       fact that CNN kernels only look at local matrix neighborhoods
       (bus ordering in the N×N matrix is a convention, not a true
       spatial coordinate — see discussion in earlier chat).

Input:  (B, 1, N, N)  — Ddc matrix  (N = number of buses, e.g. 14, 30, 57, 118)
Output: (B, 1, N, N)  — DQ matrix estimate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Channel-attention half of CBAM.

    Squeezes spatial dims via avg-pool AND max-pool, passes both through a
    shared MLP, sums them, and sigmoids to get a per-channel gate.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).flatten(1)   # (B, C)

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        gate = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * gate


class SpatialAttention(nn.Module):
    """
    Spatial-attention half of CBAM.

    Pools across the channel dimension (avg + max), concatenates,
    and applies a 7x7 conv to produce a single-channel spatial gate.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        pooled = torch.cat([avg_out, max_out], dim=1)   # (B, 2, H, W)

        gate = torch.sigmoid(self.conv(pooled))          # (B, 1, H, W)
        return x * gate


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module: channel attention then spatial attention,
    applied sequentially (as in the original paper). Used to refine skip-connection
    features before they're concatenated into the decoder.
    """

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        # inputs shrink fast; clamp spatial kernel so it never exceeds feature size
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class SelfAttention2D(nn.Module):
    """
    Standard scaled dot-product self-attention over spatial positions,
    applied at the bottleneck. Each spatial location (here, bus-pair
    position) attends to every other position — including ones far
    apart in matrix index, which a local conv kernel cannot reach.

    Cheap here because the bottleneck spatial size is only 1x1 to 3x3
    depending on architecture depth.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)                                  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)                       # each (B, C, H, W)

        # Reshape to (B, heads, head_dim, H*W) for multi-head attention
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        # Attention over spatial positions: (B, heads, HW, HW)
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)      # (B, heads, head_dim, HW)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return residual + out   # residual connection


# ---------------------------------------------------------------------------
# Building blocks (shared with base U-Net)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock followed by MaxPool2d (downsampling)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class AttentionDecoderBlock(nn.Module):
    """
    Upsample -> refine skip via CBAM -> concatenate -> ConvBlock.

    The CBAM refinement is the key addition: instead of concatenating
    the raw encoder skip feature map, we first re-weight it by channel
    and spatial importance, letting the decoder focus on the most
    relevant bus-pair regions of the skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int, skip_channels: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.cbam = CBAM(skip_channels)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        skip = self.cbam(skip)          # attention-refined skip features
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full Attention U-Net
# ---------------------------------------------------------------------------

class VectorHead(nn.Module):
    """
    Head for per-bus vector targets (e.g. Vmag), used in place of the
    matrix head when output_mode='vector'.

    Each row i of the final (c, N, N) feature map already summarizes
    bus i's dependencies on every other bus (that's what a row of the
    Ddc/DQ-style matrix means), so we keep that row structure and
    collapse across columns rather than flattening everything into a
    big fully-connected layer:

        (B, c, N, N) --1x1 conv + ReLU--> (B, c, N, N)
                     --mean over columns--> (B, c, N)
                     --1x1 conv (channels -> 1)--> (B, 1, N)
                     --squeeze--> (B, N)

    No final activation: unlike the normalized [0,1] DQ/Ddc matrices,
    Vmag (or other physical per-bus quantities) is not bounded to
    [0,1], so this head outputs raw (unbounded) values.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.refine(x)          # (B, c, N, N)
        x = x.mean(dim=3)           # (B, c, N)   — average each bus-row over columns
        x = self.project(x)         # (B, 1, N)
        return x.squeeze(1)         # (B, N)


class AttentionUNet(nn.Module):
    """
    U-Net for N×N single-channel input (N = number of buses), augmented with:
      - CBAM on every skip connection (channel + spatial attention)
      - Multi-head self-attention at the bottleneck

    Output head is selectable:
      - output_mode='matrix' (default): (B, 1, N, N) matrix, e.g. DQ estimation.
      - output_mode='vector':           (B, N) per-bus vector, e.g. Vmag estimation.

    Channel progression: 1 -> c -> 2c -> 4c -> [self-attn] -> 8c (bottleneck)
                                  |     |    |
                                CBAM  CBAM  CBAM  (on skip connections)
    """

    def __init__(self, base_channels: int = 32, attn_heads: int = 4, output_mode: str = 'matrix'):
        super().__init__()
        if output_mode not in ('matrix', 'vector'):
            raise ValueError(f"output_mode must be 'matrix' or 'vector', got {output_mode!r}")
        self.output_mode = output_mode
        c = base_channels

        # Encoder
        self.enc1 = EncoderBlock(1,     c)       # N   -> N//2, out: c
        self.enc2 = EncoderBlock(c,     c * 2)   # N//2 -> N//4, out: 2c
        self.enc3 = EncoderBlock(c * 2, c * 4)   # N//4 -> N//8, out: 4c

        # Bottleneck: conv projection + self-attention
        self.bottleneck_conv = ConvBlock(c * 4, c * 8)        # 1x1, out: 8c
        self.bottleneck_attn = SelfAttention2D(c * 8, num_heads=attn_heads)

        # Decoder with CBAM-refined skips
        self.dec3 = AttentionDecoderBlock(c * 8, c * 4, skip_channels=c * 4)  # 1->3
        self.dec2 = AttentionDecoderBlock(c * 4, c * 2, skip_channels=c * 2)  # 3->7
        self.dec1 = AttentionDecoderBlock(c * 2, c,     skip_channels=c)      # N//2 -> N

        # Output head
        if output_mode == 'matrix':
            self.head = nn.Sequential(
                nn.Conv2d(c, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.head = VectorHead(c)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, N, N)  — normalized Ddc matrix  (N = number of buses)
        Returns:
            (B, 1, N, N) if output_mode='matrix'  — estimated DQ matrix
            (B, N)       if output_mode='vector'  — estimated per-bus vector (e.g. Vmag)
        """
        # Encoder path
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        # Bottleneck: local conv features + global self-attention
        x = self.bottleneck_conv(x)
        x = self.bottleneck_attn(x)

        # Decoder path with attention-refined skips
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=14,
                        help="Number of buses (matrix side length), e.g. 14, 30, 57, 118")
    args = parser.parse_args()
    N = args.case

    dummy = torch.zeros(4, 1, N, N)

    for mode in ("matrix", "vector"):
        model = AttentionUNet(base_channels=32, attn_heads=4, output_mode=mode)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        out = model(dummy)
        expected = dummy.shape if mode == "matrix" else (dummy.shape[0], N)
        print(f"[{mode}] Model parameters: {total_params:,}")
        print(f"[{mode}] Input shape:  {dummy.shape}")
        print(f"[{mode}] Output shape: {out.shape}")
        assert tuple(out.shape) == tuple(expected), "Shape mismatch!"
        print(f"[{mode}] Shape check passed for case{N}.\n")

    # Compare param count against plain U-Net (matrix mode, for reference)
    try:
        from unet_model import UNet
        attn_matrix  = AttentionUNet(base_channels=32, attn_heads=4, output_mode="matrix")
        attn_params  = sum(p.numel() for p in attn_matrix.parameters() if p.requires_grad)
        plain = UNet(base_channels=32, output_mode="matrix")
        plain_params = sum(p.numel() for p in plain.parameters() if p.requires_grad)
        print(f"Plain U-Net parameters:     {plain_params:,}")
        print(f"Attention U-Net parameters: {attn_params:,}")
        print(f"Overhead: +{attn_params - plain_params:,} params "
              f"({100*(attn_params/plain_params - 1):.1f}% more)")
    except ImportError:
        pass