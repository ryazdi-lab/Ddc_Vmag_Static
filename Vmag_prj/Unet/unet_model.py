"""
U-Net model for reactive dependency matrix (DQ) estimation from DC dependency matrix (Ddc).

Architecture: U-Net with encoder-decoder structure and skip connections.
Input:  (B, 1, 14, 14)  — Ddc matrix
Output: (B, 1, 14, 14)  — DQ matrix estimate
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU layers."""

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
        skip = self.conv(x)   # saved for skip connection
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """Bilinear upsampling → concatenate skip → ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # in_channels = channels from below + channels from skip
        self.up   = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims differ (can happen with odd sizes, safe guard)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net
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


class UNet(nn.Module):
    """
    U-Net for 14×14 single-channel input, with a selectable output head:

      - output_mode='matrix' (default): (B, 1, 14, 14) matrix, e.g. DQ estimation.
      - output_mode='vector':           (B, 14) per-bus vector, e.g. Vmag estimation.

    Encoder depth is deliberately kept shallow (3 levels) because
    14×14 → 7×7 → 3×3 exhausts spatial resolution quickly.

    Channel progression: 1 → 32 → 64 → 128  (bottleneck)
                                 ↑    ↑    ↑
                              skip  skip  skip
    """

    def __init__(self, base_channels: int = 32, output_mode: str = 'matrix'):
        super().__init__()
        if output_mode not in ('matrix', 'vector'):
            raise ValueError(f"output_mode must be 'matrix' or 'vector', got {output_mode!r}")
        self.output_mode = output_mode
        c = base_channels

        # Encoder
        self.enc1 = EncoderBlock(1,     c)       # 14→7,  out: c
        self.enc2 = EncoderBlock(c,     c * 2)   # 7→3,   out: 2c
        self.enc3 = EncoderBlock(c * 2, c * 4)   # 3→1,   out: 4c

        # Bottleneck
        self.bottleneck = ConvBlock(c * 4, c * 8)  # 1×1,   out: 8c

        # Decoder
        self.dec3 = DecoderBlock(c * 8, c * 4)   # 1→3,   cat with enc3 skip
        self.dec2 = DecoderBlock(c * 4, c * 2)   # 3→7,   cat with enc2 skip
        self.dec1 = DecoderBlock(c * 2, c)        # 7→14,  cat with enc1 skip

        # Output head
        if output_mode == 'matrix':
            # project to 1 channel, sigmoid to keep values in [0,1]
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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 14, 14)  — normalized Ddc matrix
        Returns:
            (B, 1, 14, 14) if output_mode='matrix'  — estimated DQ matrix
            (B, 14)        if output_mode='vector'  — estimated per-bus vector (e.g. Vmag)
        """
        # Encoder path
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dummy = torch.zeros(4, 1, 14, 14)

    for mode in ("matrix", "vector"):
        model = UNet(base_channels=32, output_mode=mode)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        out = model(dummy)
        print(f"[{mode}] Model parameters: {total_params:,}")
        print(f"[{mode}] Input shape:  {dummy.shape}")
        print(f"[{mode}] Output shape: {out.shape}")
        expected = dummy.shape if mode == "matrix" else (dummy.shape[0], 14)
        assert tuple(out.shape) == tuple(expected), "Shape mismatch!"
        print(f"[{mode}] Shape check passed.\n")
