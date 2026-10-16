"""
Centrality-Aware Attention U-Net for per-bus voltage magnitude (Vmag) estimation.

This is the Vmag counterpart of centrality_unet_model.CentralityAwareUNet.

Why it differs from the DQ version:
    CentralityAwareUNet derives betweenness/closeness from its OWN predicted
    DQ matrix via CentralityLayer (row/column sums of dq_pred). That only
    works because the model's output is itself a matrix.

    Vmag is a per-bus VECTOR, not a matrix, so there is no predicted matrix
    to derive centrality from downstream of the decoder. The "decoder-derived
    centrality" term therefore has no Vmag analogue and is dropped here.

What is kept:
    - The same encoder (CBAM-refined skips) and the same bottleneck stack
      (axial row/column attention + global self-attention) as
      CentralityAwareUNet — these don't depend on the output head at all.
    - The auxiliary CentralityHead: a small FC branch off the bottleneck
      that predicts betweenness/closeness directly from pooled bottleneck
      features. This needs no matrix output, so it transfers unchanged.
      It is supervised against betweenness/closeness computed from the
      ground-truth DQ matrix (dataset's 'y' key), giving the bottleneck an
      explicit centrality-aware inductive bias even though the main task
      is Vmag regression.

Input:  (B, 1, N, N)  — Ddc matrix  (N = number of buses)
Outputs:
  vmag_pred:         (B, N)  — estimated per-bus voltage magnitude
  betweenness_head:  (B, N)  — auxiliary FC-head prediction (from bottleneck)
  closeness_head:    (B, N)  — auxiliary FC-head prediction (from bottleneck)
"""

import torch
import torch.nn as nn

from attention_unet_model import (
    ConvBlock,
    EncoderBlock,
    AttentionDecoderBlock,
    SelfAttention2D,
    VectorHead,
)
from centrality_unet_model import AxialAttentionBlock, CentralityHead


class CentralityAwareVmagUNet(nn.Module):
    """
    Attention U-Net (CBAM skips + axial + global bottleneck attention, same
    as CentralityAwareUNet) but with a VectorHead output for Vmag instead of
    a matrix output for DQ, plus the auxiliary FC CentralityHead for
    centrality-aware supervision.
    """

    def __init__(self, base_channels: int = 32, attn_heads: int = 4, num_buses: int = 14, n_buses: int = None):
        # n_buses is an alias for num_buses for API consistency; n_buses takes priority if both given
        if n_buses is not None:
            num_buses = n_buses
        super().__init__()
        c = base_channels
        self.num_buses = num_buses

        # Encoder (same as AttentionUNet / CentralityAwareUNet)
        self.enc1 = EncoderBlock(1,     c)       # N   -> N//2, out: c
        self.enc2 = EncoderBlock(c,     c * 2)   # N//2 -> N//4, out: 2c
        self.enc3 = EncoderBlock(c * 2, c * 4)   # N//4 -> N//8, out: 4c

        # Bottleneck: conv projection -> axial row/col attention -> global self-attention
        self.bottleneck_conv = ConvBlock(c * 4, c * 8)             # 1x1, out: 8c
        self.axial_attn      = AxialAttentionBlock(c * 8, num_heads=attn_heads)
        self.global_attn      = SelfAttention2D(c * 8, num_heads=attn_heads)

        # Decoder with CBAM-refined skips (same as AttentionUNet)
        self.dec3 = AttentionDecoderBlock(c * 8, c * 4, skip_channels=c * 4)  # 1->3
        self.dec2 = AttentionDecoderBlock(c * 4, c * 2, skip_channels=c * 2)  # 3->7
        self.dec1 = AttentionDecoderBlock(c * 2, c,     skip_channels=c)      # N//2 -> N

        # Output head -> Vmag vector (unbounded, no final activation — see VectorHead docstring)
        self.head = VectorHead(c)

        # Auxiliary centrality head (bottleneck-only; no matrix needed)
        self.centrality_head = CentralityHead(bottleneck_channels=c * 8, num_buses=num_buses)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 1, N, N)  — normalized Ddc matrix (N = number of buses)
        Returns:
            dict with keys:
              'vmag_pred'        : (B, N)  — estimated per-bus voltage magnitude
              'betweenness_head' : (B, N)  — auxiliary FC head prediction
              'closeness_head'   : (B, N)  — auxiliary FC head prediction
        """
        # Encoder path
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        # Bottleneck: local conv -> axial row/col attention -> global self-attention
        x = self.bottleneck_conv(x)
        x = self.axial_attn(x)
        bottleneck_features = self.global_attn(x)

        # Auxiliary centrality head (off bottleneck, before decoding)
        betweenness_head, closeness_head = self.centrality_head(bottleneck_features)

        # Decoder path with attention-refined skips
        x = self.dec3(bottleneck_features, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        vmag_pred = self.head(x)

        return {
            'vmag_pred':        vmag_pred,
            'betweenness_head': betweenness_head,
            'closeness_head':   closeness_head,
        }


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

    model = CentralityAwareVmagUNet(base_channels=32, attn_heads=4, num_buses=N)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    dummy = torch.rand(4, 1, N, N)
    out = model(dummy)

    for k, v in out.items():
        print(f"{k:>18}: {tuple(v.shape)}")

    assert out['vmag_pred'].shape == (4, N)
    assert out['betweenness_head'].shape == (4, N)
    assert out['closeness_head'].shape == (4, N)
    print(f"\nShape checks passed for case{N}.")

    try:
        from centrality_unet_model import CentralityAwareUNet
        dq_model = CentralityAwareUNet(base_channels=32, attn_heads=4, num_buses=N)
        dq_params = sum(p.numel() for p in dq_model.parameters() if p.requires_grad)
        print(f"\nCentralityAwareUNet (DQ) parameters:     {dq_params:,}")
        print(f"CentralityAwareVmagUNet parameters:       {total_params:,}")
    except ImportError:
        pass
