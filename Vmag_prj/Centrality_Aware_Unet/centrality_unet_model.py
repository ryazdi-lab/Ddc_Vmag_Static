"""
Centrality-Aware Attention U-Net for reactive dependency matrix (DQ) estimation.

Motivation (from Shahraeini, "Convolutional Estimation of Reactive Betweenness
and Dependency from DC Load Flow Data", ICSPIS 2025):

    Betweenness of bus k:  C_b(k) = sum_s D(s, k)   <- COLUMN sum
    Closeness  of bus k:   C_c(k) = sum_t D(k, t)   <- ROW sum

The paper's own results (Table I) show that reactive betweenness C_b^Q
behaves *qualitatively* differently from DC betweenness C_b^dc (e.g. bus 7
jumps rank due to local voltage-support effects DC flow cannot see). This
means: a model can have excellent elementwise R²/RMSE/MAE on DQ and still
get the centrality ranking wrong, even though centrality is the quantity
power engineers actually use downstream (VAR siting, contingency screening).

This model adds three things on top of attention_unet_model.py:

  1. CentralityLayer
     A parameter-free layer that computes betweenness (column-sum) and
     closeness (row-sum) directly from a predicted DQ matrix. Used to build
     an auxiliary loss term that supervises the *aggregate* quantities,
     not just individual matrix cells.

  2. AxialAttention (row-attention + column-attention)
     Self-attention applied along rows and along columns separately, BEFORE
     the bottleneck. Because betweenness/closeness are literally row/column
     sums, giving the network explicit row-wise and column-wise mixing is
     structurally aligned with the quantity we ultimately care about —
     rather than hoping a generic 2D attention head discovers this.

  3. CentralityHead
     A small fully-connected branch off the bottleneck features that
     directly predicts the N-dim betweenness vector and N-dim closeness
     vector (N = number of buses). Trained jointly with the main decoder,
     this forces the bottleneck representation to encode centrality-relevant
     information directly rather than leaving it purely emergent in the decoder.

Input:  (B, 1, N, N)  — Ddc matrix  (N = number of buses, e.g. 14, 30, 57, 118)
Outputs:
  dq_pred:            (B, 1, N, N)  — estimated DQ matrix
  betweenness_decoder: (B, N)       — column-sum of dq_pred (via CentralityLayer)
  closeness_decoder:   (B, N)       — row-sum of dq_pred (via CentralityLayer)
  betweenness_head:    (B, N)       — FC-head prediction (auxiliary, from bottleneck)
  closeness_head:       (B, N)       — FC-head prediction (auxiliary, from bottleneck)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_unet_model import (
    CBAM,
    ConvBlock,
    EncoderBlock,
    AttentionDecoderBlock,
)


# ---------------------------------------------------------------------------
# 1. Differentiable centrality layer
# ---------------------------------------------------------------------------

class CentralityLayer(nn.Module):
    """
    Parameter-free layer computing betweenness and closeness from a
    dependency matrix, per the paper's Eq. (2):

        Betweenness C_b(k) = sum_s D(s, k)   -> column sum -> dim=2 (rows) in (B,1,N,N)
        Closeness   C_c(k) = sum_t D(k, t)   -> row sum    -> dim=3 (cols) in (B,1,N,N)

    Fully differentiable (just torch.sum), so it can sit directly in the
    forward pass and contribute gradients back through the decoder.
    """

    def forward(self, D: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            D: (B, 1, N, N) dependency matrix, D[..., s, k] = dependency of bus s on bus k
        Returns:
            betweenness: (B, N) — column sums, C_b(k) = sum_s D(s,k)
            closeness:   (B, N) — row sums,    C_c(s) = sum_k D(s,k)
        """
        D = D.squeeze(1)                  # (B, N, N)
        betweenness = D.sum(dim=1)        # sum over rows (s) for each column k -> (B, N)
        closeness   = D.sum(dim=2)        # sum over cols (k) for each row s    -> (B, N)
        return betweenness, closeness


# ---------------------------------------------------------------------------
# 2. Axial attention (row-attention, column-attention)
# ---------------------------------------------------------------------------

class AxialAttention(nn.Module):
    """
    Self-attention restricted to one axis at a time.

    mode='row':    each row attends across its own N columns
                    (mirrors closeness = row-sum: lets the network learn
                     which column entries jointly determine a bus's
                     outgoing dependency total)
    mode='column': each column attends across its own N rows
                    (mirrors betweenness = column-sum: lets the network
                     learn which row entries jointly determine a bus's
                     mediation total)

    Implemented by folding the non-attention axis into the batch dimension
    so standard multi-head attention machinery applies per-row or per-column.
    """

    def __init__(self, channels: int, num_heads: int = 4, mode: str = 'row'):
        super().__init__()
        assert mode in ('row', 'column')
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.mode      = mode
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
        qkv = self.qkv(x)                          # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)               # each (B, C, H, W)

        if self.mode == 'row':
            # Attend across columns (W) within each row independently.
            # Fold (B, H) into batch: each row becomes its own attention problem over W positions.
            q = q.permute(0, 2, 1, 3).reshape(B * H, self.num_heads, self.head_dim, W)
            k = k.permute(0, 2, 1, 3).reshape(B * H, self.num_heads, self.head_dim, W)
            v = v.permute(0, 2, 1, 3).reshape(B * H, self.num_heads, self.head_dim, W)
        else:
            # Attend across rows (H) within each column independently.
            q = q.permute(0, 3, 1, 2).reshape(B * W, self.num_heads, self.head_dim, H)
            k = k.permute(0, 3, 1, 2).reshape(B * W, self.num_heads, self.head_dim, H)
            v = v.permute(0, 3, 1, 2).reshape(B * W, self.num_heads, self.head_dim, H)

        attn = torch.einsum('nhdl,nhdm->nhlm', q, k) * self.scale   # (N, heads, L, L)
        attn = torch.softmax(attn, dim=-1)
        out  = torch.einsum('nhlm,nhdm->nhdl', attn, v)              # (N, heads, head_dim, L)

        if self.mode == 'row':
            out = out.reshape(B, H, C, W).permute(0, 2, 1, 3)        # back to (B, C, H, W)
        else:
            out = out.reshape(B, W, C, H).permute(0, 2, 3, 1)        # back to (B, C, H, W)

        out = self.proj(out)
        return residual + out   # residual connection


class AxialAttentionBlock(nn.Module):
    """Row-attention followed by column-attention, each with its own residual."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.row_attn = AxialAttention(channels, num_heads, mode='row')
        self.col_attn = AxialAttention(channels, num_heads, mode='column')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.row_attn(x)
        x = self.col_attn(x)
        return x


# ---------------------------------------------------------------------------
# 3. FC centrality head (auxiliary branch from bottleneck)
# ---------------------------------------------------------------------------

class CentralityHead(nn.Module):
    """
    Small fully-connected branch that predicts betweenness and closeness
    vectors directly from pooled bottleneck features.

    This is auxiliary supervision: it doesn't feed back into the DQ matrix
    reconstruction, but the gradient it produces shapes the bottleneck
    representation to explicitly encode centrality-relevant structure.
    """

    def __init__(self, bottleneck_channels: int, num_buses: int = 14, hidden: int = 64, n_buses: int = None):
        # n_buses is an alias for num_buses for API consistency; n_buses takes priority if both given
        if n_buses is not None:
            num_buses = n_buses
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(bottleneck_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.betweenness_out = nn.Linear(hidden, num_buses)
        self.closeness_out   = nn.Linear(hidden, num_buses)

    def forward(self, bottleneck_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(bottleneck_features).flatten(1)   # (B, C)
        x = self.fc(x)                                   # (B, hidden)
        betweenness = self.betweenness_out(x)             # (B, N)
        closeness   = self.closeness_out(x)                # (B, N)
        return betweenness, closeness


# ---------------------------------------------------------------------------
# Full Centrality-Aware Attention U-Net
# ---------------------------------------------------------------------------

class CentralityAwareUNet(nn.Module):
    """
    Attention U-Net (CBAM skips + bottleneck self-attention, as before) with:
      - Axial row/column attention inserted before the bottleneck
      - A differentiable CentralityLayer computing betweenness/closeness
        from the predicted DQ matrix
      - An auxiliary FC CentralityHead predicting betweenness/closeness
        directly from bottleneck features

    Returns a dict of outputs so the training loop can build a multi-task
    loss: matrix reconstruction + decoder-derived centrality + head-predicted
    centrality, all compared against the ground-truth DQ's betweenness/closeness.
    """

    def __init__(self, base_channels: int = 32, attn_heads: int = 4, num_buses: int = 14, n_buses: int = None):
        # n_buses is an alias for num_buses for API consistency; n_buses takes priority if both given
        if n_buses is not None:
            num_buses = n_buses
        super().__init__()
        c = base_channels
        self.num_buses = num_buses

        # Encoder (same as AttentionUNet)
        self.enc1 = EncoderBlock(1,     c)       # N   -> N//2, out: c
        self.enc2 = EncoderBlock(c,     c * 2)   # N//2 -> N//4, out: 2c
        self.enc3 = EncoderBlock(c * 2, c * 4)   # N//4 -> N//8, out: 4c

        # Bottleneck: conv projection -> axial row/col attention -> global self-attention
        self.bottleneck_conv  = ConvBlock(c * 4, c * 8)             # 1x1, out: 8c
        self.axial_attn       = AxialAttentionBlock(c * 8, num_heads=attn_heads)
        from attention_unet_model import SelfAttention2D
        self.global_attn      = SelfAttention2D(c * 8, num_heads=attn_heads)

        # Decoder with CBAM-refined skips (same as AttentionUNet)
        self.dec3 = AttentionDecoderBlock(c * 8, c * 4, skip_channels=c * 4)  # 1->3
        self.dec2 = AttentionDecoderBlock(c * 4, c * 2, skip_channels=c * 2)  # 3->7
        self.dec1 = AttentionDecoderBlock(c * 2, c,     skip_channels=c)      # N//2 -> N

        # Output head -> DQ matrix
        self.head = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Centrality components
        self.centrality_layer = CentralityLayer()
        self.centrality_head  = CentralityHead(bottleneck_channels=c * 8, num_buses=num_buses)

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
            x: (B, 1, N, N)  — normalized Ddc matrix  (N = number of buses)
        Returns:
            dict with keys:
              'dq_pred'              : (B, 1, N, N)
              'betweenness_decoder'  : (B, N)  — derived from dq_pred via CentralityLayer
              'closeness_decoder'    : (B, N)  — derived from dq_pred via CentralityLayer
              'betweenness_head'     : (B, N)  — auxiliary FC head prediction
              'closeness_head'       : (B, N)  — auxiliary FC head prediction
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

        dq_pred = self.head(x)

        # Derived centrality directly from the predicted matrix
        betweenness_decoder, closeness_decoder = self.centrality_layer(dq_pred)

        return {
            'dq_pred':             dq_pred,
            'betweenness_decoder': betweenness_decoder,
            'closeness_decoder':   closeness_decoder,
            'betweenness_head':    betweenness_head,
            'closeness_head':      closeness_head,
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

    model = CentralityAwareUNet(base_channels=32, attn_heads=4, n_buses=N)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    dummy = torch.rand(4, 1, N, N)
    out = model(dummy)

    for k, v in out.items():
        print(f"{k:>22}: {tuple(v.shape)}")

    assert out['dq_pred'].shape == dummy.shape
    assert out['betweenness_decoder'].shape == (4, N)
    assert out['closeness_decoder'].shape == (4, N)
    assert out['betweenness_head'].shape == (4, N)
    assert out['closeness_head'].shape == (4, N)
    print(f"\nShape checks passed for case{N}.")

    # Sanity-check the CentralityLayer against a manual sum on a known matrix
    layer = CentralityLayer()
    test_D = torch.arange(N * N, dtype=torch.float32).reshape(1, 1, N, N)
    b, c_ = layer(test_D)
    expected_betweenness = test_D.squeeze().sum(dim=0)  # column sums
    expected_closeness   = test_D.squeeze().sum(dim=1)  # row sums
    assert torch.allclose(b.squeeze(), expected_betweenness)
    assert torch.allclose(c_.squeeze(), expected_closeness)
    print("CentralityLayer verified against manual row/column sums.")

    # Compare param count against plain AttentionUNet
    try:
        from attention_unet_model import AttentionUNet
        baseline = AttentionUNet(base_channels=32)
        baseline_params = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
        print(f"\nAttentionUNet parameters:        {baseline_params:,}")
        print(f"CentralityAwareUNet parameters:  {total_params:,}")
        print(f"Overhead: +{total_params - baseline_params:,} params "
              f"({100*(total_params/baseline_params - 1):.1f}% more)")
    except ImportError:
        pass
