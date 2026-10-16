"""
Pure Transformer models for reactive dependency matrix (DQ) estimation
from the DC dependency matrix (Ddc). No convolutions anywhere.

Input:  (B, 1, N, N)  — Ddc matrix   (N = number of buses, e.g. 14, 30, 57, 118)
Output: (B, 1, N, N)  — DQ matrix estimate

Two models are provided:

  1. DQTransformer
     Treats every one of the N*N matrix entries as a token (scalar -> d_model
     via a linear embedding), adds learned 2D (row, col) positional
     embeddings, runs a standard Transformer encoder stack with *global*
     self-attention over all N*N tokens, then projects each token back to a
     scalar and reshapes to (B, 1, N, N). This is the direct, no-frills
     "attention is all you need" replacement for the CNN U-Net.

  2. AxialTransformerDQNet
     Same tokenization, but instead of one big global-attention stack it
     alternates ROW self-attention (each row's N tokens attend to each
     other) and COLUMN self-attention (each column's N tokens attend to
     each other), Transformer-style (LayerNorm -> MHA -> residual,
     LayerNorm -> FFN -> residual), stacked for `depth` blocks.
     This mirrors the paper's centrality definitions directly:
         Betweenness C_b(k) = sum_s D(s,k)   <- a COLUMN quantity
         Closeness   C_c(s) = sum_t D(s,t)   <- a ROW quantity
     so giving the network explicit row-only and column-only attention
     (with NO convolutions at all, unlike centrality_unet_model.py's
     AxialAttention which still sits inside a CNN) is structurally aligned
     with what the model ultimately needs to get right. This is the
     recommended model if you want the same "axial" inductive bias as
     centrality_unet_model.py but with zero CNN layers.

  3. CentralityAwareTransformer
     AxialTransformerDQNet + an auxiliary FC centrality head (predicting
     betweenness/closeness straight from pooled tokens) + the same
     differentiable CentralityLayer used by centrality_unet_model.py.
     Returns the *same* output dict keys as CentralityAwareUNet
     ('dq_pred', 'betweenness_decoder', 'closeness_decoder',
     'betweenness_head', 'closeness_head'), so it is a drop-in replacement
     for CentralityAwareUNet and works unmodified with centrality_loss.py
     (CentralityAwareLoss / compute_full_metrics).

All three models accept any square N (14/30/57/118 buses) via `n_buses`.
"""

import math
import torch
import torch.nn as nn


class CentralityLayer(nn.Module):
    """Parameter-free layer: betweenness = column sums, closeness = row sums.
    Duplicated from centrality_unet_model.py so this file has zero CNN deps."""

    def forward(self, D):
        D = D.squeeze(1)
        betweenness = D.sum(dim=1)
        closeness   = D.sum(dim=2)
        return betweenness, closeness


# ---------------------------------------------------------------------------
# Shared bits
# ---------------------------------------------------------------------------

class ScalarTokenEmbedding(nn.Module):
    """Embeds each scalar matrix entry into a d_model vector, plus learned
    row and column positional embeddings (added, not concatenated)."""

    def __init__(self, d_model: int, n_buses: int):
        super().__init__()
        self.n = n_buses
        self.value_proj = nn.Linear(1, d_model)
        self.row_pos = nn.Parameter(torch.randn(n_buses, d_model) * 0.02)
        self.col_pos = nn.Parameter(torch.randn(n_buses, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, N, N)
        Returns:
            tokens: (B, N, N, d_model)
        """
        B, _, N, _ = x.shape
        tok = self.value_proj(x.permute(0, 2, 3, 1))       # (B, N, N, d_model)
        tok = tok + self.row_pos[None, :, None, :] + self.col_pos[None, None, :, :]
        return tok


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class OutputProjection(nn.Module):
    """Projects (B, N, N, d_model) tokens back to a (B, 1, N, N) matrix."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.proj(tokens)                # (B, N, N, 1)
        out = out.permute(0, 3, 1, 2)           # (B, 1, N, N)
        return torch.sigmoid(out)


# ---------------------------------------------------------------------------
# 1. Global-attention transformer (flatten the whole matrix into N*N tokens)
# ---------------------------------------------------------------------------

class DQTransformer(nn.Module):
    """
    Plain Transformer encoder over all N*N matrix entries, no CNN anywhere.

    Every cell (i, j) of Ddc is one token. Standard multi-head self-attention
    lets every token attend to every other token directly (global receptive
    field from layer 1, unlike the CNN U-Net which needs several conv+pool
    stages to get there).
    """

    def __init__(
        self,
        n_buses:    int   = 14,
        d_model:    int   = 128,
        depth:      int   = 6,
        num_heads:  int   = 8,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.embed = ScalarTokenEmbedding(d_model, n_buses)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = OutputProjection(d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the final (B, N, N, d_model) token grid (used by the
        Vmag-head variant to pool features into a per-bus vector)."""
        B, _, N, _ = x.shape
        tokens = self.embed(x).reshape(B, N * N, -1)   # (B, N*N, d_model)
        tokens = self.encoder(tokens)                   # (B, N*N, d_model)
        return tokens.reshape(B, N, N, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, N, N) normalized Ddc matrix
        Returns:
            (B, 1, N, N) estimated DQ matrix
        """
        tokens = self.encode(x)
        return self.out(tokens)


# ---------------------------------------------------------------------------
# 2. Axial transformer (row self-attention <-> column self-attention)
# ---------------------------------------------------------------------------

class AxialSelfAttentionBlock(nn.Module):
    """
    One row-attention sub-block followed by one column-attention sub-block,
    each a full Transformer block (pre-LN -> MHA -> residual,
    pre-LN -> FFN -> residual). Pure nn.MultiheadAttention, no convolutions.
    """

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.row_norm1 = nn.LayerNorm(d_model)
        self.row_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.row_norm2 = nn.LayerNorm(d_model)
        self.row_ff    = FeedForward(d_model, mlp_ratio, dropout)

        self.col_norm1 = nn.LayerNorm(d_model)
        self.col_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.col_norm2 = nn.LayerNorm(d_model)
        self.col_ff    = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, N, d_model) -- dim1=row index, dim2=col index"""
        B, N, _, D = x.shape

        # --- row attention: each row's N tokens (varying column) attend to each other ---
        r = x.reshape(B * N, N, D)                       # (B*N_rows, N_cols, D)
        r = r + self.row_attn(self.row_norm1(r), self.row_norm1(r), self.row_norm1(r))[0]
        r = r + self.row_ff(self.row_norm2(r))
        x = r.reshape(B, N, N, D)

        # --- column attention: each column's N tokens (varying row) attend to each other ---
        c = x.permute(0, 2, 1, 3).reshape(B * N, N, D)    # (B*N_cols, N_rows, D)
        c = c + self.col_attn(self.col_norm1(c), self.col_norm1(c), self.col_norm1(c))[0]
        c = c + self.col_ff(self.col_norm2(c))
        x = c.reshape(B, N, N, D).permute(0, 2, 1, 3)

        return x


class AxialTransformerDQNet(nn.Module):
    """
    Stack of AxialSelfAttentionBlock. No convolutions, no pooling.

    Row attention aligns with closeness (row sums), column attention aligns
    with betweenness (column sums) -- see module docstring.
    """

    def __init__(
        self,
        n_buses:    int   = 14,
        d_model:    int   = 96,
        depth:      int   = 4,
        num_heads:  int   = 6,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.d_model = d_model
        self.embed = ScalarTokenEmbedding(d_model, n_buses)
        self.blocks = nn.ModuleList([
            AxialSelfAttentionBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.out = OutputProjection(d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the final (B, N, N, d_model) token grid (used by the
        centrality-aware subclass to pool features for the FC head)."""
        tokens = self.embed(x)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(x)
        return self.out(tokens)


# ---------------------------------------------------------------------------
# 2b. Axial transformer, per-bus vector head (for Vmag instead of a matrix)
# ---------------------------------------------------------------------------

class VmagHead(nn.Module):
    """
    Projects the (B, N, N, d_model) token grid down to one scalar per bus.

    Each bus corresponds to one ROW of the token grid (row i = bus i's
    dependency values against every other bus), so we mean-pool each row
    across the column dimension to get a (B, N, d_model) per-bus feature,
    then project to a scalar. No sigmoid: Vmag is a per-unit voltage
    magnitude (typically ~0.9-1.1), not a value normalized to [0, 1] the
    way DQ is assumed to be.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, N, d_model) -> (B, N) one scalar per bus (row)."""
        row_repr = tokens.mean(dim=2)          # (B, N, d_model)
        out = self.proj(self.norm(row_repr))   # (B, N, 1)
        return out.squeeze(-1)                 # (B, N)


class AxialTransformerVmagNet(nn.Module):
    """
    Same axial (row/col) transformer backbone as AxialTransformerDQNet
    (reused via composition, so the two stay in sync automatically), but
    predicts a per-bus vector target (e.g. Vmag) instead of an (N, N)
    matrix. Output shape: (B, n_buses) -- matches DependencyMatrixDataset's
    (N, n_buses) shape when loaded with target_key='Vmag'.
    """

    def __init__(
        self,
        n_buses:    int   = 14,
        d_model:    int   = 96,
        depth:      int   = 4,
        num_heads:  int   = 6,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.backbone = AxialTransformerDQNet(
            n_buses=n_buses, d_model=d_model, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
        )
        self.vmag_head = VmagHead(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, N, N) normalized Ddc matrix
        Returns:
            (B, n_buses) estimated Vmag vector
        """
        tokens = self.backbone.encode(x)   # (B, N, N, d_model), via the shared axial backbone
        return self.vmag_head(tokens)      # (B, n_buses)


class DQTransformerVmagNet(nn.Module):
    """
    Same global-attention backbone as DQTransformer, but predicts a
    per-bus vector target (e.g. Vmag) instead of an (N, N) matrix, using
    the same VmagHead (row-wise pooling) as AxialTransformerVmagNet.
    Output shape: (B, n_buses).
    """

    def __init__(
        self,
        n_buses:    int   = 14,
        d_model:    int   = 128,
        depth:      int   = 6,
        num_heads:  int   = 8,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_buses = n_buses
        self.backbone = DQTransformer(
            n_buses=n_buses, d_model=d_model, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
        )
        self.vmag_head = VmagHead(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, N, N) normalized Ddc matrix
        Returns:
            (B, n_buses) estimated Vmag vector
        """
        tokens = self.backbone.encode(x)   # (B, N, N, d_model), via the shared global-attention backbone
        return self.vmag_head(tokens)      # (B, n_buses)


# ---------------------------------------------------------------------------
# 3. Centrality-aware axial transformer (drop-in for CentralityAwareUNet)
# ---------------------------------------------------------------------------

class CentralityHead(nn.Module):
    """FC branch predicting betweenness/closeness from pooled token features.
    Same shape contract as centrality_unet_model.CentralityHead."""

    def __init__(self, d_model: int, num_buses: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.betweenness_out = nn.Linear(hidden, num_buses)
        self.closeness_out   = nn.Linear(hidden, num_buses)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(pooled)
        return self.betweenness_out(h), self.closeness_out(h)


class CentralityAwareTransformer(nn.Module):
    """
    Pure-transformer counterpart of CentralityAwareUNet.

    Pipeline:
        Ddc -> token embedding -> axial (row/col) transformer blocks
            -> dq_pred (linear projection + sigmoid)
            -> betweenness_decoder / closeness_decoder = CentralityLayer(dq_pred)
        pooled tokens (mean over N*N) -> CentralityHead
            -> betweenness_head / closeness_head

    Output dict keys match CentralityAwareUNet exactly, so this is a
    drop-in replacement usable directly with centrality_loss.py's
    CentralityAwareLoss / compute_full_metrics without any changes there.
    """

    def __init__(
        self,
        n_buses:    int   = 14,
        d_model:    int   = 96,
        depth:      int   = 4,
        num_heads:  int   = 6,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.1,
        num_buses:  int   = None,   # alias kept for API parity with CentralityAwareUNet
    ):
        super().__init__()
        if num_buses is not None:
            n_buses = num_buses
        self.n_buses = n_buses

        self.backbone = AxialTransformerDQNet(
            n_buses=n_buses, d_model=d_model, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
        )
        self.centrality_layer = CentralityLayer()
        self.centrality_head  = CentralityHead(d_model, n_buses, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 1, N, N) normalized Ddc matrix
        Returns:
            dict with the same 5 keys as CentralityAwareUNet.forward()
        """
        B = x.shape[0]
        tokens = self.backbone.encode(x)                # (B, N, N, d_model)

        pooled = tokens.mean(dim=(1, 2))                 # (B, d_model) global pool
        betweenness_head, closeness_head = self.centrality_head(pooled)

        dq_pred = self.backbone.out(tokens)              # (B, 1, N, N)
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

    dummy = torch.rand(4, 1, N, N)

    print(f"=== case{N} sanity checks ===\n")

    for name, model in [
        ("DQTransformer", DQTransformer(n_buses=N)),
        ("AxialTransformerDQNet", AxialTransformerDQNet(n_buses=N)),
    ]:
        out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert out.shape == dummy.shape, f"{name} shape mismatch!"
        print(f"{name:>22}: params={n_params:,}  out_shape={tuple(out.shape)}")

    for name, model in [
        ("DQTransformerVmagNet", DQTransformerVmagNet(n_buses=N)),
        ("AxialTransformerVmagNet", AxialTransformerVmagNet(n_buses=N)),
    ]:
        out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert out.shape == (4, N), f"{name} shape mismatch! got {tuple(out.shape)}"
        print(f"{name:>22}: params={n_params:,}  out_shape={tuple(out.shape)}")

    model = CentralityAwareTransformer(n_buses=N)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = model(dummy)
    print(f"\nCentralityAwareTransformer: params={n_params:,}")
    for k, v in out.items():
        print(f"  {k:>22}: {tuple(v.shape)}")
    assert out['dq_pred'].shape == dummy.shape
    assert out['betweenness_decoder'].shape == (4, N)
    assert out['closeness_decoder'].shape == (4, N)
    assert out['betweenness_head'].shape == (4, N)
    assert out['closeness_head'].shape == (4, N)
    print(f"\nShape checks passed for case{N}.")
