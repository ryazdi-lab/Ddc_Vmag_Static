"""
Loss function and evaluation metrics for DQ estimation.

Loss   : Combined MSE + MAE  (equal weighting, configurable)
Metrics: R², RMSE, MAE       (computed element-wise over flattened predictions)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CombinedMSEMAELoss(nn.Module):
    """
    L = alpha * MSE + (1 - alpha) * MAE

    MSE drives the optimizer to penalize large errors heavily (→ lower RMSE).
    MAE keeps the loss robust to outliers and directly targets MAE.
    Default alpha=0.5 gives equal weight to both terms.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self.alpha = alpha
        self.mse   = nn.MSELoss()
        self.mae   = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.mae(pred, target)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    pred:   torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """
    Compute R², RMSE, and MAE between predictions and targets.

    Inputs are flattened before computation so the metrics are
    computed over all matrix elements and all samples jointly.

    Args:
        pred:   model output,   any shape (B, 1, 14, 14) or (N,)
        target: ground truth,   same shape as pred

    Returns:
        dict with keys 'r2', 'rmse', 'mae'
    """
    pred   = pred.detach().float().flatten()
    target = target.detach().float().flatten()

    # MAE
    mae = torch.mean(torch.abs(pred - target)).item()

    # RMSE
    mse  = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse).item()

    # R²  =  1 - SS_res / SS_tot
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    # Guard against degenerate case where target is constant
    r2 = (1.0 - ss_res / (ss_tot + 1e-10)).item()

    return {'r2': r2, 'rmse': rmse, 'mae': mae}


def aggregate_metrics(metric_list: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of per-batch metric dicts into a single dict."""
    keys = metric_list[0].keys()
    return {k: sum(m[k] for m in metric_list) / len(metric_list) for k in keys}


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    criterion = CombinedMSEMAELoss(alpha=0.5)

    pred   = torch.rand(8, 1, 14, 14)
    target = torch.rand(8, 1, 14, 14)

    loss    = criterion(pred, target)
    metrics = compute_metrics(pred, target)

    print(f"Combined loss : {loss.item():.6f}")
    print(f"R²            : {metrics['r2']:.4f}")
    print(f"RMSE          : {metrics['rmse']:.6f}")
    print(f"MAE           : {metrics['mae']:.6f}")

    # Perfect prediction should give R²=1, RMSE=0, MAE=0
    perfect = compute_metrics(target, target)
    print(f"\nPerfect prediction: {perfect}")
