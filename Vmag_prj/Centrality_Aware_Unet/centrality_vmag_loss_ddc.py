"""
Multi-task loss and metrics for the Centrality-Aware Vmag U-Net — DDC VARIANT.

Combines:
  1. Vmag loss        — Combined MSE+MAE on the predicted per-bus voltage
                         magnitude vector
  2. Head-centrality   — MSE between the auxiliary FC head's direct
                          betweenness/closeness prediction and betweenness/
                          closeness derived from the Ddc INPUT matrix
                          (not a DQ target — this variant needs no DQ data)

Caveat: DC-derived centrality (from Ddc) is a proxy for true reactive
centrality (from DQ). The source paper notes DC and reactive betweenness
can rank buses differently — this variant gives the bottleneck a "DC-flow-
implied importance" inductive bias, which is weaker than supervising
against true DQ-derived centrality, but requires no DQ ground truth at all.
"""

import torch
import torch.nn as nn

from metrics import CombinedMSEMAELoss, compute_metrics
from centrality_unet_model import CentralityLayer

_centrality_layer = CentralityLayer()


def compute_ground_truth_centrality_from_ddc(ddc_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute betweenness and closeness from the Ddc INPUT matrix (not a DQ
    target), using the same row/column-sum definitions as the DQ version:
        Betweenness C_b(k) = sum_s D(s,k)  (column sum)
        Closeness   C_c(s) = sum_t D(s,t)  (row sum)

    Args:
        ddc_input: (B, 1, N, N) Ddc dependency matrix (the model's own input)
    Returns:
        betweenness_gt: (B, N)
        closeness_gt:   (B, N)
    """
    return _centrality_layer(ddc_input)


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------

class CentralityAwareVmagLoss(nn.Module):
    """
    L_total = w_vmag    * (alpha*MSE + (1-alpha)*MAE) on Vmag
            + w_head_c  * MSE on (betweenness_head, closeness_head)

    The head-centrality term is compared against betweenness/closeness
    derived from the Ddc INPUT (via CentralityLayer), so no DQ ground
    truth is required.
    """

    def __init__(
        self,
        alpha:    float = 0.5,   # MSE vs MAE weighting in the Vmag loss
        w_vmag:   float = 1.0,
        w_head_c: float = 0.3,
    ):
        super().__init__()
        self.vmag_loss = CombinedMSEMAELoss(alpha=alpha)
        self.mse       = nn.MSELoss()

        self.w_vmag   = w_vmag
        self.w_head_c = w_head_c

    def forward(
        self,
        outputs:     dict[str, torch.Tensor],
        ddc_input:   torch.Tensor,
        vmag_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            outputs:     dict returned by CentralityAwareVmagUNet.forward()
            ddc_input:   (B, 1, N, N) Ddc matrix — the model's own input,
                         used ONLY to derive ground-truth centrality
            vmag_target: (B, N)       ground-truth per-bus voltage magnitude

        Returns:
            total_loss: scalar tensor (use for .backward())
            breakdown:  dict of float components for logging
        """
        betweenness_gt, closeness_gt = compute_ground_truth_centrality_from_ddc(ddc_input)

        # 1. Main Vmag regression loss
        loss_vmag = self.vmag_loss(outputs['vmag_pred'], vmag_target)

        # 2. Auxiliary FC head centrality loss (shapes bottleneck features)
        loss_head_c = (
            self.mse(outputs['betweenness_head'], betweenness_gt) +
            self.mse(outputs['closeness_head'],   closeness_gt)
        )

        total = self.w_vmag * loss_vmag + self.w_head_c * loss_head_c

        breakdown = {
            'loss_total':   total.item(),
            'loss_vmag':    loss_vmag.item(),
            'loss_head_c':  loss_head_c.item(),
        }

        return total, breakdown


# ---------------------------------------------------------------------------
# Extended metrics: Vmag-level + head-centrality-level
# ---------------------------------------------------------------------------

def compute_vmag_full_metrics(
    outputs:     dict[str, torch.Tensor],
    ddc_input:   torch.Tensor,
    vmag_target: torch.Tensor,
) -> dict[str, float]:
    """
    Compute R², RMSE, MAE on two things:
      - the Vmag vector                                (vmag_r2, vmag_rmse, vmag_mae)
      - FC-head betweenness/closeness, vs Ddc-derived   (head_r2, head_rmse, head_mae)
    """
    betweenness_gt, closeness_gt = compute_ground_truth_centrality_from_ddc(ddc_input)

    vmag_m = compute_metrics(outputs['vmag_pred'], vmag_target)

    head_pred = torch.cat([outputs['betweenness_head'], outputs['closeness_head']], dim=1)
    head_gt   = torch.cat([betweenness_gt, closeness_gt], dim=1)
    head_m    = compute_metrics(head_pred, head_gt)

    return {
        'vmag_r2':   vmag_m['r2'],
        'vmag_rmse': vmag_m['rmse'],
        'vmag_mae':  vmag_m['mae'],
        'head_r2':   head_m['r2'],
        'head_rmse': head_m['rmse'],
        'head_mae':  head_m['mae'],
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from centrality_vmag_model import CentralityAwareVmagUNet

    N = 14
    model = CentralityAwareVmagUNet(base_channels=32, attn_heads=4, num_buses=N)
    criterion = CentralityAwareVmagLoss()

    dummy_input = torch.rand(8, 1, N, N)
    dummy_vmag  = torch.rand(8, N)

    outputs = model(dummy_input)
    loss, breakdown = criterion(outputs, dummy_input, dummy_vmag)

    print("Loss breakdown:")
    for k, v in breakdown.items():
        print(f"  {k:>12}: {v:.6f}")

    metrics = compute_vmag_full_metrics(outputs, dummy_input, dummy_vmag)
    print("\nFull metrics:")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")
