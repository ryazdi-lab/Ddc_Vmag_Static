"""
PyTorch Dataset for dependency matrix inputs (Ddc) paired with either
a dependency-matrix target (DQ) or a per-bus vector target (Vmag).

Each .npz file contains:
    X:    (14, 14, N)  — Ddc input matrices        (always present)
    y:    (14, 14, N)  — DQ target matrices         (matrix-task target)
    Vmag: (14, N)      — per-bus voltage magnitude  (vector-task target)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DependencyMatrixDataset(Dataset):
    """
    Loads a single split (train / val / test) from an .npz file.

    X is always the (14, 14, N) Ddc matrix, reshaped to (N, 1, 14, 14)
    — standard PyTorch (batch, channel, H, W) convention.

    The target is selected via `target_key`:
      - 'y'    (default): (14, 14, N) DQ matrix    → reshaped to (N, 1, 14, 14)
      - 'Vmag':           (14, N) per-bus vector    → reshaped to (N, 14)

    The dataset auto-detects which reshape to use from the target
    array's number of dimensions, so any similarly-shaped key works
    without extra configuration.
    """

    def __init__(self, npz_path: str, target_key: str = 'y'):
        """
        Args:
            npz_path:   path to dataset_train.npz / dataset_val.npz / dataset_test.npz
            target_key: which array in the .npz to use as the target
                        ('y' for DQ matrix, 'Vmag' for per-bus voltage vector)
        """
        data = np.load(npz_path)

        if target_key not in data.files:
            raise KeyError(
                f"target_key={target_key!r} not found in {npz_path}. "
                f"Available keys: {list(data.files)}"
            )

        # (14, 14, N) → (N, 14, 14) → (N, 1, 14, 14)
        self.X = torch.from_numpy(
            data['X'].transpose(2, 0, 1)[:, np.newaxis, :, :]
        ).float()

        target_raw = data[target_key]
        self.target_key = target_key

        if target_raw.ndim == 3:
            # Matrix target, e.g. DQ: (14, 14, N) → (N, 1, 14, 14)
            self.y = torch.from_numpy(
                target_raw.transpose(2, 0, 1)[:, np.newaxis, :, :]
            ).float()
        elif target_raw.ndim == 2:
            # Per-bus vector target, e.g. Vmag: (14, N) → (N, 14)
            self.y = torch.from_numpy(target_raw.transpose(1, 0)).float()
        else:
            raise ValueError(
                f"Unsupported target array ndim={target_raw.ndim} for key {target_key!r}"
            )

        assert self.X.shape[0] == self.y.shape[0], (
            f"Sample-count mismatch: X={self.X.shape}, y={self.y.shape}"
        )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def __repr__(self) -> str:
        return (
            f"DependencyMatrixDataset("
            f"n={len(self)}, target_key={self.target_key!r}, "
            f"X={tuple(self.X.shape)}, y={tuple(self.y.shape)})"
        )


def get_dataloaders(
    train_path: str,
    val_path:   str,
    test_path:  str,
    batch_size: int  = 64,
    num_workers: int = 0,
    target_key: str  = 'y',
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders for all three splits.

    Args:
        train_path:  path to dataset_train.npz
        val_path:    path to dataset_val.npz
        test_path:   path to dataset_test.npz
        batch_size:  samples per mini-batch
        num_workers: parallel data-loading workers (0 = main process only)
        target_key:  'y' for DQ matrix target, 'Vmag' for per-bus vector target

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = DependencyMatrixDataset(train_path, target_key=target_key)
    val_ds   = DependencyMatrixDataset(val_path,   target_key=target_key)
    test_ds  = DependencyMatrixDataset(test_path,  target_key=target_key)

    print(f"Train : {train_ds}")
    print(f"Val   : {val_ds}")
    print(f"Test  : {test_ds}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    paths = {
        'train': 'dataset_train.npz',
        'val':   'dataset_val.npz',
        'test':  'dataset_test.npz',
    }

    train_loader, val_loader, test_loader = get_dataloaders(**paths)

    X_batch, y_batch = next(iter(train_loader))
    print(f"\nSample batch  — X: {X_batch.shape}, y: {y_batch.shape}")
    print(f"X range: [{X_batch.min():.4f}, {X_batch.max():.4f}]")
    print(f"y range: [{y_batch.min():.4f}, {y_batch.max():.4f}]")
