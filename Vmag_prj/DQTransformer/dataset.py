"""
PyTorch Dataset for dependency matrix inputs (Ddc) paired with either
a dependency-matrix target (DQ) or a per-bus vector target (Vmag).

Each .npz file contains:
    X:    (N_bus, N_bus, N)  — Ddc input matrices        (always present)
    y:    (N_bus, N_bus, N)  — DQ target matrices         (matrix-task target)
    Vmag: (N_bus, N)         — per-bus voltage magnitude  (vector-task target)

N_bus (the matrix side length / number of buses) is read directly from the
data, not hardcoded — so the same loader works unmodified for case14, case30,
case57, case118, etc. This matches the generic `n_buses` design used by
transformer_model.py (DQTransformer / AxialTransformerDQNet /
CentralityAwareTransformer), whose outputs are shaped (B, 1, N_bus, N_bus).
train_transformer_*.py builds those models with `n_buses=args.case`, so pass
`expected_n_buses=args.case` here too if you want a fast, clear error instead
of a confusing shape mismatch deep inside the model when --case doesn't
match the .npz file you pointed it at.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DependencyMatrixDataset(Dataset):
    """
    Loads a single split (train / val / test) from an .npz file.

    X is always the (N_bus, N_bus, N) Ddc matrix, reshaped to
    (N, 1, N_bus, N_bus) — standard PyTorch (batch, channel, H, W)
    convention. N_bus is inferred from the data itself.

    The target is selected via `target_key`:
      - 'y'    (default): (N_bus, N_bus, N) DQ matrix   → reshaped to (N, 1, N_bus, N_bus)
      - 'Vmag':            (N_bus, N) per-bus vector      → reshaped to (N, N_bus)

    The dataset auto-detects which reshape to use from the target
    array's number of dimensions, so any similarly-shaped key works
    without extra configuration.
    """

    def __init__(self, npz_path: str, target_key: str = 'y', expected_n_buses: int = None):
        """
        Args:
            npz_path:          path to dataset_train.npz / dataset_val.npz / dataset_test.npz
            target_key:        which array in the .npz to use as the target
                                ('y' for DQ matrix, 'Vmag' for per-bus voltage vector)
            expected_n_buses:  if given, raises a clear error immediately when the loaded
                                matrix's side length doesn't match (e.g. you pointed --case 14
                                at a case118 .npz file), instead of failing later with an
                                opaque shape-mismatch error inside the model.
        """
        data = np.load(npz_path)

        if target_key not in data.files:
            raise KeyError(
                f"target_key={target_key!r} not found in {npz_path}. "
                f"Available keys: {list(data.files)}"
            )

        X_raw = data['X']
        if X_raw.shape[0] != X_raw.shape[1]:
            raise ValueError(f"Expected square Ddc matrices in {npz_path}, got shape {X_raw.shape[:2]}")
        n_buses = X_raw.shape[0]

        if expected_n_buses is not None and n_buses != expected_n_buses:
            raise ValueError(
                f"n_buses mismatch in {npz_path}: data has {n_buses} buses but "
                f"expected_n_buses={expected_n_buses} (check your --case argument "
                f"matches the dataset file you're pointing to)."
            )

        self.n_buses = n_buses

        # (N_bus, N_bus, N) → (N, N_bus, N_bus) → (N, 1, N_bus, N_bus)
        self.X = torch.from_numpy(
            X_raw.transpose(2, 0, 1)[:, np.newaxis, :, :]
        ).float()

        target_raw = data[target_key]
        self.target_key = target_key

        if target_raw.ndim == 3:
            # Matrix target, e.g. DQ: (N_bus, N_bus, N) → (N, 1, N_bus, N_bus)
            if target_raw.shape[0] != target_raw.shape[1]:
                raise ValueError(f"Expected square target matrices in {npz_path}, got shape {target_raw.shape[:2]}")
            self.y = torch.from_numpy(
                target_raw.transpose(2, 0, 1)[:, np.newaxis, :, :]
            ).float()
        elif target_raw.ndim == 2:
            # Per-bus vector target, e.g. Vmag: (N_bus, N) → (N, N_bus)
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
            f"n={len(self)}, n_buses={self.n_buses}, target_key={self.target_key!r}, "
            f"X={tuple(self.X.shape)}, y={tuple(self.y.shape)})"
        )


def get_dataloaders(
    train_path: str,
    val_path:   str,
    test_path:  str,
    batch_size: int  = 64,
    num_workers: int = 0,
    target_key: str  = 'y',
    expected_n_buses: int = None,
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
        expected_n_buses: if given (e.g. pass your --case value), raises a clear
                           error immediately if the loaded data's bus count doesn't
                           match, instead of an opaque shape mismatch inside the model

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = DependencyMatrixDataset(train_path, target_key=target_key, expected_n_buses=expected_n_buses)
    val_ds   = DependencyMatrixDataset(val_path,   target_key=target_key, expected_n_buses=expected_n_buses)
    test_ds  = DependencyMatrixDataset(test_path,  target_key=target_key, expected_n_buses=expected_n_buses)

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=None,
                        help="Expected number of buses, e.g. 14, 30, 57, 118. "
                             "If given, validates it against the loaded .npz files.")
    parser.add_argument("--target_key", type=str, default="y",
                        help="'y' for DQ matrix target, 'Vmag' for per-bus vector target")
    parser.add_argument("--train", type=str, default="dataset_train.npz")
    parser.add_argument("--val",   type=str, default="dataset_val.npz")
    parser.add_argument("--test",  type=str, default="dataset_test.npz")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(
        args.train, args.val, args.test,
        target_key=args.target_key,
        expected_n_buses=args.case,
    )

    X_batch, y_batch = next(iter(train_loader))
    print(f"\nSample batch  — X: {X_batch.shape}, y: {y_batch.shape}")
    print(f"X range: [{X_batch.min():.4f}, {X_batch.max():.4f}]")
    print(f"y range: [{y_batch.min():.4f}, {y_batch.max():.4f}]")
