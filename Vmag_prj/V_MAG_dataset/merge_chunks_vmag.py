"""
merge_chunks_vmag.py

Companion to merge_chunks.py for chunks produced by data_generation_vmag.py
(i.e. chunks that contain a 'Vmag' array of shape (n_bus, k) alongside
Ddc / DQ / DP of shape (n_bus, n_bus, k)).

Uses the IDENTICAL manifest fields, scenario counting, and shuffle/seed
logic as merge_chunks.py (train_ratio/val_ratio from manifest, seed =
manifest['seed'] + 1) so the train/val/test split is exactly the same
split merge_chunks.py would produce on the Ddc/DQ/DP side — the only
difference is Vmag is carried along too, indexed by the same
global scenario index, so per-scenario alignment is preserved end to end.

Output files (in addition to the usual dataset_*.npz):
    dataset_train.npz  — X (Ddc), y (DQ), Vmag
    dataset_val.npz
    dataset_test.npz
    Vmag_merged.npz    (if save_merged=True), alongside Ddc/DQ/DP_merged.npz

Usage:
    python merge_chunks_vmag.py --chunks_dir chunksV118 --out_dir dataV118
"""

import argparse
import json
import os

import numpy as np
from tqdm import tqdm


def _load_manifest(chunks_dir: str) -> dict:
    manifest_path = os.path.join(chunks_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No manifest.json found in '{chunks_dir}'.\n"
            f"Run data_generation_vmag.py --out_dir {chunks_dir} first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    if not manifest.get('has_vmag', False):
        raise ValueError(
            f"'{manifest_path}' was not produced by data_generation_vmag.py "
            f"(no 'has_vmag' flag / no Vmag arrays in its chunks). Use "
            f"merge_chunks.py for plain Ddc/DQ/DP chunks."
        )
    return manifest


def _chunk_paths(chunks_dir: str, manifest: dict) -> list:
    paths = [os.path.join(chunks_dir, fname) for fname in manifest['chunk_files']]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing chunk files:\n" + "\n".join(missing))
    return paths


def _total_scenarios(paths: list) -> int:
    total = 0
    for p in paths:
        with np.load(p) as d:
            total += d['Ddc'].shape[2]
    return total


def merge_chunks_vmag(
    chunks_dir:    str   = "chunksV",
    out_dir:       str   = ".",
    train_ratio:   float = None,
    val_ratio:     float = None,
    seed:          int   = None,
    delete_chunks: bool  = False,
    save_merged:   bool  = True,
):
    os.makedirs(out_dir, exist_ok=True)

    manifest = _load_manifest(chunks_dir)
    paths    = _chunk_paths(chunks_dir, manifest)

    train_ratio = train_ratio if train_ratio is not None else manifest['train_ratio']
    val_ratio   = val_ratio   if val_ratio   is not None else manifest['val_ratio']
    # Same rule as merge_chunks.py: seed = manifest seed + 1.
    seed        = seed        if seed        is not None else manifest['seed'] + 1

    n_bus = manifest['n_bus']
    print(f"Case: {manifest['case_num']}-bus  |  n_bus={n_bus}  |  Chunks: {len(paths)}")

    print("Counting scenarios across chunks...")
    n_total = _total_scenarios(paths)
    print(f"Total scenarios: {n_total}")

    # --- Identical shuffle logic to merge_chunks.py ---
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    n_train = int(np.floor(n_total * train_ratio))
    n_val   = int(np.floor(n_total * val_ratio))
    n_test  = n_total - n_train - n_val

    train_set = set(perm[:n_train].tolist())
    val_set   = set(perm[n_train:n_train + n_val].tolist())
    test_set  = set(perm[n_train + n_val:].tolist())

    print(f"Split: train={n_train}, val={n_val}, test={n_test}\n")

    splits = {
        'train': (train_set, n_train),
        'val':   (val_set,   n_val),
        'test':  (test_set,  n_test),
    }

    for split_name, (split_set, split_size) in splits.items():
        print(f"Building {split_name} split ({split_size} scenarios)...")

        X_split    = np.zeros((n_bus, n_bus, split_size), dtype=np.float64)
        y_split    = np.zeros((n_bus, n_bus, split_size), dtype=np.float64)
        DP_split   = np.zeros((n_bus, n_bus, split_size), dtype=np.float64)
        Vmag_split = np.zeros((n_bus, split_size),         dtype=np.float64)

        write_ptr  = 0
        global_idx = 0

        for chunk_path in tqdm(paths, desc=f"  Chunks ({split_name})", leave=False):
            with np.load(chunk_path) as chunk:
                Ddc_c  = chunk['Ddc']
                DQ_c   = chunk['DQ']
                DP_c   = chunk['DP']
                Vmag_c = chunk['Vmag']          # (n_bus, k)
                k = Ddc_c.shape[2]

                for local_i in range(k):
                    if global_idx in split_set:
                        X_split[:, :, write_ptr]  = Ddc_c[:, :, local_i]
                        y_split[:, :, write_ptr]  = DQ_c[:, :, local_i]
                        DP_split[:, :, write_ptr] = DP_c[:, :, local_i]
                        Vmag_split[:, write_ptr]  = Vmag_c[:, local_i]
                        write_ptr += 1
                    global_idx += 1

        assert write_ptr == split_size, \
            f"Expected {split_size} scenarios in {split_name}, got {write_ptr}"

        split_path = os.path.join(out_dir, f"dataset_{split_name}.npz")
        np.savez_compressed(split_path, X=X_split, y=y_split, Vmag=Vmag_split)
        print(f"  \u2192 {split_path}  (X: {X_split.shape}, Vmag: {Vmag_split.shape})")

        del X_split, y_split, DP_split, Vmag_split

    if save_merged:
        print("\nBuilding merged arrays (Ddc, DQ, DP, Vmag)...")
        Ddc_all  = np.zeros((n_bus, n_bus, n_total), dtype=np.float64)
        DQ_all   = np.zeros((n_bus, n_bus, n_total), dtype=np.float64)
        DP_all   = np.zeros((n_bus, n_bus, n_total), dtype=np.float64)
        Vmag_all = np.zeros((n_bus, n_total),         dtype=np.float64)

        write_ptr = 0
        for chunk_path in tqdm(paths, desc="  Chunks (merged)"):
            with np.load(chunk_path) as chunk:
                k = chunk['Ddc'].shape[2]
                Ddc_all[:, :, write_ptr:write_ptr + k]  = chunk['Ddc']
                DQ_all[:, :,  write_ptr:write_ptr + k]  = chunk['DQ']
                DP_all[:, :,  write_ptr:write_ptr + k]  = chunk['DP']
                Vmag_all[:,   write_ptr:write_ptr + k]  = chunk['Vmag']
                write_ptr += k

        np.savez_compressed(os.path.join(out_dir, "Ddc_merged.npz"),  Ddc=Ddc_all)
        np.savez_compressed(os.path.join(out_dir, "DQ_merged.npz"),   DQ=DQ_all)
        np.savez_compressed(os.path.join(out_dir, "DP_merged.npz"),   DP=DP_all)
        np.savez_compressed(os.path.join(out_dir, "Vmag_merged.npz"), Vmag=Vmag_all)
        print(f"  Merged arrays saved. Ddc shape: {Ddc_all.shape}, Vmag shape: {Vmag_all.shape}")
        del Ddc_all, DQ_all, DP_all, Vmag_all

    if delete_chunks:
        print("\nDeleting chunk files...")
        for p in paths:
            os.remove(p)
        manifest_path = os.path.join(chunks_dir, 'manifest.json')
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        try:
            os.rmdir(chunks_dir)
            print(f"Removed chunk directory '{chunks_dir}'.")
        except OSError:
            print(f"Chunk files deleted. Directory '{chunks_dir}' kept (not empty).")

    print("\nDone. Output files:")
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"  {f:<30}  {size_mb:>8.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Vmag-enabled chunk .npz files into train/val/test splits"
    )
    parser.add_argument("--chunks_dir",    type=str,   default="chunksV")
    parser.add_argument("--out_dir",       type=str,   default=".")
    parser.add_argument("--train_ratio",   type=float, default=None)
    parser.add_argument("--val_ratio",     type=float, default=None)
    parser.add_argument("--seed",          type=int,   default=None)
    parser.add_argument("--delete_chunks", action="store_true")
    parser.add_argument("--no_merged",     action="store_true")
    args = parser.parse_args()

    merge_chunks_vmag(
        chunks_dir    = args.chunks_dir,
        out_dir       = args.out_dir,
        train_ratio   = args.train_ratio,
        val_ratio     = args.val_ratio,
        seed          = args.seed,
        delete_chunks = args.delete_chunks,
        save_merged   = not args.no_merged,
    )


if __name__ == "__main__":
    main()
