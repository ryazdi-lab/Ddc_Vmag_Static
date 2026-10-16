"""
data_generation.py

Generates the dataset of (Ddc, DQ, DP) dependency matrix triples for a
chosen IEEE test case.

Memory-safe design — solves the OOM error for large cases (case118, case300):
  Instead of pre-allocating the entire dataset in RAM, scenarios are generated
  in small chunks and written directly to disk as .npz files. A separate
  merge_chunks.py script then reads them back, shuffles, splits into
  train/val/test, and deletes the chunk files.

  Memory used at any time = chunk_size × (per-scenario footprint), not
  n_scenarios × footprint.  For case118 with chunk_size=200, this is
  ~340 MB peak instead of 18+ GB.

Usage:
    python data_generation.py                         # case14, 11050 scenarios
    python data_generation.py --case 118              # IEEE 118-bus
    python data_generation.py --case 118 --chunk_size 200
    python data_generation.py --n_scenarios 500       # quick test
    python data_generation.py --n_workers 4           # parallel workers
    python data_generation.py --out_dir ./chunks118   # output folder
    python data_generation.py --help

After running, call:
    python merge_chunks.py --chunks_dir ./chunks118 --out_dir ./data118
"""

import argparse
import copy
import os
import warnings

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from bdep             import bdep
from power_flow_utils import (
    build_adj_P, build_adj_Q,
    build_flow_matrix_P, build_flow_matrix_Q,
)


# ---------------------------------------------------------------------------
# Automatic safe chunk-size per case
# ---------------------------------------------------------------------------

# Per-scenario memory (bytes): 3 × n_bus² + 2 × h_P² + h_Q²  (float64)
# We target ≤ 512 MB per chunk.

_TARGET_MB = 4

def _auto_chunk_size(n_bus: int, h_P: int, h_Q: int) -> int:
    bytes_per = (n_bus**2 * 3 + h_P**2 * 2 + h_Q**2) * 8
    mb_per    = bytes_per / 1e6
    return max(1, int(_TARGET_MB / mb_per))


# ---------------------------------------------------------------------------
# Test case loader
# ---------------------------------------------------------------------------

CASE_MAP = {
    5:   pn.case5,
    14:  pn.case14,
    30:  pn.case_ieee30,
    39:  pn.case39,
    57:  pn.case57,
    89:  pn.case89pegase,
    118: pn.case118,
    300: pn.case300,
}


def load_case(case_num: int):
    if case_num not in CASE_MAP:
        raise ValueError(f"Unsupported case: {case_num}. Choose from {list(CASE_MAP.keys())}")
    return CASE_MAP[case_num]()


# ---------------------------------------------------------------------------
# Random perturbation
# ---------------------------------------------------------------------------

def perturb_network(net_original, std: float = 0.3, prob: float = 0.5, seed=None):
    """Deep-copy and randomly perturb loads/generators, matching MATLAB logic."""
    rng = np.random.default_rng(seed)
    net = copy.deepcopy(net_original)

    for idx in net.load.index:
        if rng.random() < prob:
            net.load.at[idx, 'p_mw']    *= (1 + rng.standard_normal() * std)
        if rng.random() < prob:
            net.load.at[idx, 'q_mvar']  *= (1 + rng.standard_normal() * std)

    for idx in net.gen.index:
        if rng.random() < prob:
            net.gen.at[idx, 'p_mw']     *= (1 + rng.standard_normal() * std)

    return net


# ---------------------------------------------------------------------------
# Single-scenario pipeline  (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def run_scenario(args_tuple):
    """
    Worker entry point. Returns a flat dict of arrays, or None on non-convergence.
    """
    net_base, adj_P_base, adj_Q_base, n_bus, rng_seed = args_tuple

    net = perturb_network(net_base, seed=rng_seed)

    # AC power flow
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp.runpp(net, algorithm='nr', numba=False, max_iteration=50)
        if not net.converged:
            return None
    except Exception:
        return None

    # DC power flow
    net_dc = copy.deepcopy(net)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp.rundcpp(net_dc, numba=False)
        if not net_dc.converged:
            return None
    except Exception:
        return None

    FlowP  = build_flow_matrix_P(net)
    FlowQ  = build_flow_matrix_Q(net)
    FlowDC = build_flow_matrix_P(net_dc)

    def safe_norm(M):
        mx = np.max(np.abs(M))
        return M / mx if mx > 0 else M

    h_P = adj_P_base.shape[0]
    h_Q = adj_Q_base.shape[0]

    return {
        'DP':   safe_norm(bdep(adj_P_base[:n_bus, :n_bus], FlowP[:n_bus,  :n_bus])),
        'DQ':   safe_norm(bdep(adj_P_base[:n_bus, :n_bus], FlowQ[:n_bus,  :n_bus])),
        'Ddc':  safe_norm(bdep(adj_P_base[:n_bus, :n_bus], FlowDC[:n_bus, :n_bus])),
        'DHP':  safe_norm(bdep(adj_P_base,                 FlowP[:h_P,    :h_P])),
        'DHQ':  safe_norm(bdep(adj_Q_base,                 FlowQ[:h_Q,    :h_Q])),
        'DHdc': safe_norm(bdep(adj_P_base,                 FlowDC[:h_P,   :h_P])),
    }


# ---------------------------------------------------------------------------
# Chunk writer  (single scenario — immediate save)
# ---------------------------------------------------------------------------

def _write_chunk(chunk_results: list, chunk_idx: int, out_dir: str,
                 n_bus: int, h_P: int, h_Q: int, save_extras: bool):
    """
    Pack a list of per-scenario dicts into arrays and write one .npz chunk.
    Only the fields needed by merge_chunks.py are written.
    """
    DP   = np.stack([r['DP']  for r in chunk_results], axis=2)   # (n,n,k)
    DQ   = np.stack([r['DQ']  for r in chunk_results], axis=2)
    Ddc  = np.stack([r['Ddc'] for r in chunk_results], axis=2)

    path = os.path.join(out_dir, f"chunk_{chunk_idx:05d}.npz")

    save_kwargs = dict(DP=DP, DQ=DQ, Ddc=Ddc)

    if save_extras:
        save_kwargs['DHP']  = np.stack([r['DHP']  for r in chunk_results], axis=2)
        save_kwargs['DHQ']  = np.stack([r['DHQ']  for r in chunk_results], axis=2)
        save_kwargs['DHdc'] = np.stack([r['DHdc'] for r in chunk_results], axis=2)

    # np.savez automatically appends '.npz' to whatever path you give it.
    # So we use a temp stem (no extension) and let NumPy produce <stem>.npz,
    # then rename to the real destination — atomic on both Windows and Linux.
    # A crash mid-write leaves only a <stem>.npz file that is ignored on resume.
    tmp_stem = os.path.join(out_dir, f"chunk_{chunk_idx:05d}_tmp")
    np.savez(tmp_stem, **save_kwargs)        # writes tmp_stem + ".npz"
    os.replace(tmp_stem + ".npz", path)
    return path


def _write_manifest_atomic(manifest: dict, out_dir: str):
    """Write manifest.json atomically via a temp file + rename."""
    import json
    manifest_path = os.path.join(out_dir, 'manifest.json')
    tmp_path      = manifest_path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, manifest_path)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_existing_manifest(out_dir: str) -> dict | None:
    """Return the manifest dict if one exists in out_dir, else None."""
    import json
    manifest_path = os.path.join(out_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as f:
        return json.load(f)


def _resume_state(out_dir: str, seed: int) -> tuple[int, int, int, list, object]:
    """
    Inspect out_dir for completed chunk files and return resume state.

    Returns:
        collected   : number of scenarios already saved
        chunk_idx   : index for the NEXT chunk to write
        rng_advance : how many RNG draws to skip (attempts already made)
        chunk_paths : list of already-written chunk paths
        manifest    : existing manifest dict (or None)
    """
    manifest = _load_existing_manifest(out_dir)
    if manifest is None:
        return 0, 0, 0, [], None

    chunk_files = manifest.get('chunk_files', [])
    chunk_paths = []
    collected   = 0

    for fname in chunk_files:
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            # File listed in manifest but missing — stop here, don't trust
            # anything after this gap.
            print(f"  ⚠ Chunk file missing: {fname}  — resuming up to previous chunk.")
            break
        try:
            with np.load(fpath) as d:
                k = d['Ddc'].shape[2]
            chunk_paths.append(fpath)
            collected += k
        except Exception as e:
            print(f"  ⚠ Could not read {fname}: {e}  — stopping resume here.")
            break

    chunk_idx = len(chunk_paths)

    # We can't reliably know how many RNG draws were *attempted* (failed
    # scenarios also consume draws).  We store attempts in the manifest so
    # we can advance the RNG to exactly the same position.
    rng_advance = manifest.get('attempts_so_far', collected)

    print(f"  Resuming: found {chunk_idx} complete chunks, "
          f"{collected} scenarios already saved.")

    return collected, chunk_idx, rng_advance, chunk_paths, manifest


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_dataset(
    case_num:    int   = 14,
    n_scenarios: int   = 11050,
    chunk_size:  int   = None,       # None = auto from memory budget
    train_ratio: float = 10000 / 11050,
    val_ratio:   float = 1000  / 11050,
    seed:        int   = 42,
    n_workers:   int   = 1,
    save_extras: bool  = False,
    out_dir:     str   = "chunks",
):
    """
    Generate scenarios and save them in small .npz chunk files.

    Supports resuming interrupted runs: existing chunk files in out_dir are
    detected automatically and skipped.  Every completed chunk is written to
    disk immediately (atomic rename), and the manifest is updated after each
    chunk so a crash loses at most one chunk worth of work.

    After this runs, call merge_chunks.py to assemble train/val/test splits.
    Chunk files are stored in out_dir and named chunk_00000.npz, chunk_00001.npz …
    """
    import json

    os.makedirs(out_dir, exist_ok=True)

    if n_workers == 0:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, n_workers)

    print(f"Loading IEEE case{case_num}...")
    net_base = load_case(case_num)
    pp.runpp(net_base, numba=False)
    adj_P_base = build_adj_P(net_base)
    adj_Q_base = build_adj_Q(net_base)
    n_bus = len(net_base.bus)
    h_P   = adj_P_base.shape[0]
    h_Q   = adj_Q_base.shape[0]

    # Determine chunk size
    if chunk_size is None:
        chunk_size = _auto_chunk_size(n_bus, h_P, h_Q)

    mb_per_chunk = chunk_size * (n_bus**2 * 3 + h_P**2 * 2 + h_Q**2) * 8 / 1e6
    n_chunks_est = int(np.ceil(n_scenarios / chunk_size))

    print(f"Buses: {n_bus}  |  h_P: {h_P}  |  h_Q: {h_Q}")
    print(f"Chunk size: {chunk_size} scenarios (~{mb_per_chunk:.0f} MB/chunk)")
    print(f"Estimated chunks: {n_chunks_est}")
    print(f"Workers: {n_workers}  |  Target scenarios: {n_scenarios}")
    print(f"Output dir: {out_dir}\n")

    # ----------------------------------------------------------------
    # Resume detection
    # ----------------------------------------------------------------
    print("Checking for existing chunks to resume from...")
    collected, chunk_idx, rng_advance, chunk_paths, prev_manifest = \
        _resume_state(out_dir, seed)

    remaining = n_scenarios - collected
    if remaining <= 0:
        print(f"Already have {collected} scenarios — nothing to do.")
        return

    print(f"Need {remaining} more scenarios.\n")

    # Advance RNG past already-used draws so new seeds don't repeat
    rng          = np.random.default_rng(seed)
    if rng_advance > 0:
        # Draw and discard rng_advance integers to restore position
        rng.integers(0, 2**31, size=rng_advance)

    max_attempts = remaining * 5
    attempts     = 0
    chunk_buf    = []  # buffer of dicts for current in-progress chunk

    pbar = tqdm(total=n_scenarios, initial=collected,
                desc="Scenarios", unit="scenario")

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _build_manifest() -> dict:
        return {
            'case_num':        case_num,
            'n_bus':           n_bus,
            'h_P':             h_P,
            'h_Q':             h_Q,
            'n_scenarios':     collected,
            'chunk_size':      chunk_size,
            'n_chunks':        chunk_idx,
            'train_ratio':     train_ratio,
            'val_ratio':       val_ratio,
            'seed':            seed,
            'save_extras':     save_extras,
            'chunk_files':     [os.path.basename(p) for p in chunk_paths],
            # Store total attempts so far so resume can advance RNG correctly
            'attempts_so_far': (prev_manifest or {}).get('attempts_so_far', 0) + attempts,
        }

    def _flush_chunk():
        """Write the current buffer as a new chunk and update the manifest."""
        nonlocal chunk_idx, collected
        if not chunk_buf:
            return
        p = _write_chunk(chunk_buf, chunk_idx, out_dir,
                         n_bus, h_P, h_Q, save_extras)
        chunk_paths.append(p)
        chunk_idx += 1
        chunk_buf.clear()
        # Update manifest immediately after each chunk is safely on disk
        _write_manifest_atomic(_build_manifest(), out_dir)

    def _store(result):
        nonlocal collected
        chunk_buf.append(result)
        collected += 1
        pbar.update(1)
        pbar.set_postfix({'attempts': attempts,
                          'conv_rate': f"{collected / max(1, attempts):.1%}",
                          'chunks':   chunk_idx})
        if len(chunk_buf) >= chunk_size:
            _flush_chunk()

    # ----------------------------------------------------------------
    # Single-process path
    # ----------------------------------------------------------------
    if n_workers == 1:
        while collected < n_scenarios and attempts < max_attempts:
            seed_i = int(rng.integers(0, 2**31))
            result = run_scenario((net_base, adj_P_base, adj_Q_base, n_bus, seed_i))
            attempts += 1
            if result is not None:
                _store(result)

    # ----------------------------------------------------------------
    # Multi-process path
    # ----------------------------------------------------------------
    else:
        queue_size = n_workers * 2

        def _make_task():
            return (net_base, adj_P_base, adj_Q_base, n_bus,
                    int(rng.integers(0, 2**31)))

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(run_scenario, _make_task()): None
                       for _ in range(min(queue_size, max_attempts))}
            attempts += len(futures)

            while futures and collected < n_scenarios and attempts < max_attempts:
                done_iter = as_completed(futures)
                future    = next(done_iter)
                del futures[future]

                result = future.result()
                if result is not None:
                    _store(result)

                if attempts < max_attempts and collected < n_scenarios:
                    futures[pool.submit(run_scenario, _make_task())] = None
                    attempts += 1

    # Flush any remaining partial chunk
    _flush_chunk()
    pbar.close()

    if collected < n_scenarios:
        print(f"\n⚠ Only collected {collected}/{n_scenarios} converged scenarios.")

    print(f"\nGenerated {collected} scenarios in {attempts} attempts "
          f"(convergence rate: {collected / max(1, attempts):.1%})")
    print(f"Wrote {chunk_idx} chunk files to '{out_dir}/'")

    # Final manifest write
    _write_manifest_atomic(_build_manifest(), out_dir)
    print(f"Manifest written to '{os.path.join(out_dir, 'manifest.json')}'")
    print(f"\nNext step:\n  python merge_chunks.py --chunks_dir {out_dir} --out_dir <output_folder>")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate dependency matrix dataset in memory-safe chunks"
    )
    parser.add_argument("--case",        type=int,   default=14)
    parser.add_argument("--n_scenarios", type=int,   default=11050)
    parser.add_argument("--chunk_size",  type=int,   default=None,
                        help="Scenarios per chunk (default: auto from 512 MB budget)")
    parser.add_argument("--n_workers",   type=int,   default=10,
                        help="0=all CPUs, 1=single-process (default: 1)")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--save_extras", action="store_true")
    parser.add_argument("--out_dir",     type=str,   default="chunks",
                        help="Folder for chunk .npz files (default: chunks/)")
    args = parser.parse_args()

    generate_dataset(
        case_num    = args.case,
        n_scenarios = args.n_scenarios,
        chunk_size  = args.chunk_size,
        n_workers   = args.n_workers,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
        save_extras = args.save_extras,
        out_dir     = args.out_dir,
    )


if __name__ == "__main__":
    main()
