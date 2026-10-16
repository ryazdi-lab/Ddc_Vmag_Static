"""
data_generation_vmag.py — GPU-accelerated dataset generation WITH bus
voltage magnitudes (V_mag) captured alongside DP / DQ / Ddc.

WHY A NEW FILE (READ THIS FIRST)
---------------------------------
The existing pipeline (data_generation_gpu.py -> merge_chunks.py) produces
DP, DQ, Ddc (and optionally DHP/DHQ/DHdc) but never saves V_mag — the
quantity the original MATLAB script captures as:

    V_mag(n,:) = results_ac_case.bus(:, VM).'      % data_generation_D_matrices.m

V_mag CANNOT be reconstructed after the fact from an already-generated
Ddc/DQ/DP dataset:
  1. Ddc/DQ/DP are lossy, non-invertible summaries of the power-flow
     solution (bdep dependency ratios, not voltages) — you cannot invert
     them back into bus voltages.
  2. The existing generation script never persists the per-scenario RNG
     seed, and with n_workers > 1 the ProcessPoolExecutor futures are
     collected via `as_completed`, so scenario order is NOT deterministic
     across runs even with a fixed seed. There is no reliable way to
     re-run with the same seed and land on the same scenario ordering.

So the only correct way to get V_mag aligned with DP/DQ/Ddc is to capture
it INSIDE the same power-flow call that produces the Flow matrices those
are derived from, and write it into the SAME chunk entry, at the SAME
local index. That is what this file does.

This script is a drop-in superset of data_generation_gpu.py: same CLI,
same DP/DQ/Ddc (+ optional DHP/DHQ/DHdc) outputs, PLUS a 'Vmag' array of
shape (n_bus, k) per chunk, guaranteed scenario-aligned with the rest.

It deliberately does NOT reuse the private chunk/manifest/resume helpers
from data_generation.py (_write_chunk, _write_manifest_atomic,
_resume_state), since their internal schema is specific to
DP/DQ/Ddc(+extras) and adding a new key to their dicts is not something
that can be verified without reading that source. Instead this file
implements its own small, self-contained, equivalent versions — same
manifest fields merge_chunks.py already expects (case_num, n_bus,
chunk_files, train_ratio, val_ratio, seed), so the *existing*
merge_chunks.py still works unmodified on Ddc/DQ/DP if you only want
those; use merge_chunks_vmag.py (companion file) if you also want Vmag
pulled into the merged/split datasets.

Usage:
    python data_generation_vmag.py --case 118 --n_scenarios 11050 --n_workers 20
    python data_generation_vmag.py --case 118 --float32   # faster on consumer GPUs

After running:
    python merge_chunks_vmag.py --chunks_dir ./chunksV118 --out_dir ./dataV118
"""
import argparse
import copy
import json
import os
import warnings

import numpy as np
import torch
import pandapower as pp

from bdep_gpu import precompute_routes, bdep_gpu_batched, safe_norm_batch
from power_flow_utils import (
    build_adj_P, build_adj_Q,
    build_flow_matrix_P, build_flow_matrix_Q,
)

# Only reuse the genuinely reusable, low-risk building blocks: network
# loading and perturbation. NOT the chunk/manifest/resume internals (see
# module docstring for why).
from data_generation import CASE_MAP, load_case, perturb_network, _auto_chunk_size


# ---------------------------------------------------------------------------
# Worker: run power flow, return F matrices AND V_mag (same net, same call)
# ---------------------------------------------------------------------------
def run_power_flow_vmag(args_tuple):
    """
    Worker entry point. Runs AC + DC power flow and returns F matrices
    plus V_mag. Capturing V_mag here — from the very same converged `net`
    used to build FlowP/FlowQ — is what guarantees it lines up 1:1 with
    the DP/DQ/Ddc scenario it belongs to. Returns None on non-convergence,
    exactly like run_power_flow in data_generation_gpu.py.
    """
    net_base, n_bus, h_P, h_Q, rng_seed = args_tuple
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

    # --- V_mag: bus voltage magnitude, per-unit ---
    # pandapower's net.res_bus is indexed 0..n_bus-1, the SAME bus
    # ordering used by build_adj_P / build_flow_matrix_P (from_bus/to_bus
    # indices), so this lines up with n_bus exactly as MATLAB's
    # results_ac_case.bus(:, VM) lines up with mpc.bus row order.
    Vmag = net.res_bus['vm_pu'].values[:n_bus].astype(np.float64)

    result = {
        'FlowP_n':  FlowP[:n_bus, :n_bus].astype(np.float64),
        'FlowQ_n':  FlowQ[:n_bus, :n_bus].astype(np.float64),
        'FlowDC_n': FlowDC[:n_bus, :n_bus].astype(np.float64),
        'Vmag':     Vmag,
    }
    if h_P > n_bus:
        result['FlowP_h']  = FlowP[:h_P, :h_P].astype(np.float64)
        result['FlowDC_h'] = FlowDC[:h_P, :h_P].astype(np.float64)
    if h_Q > n_bus:
        result['FlowQ_h']  = FlowQ[:h_Q, :h_Q].astype(np.float64)
    return result


# ---------------------------------------------------------------------------
# GPU batch processing (identical to data_generation_gpu.py, Vmag passed
# straight through untouched — it needs no GPU computation).
# ---------------------------------------------------------------------------
def process_gpu_batch_vmag(batch_results, routes_dict, device, dtype,
                            n_bus, h_P, h_Q, save_extras):
    B = len(batch_results)

    F_P_n  = torch.tensor(np.stack([r['FlowP_n']  for r in batch_results]),
                           device=device, dtype=dtype)
    F_Q_n  = torch.tensor(np.stack([r['FlowQ_n']  for r in batch_results]),
                           device=device, dtype=dtype)
    F_DC_n = torch.tensor(np.stack([r['FlowDC_n'] for r in batch_results]),
                           device=device, dtype=dtype)

    F_all = torch.cat([F_P_n, F_Q_n, F_DC_n], dim=0)
    D_all = bdep_gpu_batched(routes_dict['n'], F_all, device, dtype)
    D_all = safe_norm_batch(D_all)

    D_P, D_Q, D_DC = D_all[:B], D_all[B:2 * B], D_all[2 * B:]

    results = []
    for i in range(B):
        results.append({
            'DP':   D_P[i].cpu().numpy(),
            'DQ':   D_Q[i].cpu().numpy(),
            'Ddc':  D_DC[i].cpu().numpy(),
            'Vmag': batch_results[i]['Vmag'],   # <-- carried through unchanged
        })

    if save_extras:
        if h_P > n_bus:
            F_P_h  = torch.tensor(np.stack([r['FlowP_h']  for r in batch_results]),
                                   device=device, dtype=dtype)
            F_DC_h = torch.tensor(np.stack([r['FlowDC_h'] for r in batch_results]),
                                   device=device, dtype=dtype)
            F_h_all = torch.cat([F_P_h, F_DC_h], dim=0)
            D_h_all = bdep_gpu_batched(routes_dict['hP'], F_h_all, device, dtype)
            D_h_all = safe_norm_batch(D_h_all)
            D_HP, D_Hdc = D_h_all[:B], D_h_all[B:]
        if h_Q > n_bus:
            F_Q_h = torch.tensor(np.stack([r['FlowQ_h'] for r in batch_results]),
                                  device=device, dtype=dtype)
            D_HQ = safe_norm_batch(bdep_gpu_batched(routes_dict['hQ'], F_Q_h, device, dtype))

        for i in range(B):
            if h_P > n_bus:
                results[i]['DHP']  = D_HP[i].cpu().numpy()
                results[i]['DHdc'] = D_Hdc[i].cpu().numpy()
            if h_Q > n_bus:
                results[i]['DHQ']  = D_HQ[i].cpu().numpy()

    return results


# ---------------------------------------------------------------------------
# Self-contained chunk / manifest / resume helpers
# (deliberately independent of data_generation.py's private versions —
#  see module docstring)
# ---------------------------------------------------------------------------
def _write_chunk_vmag(chunk_buf, chunk_idx, out_dir, save_extras):
    """
    Stack a list of per-scenario result dicts into one compressed .npz.
    Keys match merge_chunks.py's expectations (Ddc, DQ, DP) plus 'Vmag',
    with the scenario axis last: (n_bus, n_bus, k) / (n_bus, k).
    """
    fname = f"chunk_{chunk_idx:05d}.npz"
    path = os.path.join(out_dir, fname)

    save_kwargs = {
        'DP':   np.stack([r['DP']   for r in chunk_buf], axis=-1),
        'DQ':   np.stack([r['DQ']   for r in chunk_buf], axis=-1),
        'Ddc':  np.stack([r['Ddc']  for r in chunk_buf], axis=-1),
        'Vmag': np.stack([r['Vmag'] for r in chunk_buf], axis=-1),  # (n_bus, k)
    }
    if save_extras and 'DHP' in chunk_buf[0]:
        save_kwargs['DHP']  = np.stack([r['DHP']  for r in chunk_buf], axis=-1)
        save_kwargs['DHdc'] = np.stack([r['DHdc'] for r in chunk_buf], axis=-1)
    if save_extras and 'DHQ' in chunk_buf[0]:
        save_kwargs['DHQ']  = np.stack([r['DHQ']  for r in chunk_buf], axis=-1)

    np.savez_compressed(path, **save_kwargs)
    return path


def _write_manifest_atomic_vmag(manifest, out_dir):
    path = os.path.join(out_dir, 'manifest.json')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def _resume_state_vmag(out_dir, seed):
    """
    Look for an existing manifest.json in out_dir and resume from it.
    Returns (collected, chunk_idx, rng_advance, chunk_paths, prev_manifest).
    """
    manifest_path = os.path.join(out_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        return 0, 0, 0, [], None

    with open(manifest_path) as f:
        prev = json.load(f)

    if prev.get('seed') != seed:
        print(f"  ! Existing manifest used seed={prev.get('seed')}, "
              f"requested seed={seed}. Starting fresh instead of resuming.")
        return 0, 0, 0, [], None

    chunk_paths = [os.path.join(out_dir, fn) for fn in prev.get('chunk_files', [])]
    chunk_paths = [p for p in chunk_paths if os.path.exists(p)]
    collected = prev.get('n_scenarios', 0)
    chunk_idx = len(chunk_paths)
    rng_advance = prev.get('attempts_so_far', 0)
    return collected, chunk_idx, rng_advance, chunk_paths, prev


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_dataset_vmag(
    case_num:    int   = 14,
    n_scenarios: int   = 11050,
    chunk_size:  int   = None,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
    n_workers:   int   = 10,
    gpu_batch:   int   = 50,
    save_extras: bool  = False,
    out_dir:     str   = "chunksV",
    device:      str   = None,
    dtype:       torch.dtype = torch.float64,
):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    if chunk_size is None:
        chunk_size = _auto_chunk_size(n_bus, h_P, h_Q)

    print(f"Buses: {n_bus}  |  h_P: {h_P}  |  h_Q: {h_Q}")
    print(f"Device: {device}  |  dtype: {dtype}")
    print(f"GPU batch size: {gpu_batch} scenarios")
    print(f"Workers: {n_workers}  |  Target scenarios: {n_scenarios}")
    print(f"Chunk size: {chunk_size} scenarios")
    print(f"Output dir: {out_dir}  (includes 'Vmag' per chunk)\n")

    print("Pre-computing shortest path routes on GPU...")
    routes_dict = {}
    routes_dict['n'] = precompute_routes(adj_P_base[:n_bus, :n_bus], device, dtype)
    print(f"  \u2713 Routes for n={n_bus} (used by DP, DQ, Ddc)")
    if save_extras:
        if h_P > n_bus:
            routes_dict['hP'] = precompute_routes(adj_P_base, device, dtype)
            print(f"  \u2713 Routes for h_P={h_P} (used by DHP, DHdc)")
        if h_Q > n_bus:
            routes_dict['hQ'] = precompute_routes(adj_Q_base, device, dtype)
            print(f"  \u2713 Routes for h_Q={h_Q} (used by DHQ)")
    print()

    print("Checking for existing chunks to resume from...")
    collected, chunk_idx, rng_advance, chunk_paths, prev_manifest = \
        _resume_state_vmag(out_dir, seed)
    remaining = n_scenarios - collected
    if remaining <= 0:
        print(f"Already have {collected} scenarios — nothing to do.")
        return
    print(f"Need {remaining} more scenarios.\n")

    rng = np.random.default_rng(seed)
    if rng_advance > 0:
        rng.integers(0, 2**31, size=rng_advance)

    max_attempts = remaining * 5
    attempts = 0
    chunk_buf = []
    gpu_buf = []

    pbar = tqdm(total=n_scenarios, initial=collected, desc="Scenarios", unit="scenario")

    def _build_manifest():
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
            'has_vmag':        True,
            'chunk_files':     [os.path.basename(p) for p in chunk_paths],
            'attempts_so_far': (prev_manifest or {}).get('attempts_so_far', 0) + attempts,
        }

    def _flush_chunk():
        nonlocal chunk_idx, collected
        if not chunk_buf:
            return
        p = _write_chunk_vmag(chunk_buf, chunk_idx, out_dir, save_extras)
        chunk_paths.append(p)
        chunk_idx += 1
        chunk_buf.clear()
        _write_manifest_atomic_vmag(_build_manifest(), out_dir)

    def _store(result):
        nonlocal collected
        chunk_buf.append(result)
        collected += 1
        pbar.update(1)
        pbar.set_postfix({
            'attempts':  attempts,
            'conv_rate': f"{collected / max(1, attempts):.1%}",
            'chunks':    chunk_idx,
        })
        if len(chunk_buf) >= chunk_size:
            _flush_chunk()

    def _process_gpu_buffer():
        nonlocal gpu_buf
        if not gpu_buf:
            return
        needed = n_scenarios - collected
        if needed <= 0:
            gpu_buf.clear()
            return
        batch_to_process = gpu_buf[:needed]
        gpu_buf.clear()
        results = process_gpu_batch_vmag(
            batch_to_process, routes_dict, device, dtype,
            n_bus, h_P, h_Q, save_extras
        )
        for r in results:
            _store(r)

    queue_size = n_workers * 2

    def _make_task():
        return (net_base, n_bus, h_P, h_Q, int(rng.integers(0, 2**31)))

    if n_workers == 1:
        while collected < n_scenarios and attempts < max_attempts:
            result = run_power_flow_vmag(_make_task())
            attempts += 1
            if result is not None:
                gpu_buf.append(result)
                if len(gpu_buf) >= gpu_batch:
                    _process_gpu_buffer()
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(run_power_flow_vmag, _make_task()): None
                for _ in range(min(queue_size, max_attempts))
            }
            attempts += len(futures)
            while futures and collected < n_scenarios and attempts < max_attempts:
                future = next(as_completed(futures))
                del futures[future]
                result = future.result()
                if result is not None:
                    gpu_buf.append(result)
                    if len(gpu_buf) >= gpu_batch:
                        _process_gpu_buffer()
                if attempts < max_attempts and collected < n_scenarios:
                    futures[pool.submit(run_power_flow_vmag, _make_task())] = None
                    attempts += 1

    _process_gpu_buffer()
    _flush_chunk()

    pbar.close()
    if collected < n_scenarios:
        print(f"\n\u26a0 Only collected {collected}/{n_scenarios} converged scenarios.")

    print(f"\nGenerated {collected} scenarios in {attempts} attempts "
          f"(convergence rate: {collected / max(1, attempts):.1%})")
    print(f"Wrote {chunk_idx} chunk files to '{out_dir}/' (each with a 'Vmag' array)")
    _write_manifest_atomic_vmag(_build_manifest(), out_dir)
    print(f"Manifest written to '{os.path.join(out_dir, 'manifest.json')}'")
    print(f"\nNext step:\n  python merge_chunks_vmag.py "
          f"--chunks_dir {out_dir} --out_dir <output_folder>")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate DP/DQ/Ddc dataset with V_mag, GPU-accelerated bdep"
    )
    parser.add_argument("--case",        type=int,   default=14)
    parser.add_argument("--n_scenarios", type=int,   default=11050)
    parser.add_argument("--chunk_size",  type=int,   default=None)
    parser.add_argument("--n_workers",   type=int,   default=10)
    parser.add_argument("--gpu_batch",   type=int,   default=50)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--save_extras", action="store_true")
    parser.add_argument("--out_dir",     type=str,   default="chunksV")
    parser.add_argument("--device",      type=str,   default=None)
    parser.add_argument("--float32",     action="store_true")
    args = parser.parse_args()

    dtype = torch.float32 if args.float32 else torch.float64

    generate_dataset_vmag(
        case_num    = args.case,
        n_scenarios = args.n_scenarios,
        chunk_size  = args.chunk_size,
        n_workers   = args.n_workers,
        gpu_batch   = args.gpu_batch,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
        save_extras = args.save_extras,
        out_dir     = args.out_dir,
        device      = args.device,
        dtype       = dtype,
    )


if __name__ == "__main__":
    main()
