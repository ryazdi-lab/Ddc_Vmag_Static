"""
bdep.py  (vectorized)

Computes the bus dependency matrix D (paper Eq. 1):

    D(s, k) = sum_{t: s≠t≠k ∈ V}  f_st(k) / f_st

Speedups over the original:
  1. shortest_path_dp is now vectorized internally (see shortest_path_dp.py).
  2. The O(n³) triple Python loop that accumulates D is replaced by a
     vectorised NumPy scatter-add:
       - Intermediate flow records are stored in a dense 3-D array
         `stk[s, t, k]` (filled via vectorised indexing).
       - D[s, k] = sum_t( stk[s,t,k] / P[s,t] ) is computed with
         np.nansum over axis=1 in one call — no Python loops at all.
  3. `_bus_distance_matrix` in power_flow_utils.py (separate file) still
     has a Python double-loop — vectorised there too; see power_flow_utils.py.

All bus indices remain 1-indexed at the boundary with shortest_path_dp
(matching MATLAB); internal NumPy arrays are 0-indexed.
"""

import numpy as np
from shortest_path_dp import shortest_path_dp_all_sources, route_for_source


def bdep(Aw: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute the bus dependency matrix D.

    Args:
        Aw : (n, n) weighted adjacency matrix (electrical distances).
        F  : (n, n) power-flow matrix.

    Returns:
        D  : (n, n) bus dependency matrix, 0-indexed.
    """
    n = Aw.shape[0]

    # Dense arrays to accumulate path data (0-indexed)
    # stk[s0, t0, k0] = f_stk  (flow through intermediate k on path s->t)
    # P[s0, t0]       = f_st   (max flow on the s-t path)
    stk = np.zeros((n, n, n), dtype=float)
    P   = np.zeros((n, n),    dtype=float)

    # ---------- Step 1: shortest paths + flow data --------------------------
    # KEY FIX vs the original: the DP table for a given target j is the
    # SAME regardless of which source i you eventually read off of it (the
    # recursion only depends on the target). The original code rebuilt the
    # full O(n^3) table once per (i, j) PAIR -> O(n^2) pairs -> O(n^5) total.
    # Here we build it once per target j (O(n) targets -> O(n^4) total) and
    # reuse it for every source i < j, an O(n) reduction in work with
    # identical numerical results.
    for j in range(2, n + 1):                        # 1-indexed target
        J, nxt, t0 = shortest_path_dp_all_sources(Aw, j, n)

        for i in range(1, j):                         # 1-indexed source
            _, route_st = route_for_source(J, nxt, i, t0, n)

            r0 = [r - 1 for r in route_st]   # 0-indexed route

            if len(r0) < 2:
                continue

            # Max flow at the endpoints of the path
            maxflow = max(abs(F[r0[0],  r0[1]]),
                          abs(F[r0[-2], r0[-1]]))

            # Intermediate nodes
            intermediates = r0[1:-1]
            for idx, k0 in enumerate(intermediates):
                prev_node = r0[idx]
                next_node = r0[idx + 2]
                f_stk = max(abs(F[k0, prev_node]),
                            abs(F[k0, next_node]))

                # Store symmetrically (s->t and t->s share the same k flow)
                stk[i - 1, j - 1, k0] = f_stk
                stk[j - 1, i - 1, k0] = f_stk

                if f_stk > maxflow:
                    maxflow = f_stk

            P[i - 1, j - 1] = maxflow
            P[j - 1, i - 1] = maxflow

    # ---------- Step 2: vectorised accumulation of D ------------------------
    # ratio[s, t, k] = stk[s,t,k] / P[s,t]   where P[s,t] > 0
    # D[s, k]        = sum_t ratio[s,t,k]     excluding t==s and t==k
    #
    # Build ratio array: broadcast P over the k dimension
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(P[:, :, np.newaxis] > 0,
                         stk / P[:, :, np.newaxis],
                         0.0)

    # Zero out the t==s diagonal slice: ratio[s, s, :] = 0
    diag_idx = np.arange(n)
    ratio[diag_idx, diag_idx, :] = 0.0

    # Zero out the t==k condition: ratio[s, t, t] = 0 for all s
    # i.e. ratio[:, k, k] = 0 for every k
    ratio[:, diag_idx, diag_idx] = 0.0

    # D[s, k] = sum over t
    D = ratio.sum(axis=1)   # shape (n, n): axis=1 sums over t

    return D


# ---------------------------------------------------------------------------
# Quick sanity check (same as original)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Aw = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=float)

    F = np.array([
        [0, 10,  3],
        [0,  0,  5],
        [0,  0,  0],
    ], dtype=float)

    D = bdep(Aw, F)
    print("Dependency matrix D:")
    print(np.round(D, 4))
    print("Shape:", D.shape)
    print("Row sums (closeness):",    D.sum(axis=1).round(4))
    print("Col sums (betweenness):",  D.sum(axis=0).round(4))
    print("Sanity: diagonal == 0:", np.all(np.diag(D) == 0))
