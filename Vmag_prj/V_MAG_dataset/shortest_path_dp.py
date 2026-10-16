"""
shortest_path_dp.py  (vectorized)

Python port of shortestPathDP.m — Bertsekas DP shortest path.

Speedup over the original:
  - The inner loop  `for i in range(n): costs = L_inf[i,:] + J[k+1,:]`
    is replaced by a single broadcast addition  L_inf + J[k+1, :]
    so the whole backward-recursion step is one NumPy operation.
  - argmin and min are computed over the axis-1 (neighbours) in one call.

Interface is identical to the original (same 1-indexed convention).
"""

import numpy as np


def _dp_table(L_inf, t0, steps, n):
    """
    Build the backward-DP tables J, nxt for a FIXED target t0.

    Critically, this table already holds the optimal cost/route from
    *every* source node i to t0 - it does not depend on any particular
    source. Building it is O(steps * n^2) == O(n^3).

    Returns:
        J   : (steps, n) optimal cost from node i to t0 with (steps-k) arcs left
        nxt : (steps, n) best next-hop index from node i at step k
    """
    J   = np.full((steps, n), np.inf)
    nxt = np.full((steps, n), -1, dtype=np.int32)

    # Base case: 1 arc remaining
    J[steps - 1, :]   = L_inf[:, t0]
    nxt[steps - 1, :] = t0          # only one hop — go straight to target

    # Vectorized backward recursion
    # costs[i, j] = L_inf[i, j] + J[k+1, j]  ->  shape (n, n)
    # min over j gives J[k, i] for all i at once
    for p in range(1, steps):
        k = steps - 1 - p
        costs = L_inf + J[k + 1, :]          # broadcast: (n,n) + (n,) -> (n,n)
        nxt[k, :] = np.argmin(costs, axis=1) # best next hop for every source
        J[k, :]   = costs[np.arange(n), nxt[k, :]]

    return J, nxt


def _extract_route(J, nxt, s0, t0, steps):
    """Trace the optimal route for source s0 out of a pre-built (J, nxt) table."""
    best_step = int(np.argmin(J[:, s0]))
    J_st      = float(J[best_step, s0])

    route_0 = [s0]
    cur     = s0
    step    = best_step
    for _ in range(steps):
        if cur == t0:
            break
        if step >= steps or nxt[step, cur] == -1:
            break
        nxt_node = int(nxt[step, cur])
        route_0.append(nxt_node)
        cur   = nxt_node
        step += 1

    # Trim at first occurrence of t0 (safety)
    if t0 in route_0:
        idx     = route_0.index(t0)
        route_0 = route_0[: idx + 1]

    return J_st, route_0


def shortest_path_dp_all_sources(L: np.ndarray, t: int, steps: int):
    """
    Compute shortest-path cost/route to a FIXED target `t` from every other
    node in one DP table build (O(n^3)) instead of rebuilding the table
    once per source - which is what happens if you call `shortest_path_dp`
    in a loop over sources for the same target (an O(n) redundant blow-up).

    Args:
        L     : (n, n) cost matrix, 0 = no direct link. 1-indexed convention
                at the boundary (matches shortest_path_dp).
        t     : 1-indexed target node
        steps : max arcs (== n)

    Returns:
        J, nxt, t0 : the raw DP tables and 0-indexed target, for repeated
                     use with `route_for_source`.
    """
    t0 = t - 1
    n  = L.shape[0]
    L_inf = np.where(L == 0, np.inf, L.astype(float))
    J, nxt = _dp_table(L_inf, t0, steps, n)
    return J, nxt, t0


def route_for_source(J: np.ndarray, nxt: np.ndarray, s: int, t0: int, steps: int):
    """
    Extract the shortest path from 1-indexed source `s` to the target
    baked into (J, nxt, t0) from `shortest_path_dp_all_sources`.
    O(steps) - no table rebuild.

    Returns:
        J_st     : optimal cost
        route_st : list of 1-indexed nodes on the optimal route
    """
    s0 = s - 1
    J_st, route_0 = _extract_route(J, nxt, s0, t0, steps)
    route_st = [r + 1 for r in route_0]
    return J_st, route_st


def shortest_path_dp(L: np.ndarray, s: int, t: int, steps: int):
    """
    Shortest path via dynamic programming (Bertsekas), vectorized.

    Kept for backward compatibility / single-pair use. If you need shortest
    paths from many sources to the SAME target (as bdep.py does for every
    target column), use `shortest_path_dp_all_sources` + `route_for_source`
    instead - it builds the O(n^3) table once and reuses it, rather than
    rebuilding it per source (an O(n) redundant blow-up, i.e. O(n^4) total
    instead of O(n^3) for all sources to one target).

    Args:
        L     : (n, n) float array, cost matrix. Zeros mean no direct link.
                Callers use 1-indexed buses; we convert internally.
        s     : 1-indexed source node
        t     : 1-indexed target node
        steps : max arcs (== n in MATLAB call from bdep)

    Returns:
        J_st      : optimal cost (float)
        route_st  : list of 1-indexed nodes on the optimal route
    """
    J, nxt, t0 = shortest_path_dp_all_sources(L, t, steps)
    return route_for_source(J, nxt, s, t0, steps)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    L = np.array([
        [0, 1, 3, 0],
        [1, 0, 1, 0],
        [3, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=float)

    J_st, route_st = shortest_path_dp(L, s=1, t=4, steps=4)
    print(f"Optimal cost:           {J_st}")
    print(f"Optimal route (1-idx):  {route_st}")
    assert J_st == 3.0,          f"Expected 3, got {J_st}"
    assert route_st == [1,2,3,4], f"Expected [1,2,3,4], got {route_st}"
    print("Sanity check passed.")
