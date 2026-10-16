"""
power_flow_utils.py  (vectorized)

Python ports of four MATLAB helper functions:
  - mpc_results2adjP.m  ->  build_adj_P(net)
  - mpc_results2adjQ.m  ->  build_adj_Q(net)
  - results2flowPH.m    ->  build_flow_matrix_P(net)
  - results2flowQH.m    ->  build_flow_matrix_Q(net)

Vectorization changes vs original:
  _bus_distance_matrix:
    - The double Python loop `for i,j: if Ybus[i,j]!=0: adj[i,j]=1/|Y|`
      is replaced by:
        mask = (off-diagonal) & (Ybus != 0)
        adj[mask] = 1 / |Ybus[mask]|
      One NumPy pass instead of n² Python iterations.

  build_flow_matrix_P / _Q:
    - The per-row pandapower `iterrows()` loops over lines and trafos are
      replaced by vectorised array indexing, reading all from/to bus
      indices and result columns at once.

Interface (function signatures, return shapes, index conventions) is
identical to the original.
"""

import numpy as np
import pandapower as pp


# ---------------------------------------------------------------------------
# Internal helper: bus-to-bus electrical distances (vectorized)
# ---------------------------------------------------------------------------

def _bus_distance_matrix(net) -> np.ndarray:
    """
    Returns the n_bus x n_bus electrical distance matrix (1/|Y_ij|) for
    off-diagonal entries where a direct link exists, 0 elsewhere.
    """
    Ybus = net._ppc['internal']['Ybus'].toarray()
    n    = Ybus.shape[0]

    # Off-diagonal mask where admittance is non-zero
    off_diag = ~np.eye(n, dtype=bool)
    abs_Y    = np.abs(Ybus)
    mask     = off_diag & (abs_Y != 0)

    adj = np.zeros((n, n))
    adj[mask] = 1.0 / abs_Y[mask]
    return adj


# ---------------------------------------------------------------------------
# build_adj_P
# ---------------------------------------------------------------------------

def build_adj_P(net) -> np.ndarray:
    """
    Active-power weighted adjacency matrix extended with virtual gen/load nodes.
    """
    n_bus    = len(net.bus)
    bus_adj  = _bus_distance_matrix(net)

    active_gen_buses = list(net.gen.loc[net.res_gen['p_mw'] != 0, 'bus'])
    ext_buses        = list(net.ext_grid['bus'])
    all_gen_buses    = ext_buses + active_gen_buses

    active_load_buses = list(net.load.loc[net.res_load['p_mw'] != 0, 'bus'])

    n_gen   = len(all_gen_buses)
    n_load  = len(active_load_buses)
    n_total = n_bus + n_gen + n_load

    adj = np.zeros((n_total, n_total))
    adj[:n_bus, :n_bus] = bus_adj

    # Virtual gen nodes — vectorised assignment
    if n_gen:
        gen_vnodes = np.arange(n_bus, n_bus + n_gen)
        gen_buses_arr = np.array(all_gen_buses)
        adj[gen_buses_arr, gen_vnodes] = 1.0
        adj[gen_vnodes, gen_buses_arr] = 1.0

    # Virtual load nodes
    if n_load:
        load_vnodes    = np.arange(n_bus + n_gen, n_bus + n_gen + n_load)
        load_buses_arr = np.array(active_load_buses)
        adj[load_buses_arr, load_vnodes] = 1.0
        adj[load_vnodes, load_buses_arr] = 1.0

    return adj


# ---------------------------------------------------------------------------
# build_adj_Q
# ---------------------------------------------------------------------------

def build_adj_Q(net) -> np.ndarray:
    """
    Reactive-power weighted adjacency matrix extended with virtual gen/load nodes.
    """
    n_bus   = len(net.bus)
    bus_adj = _bus_distance_matrix(net)

    reactive_gen_buses = list(net.gen.loc[net.res_gen['q_mvar'] != 0, 'bus'])
    ext_buses_q        = list(net.ext_grid.loc[net.res_ext_grid['q_mvar'] != 0, 'bus'])
    all_q_gen_buses    = ext_buses_q + reactive_gen_buses

    reactive_load_buses = list(net.load.loc[net.res_load['q_mvar'] != 0, 'bus'])

    n_gen   = len(all_q_gen_buses)
    n_load  = len(reactive_load_buses)
    n_total = n_bus + n_gen + n_load

    adj = np.zeros((n_total, n_total))
    adj[:n_bus, :n_bus] = bus_adj

    if n_gen:
        gen_vnodes    = np.arange(n_bus, n_bus + n_gen)
        gen_buses_arr = np.array(all_q_gen_buses)
        adj[gen_buses_arr, gen_vnodes] = 1.0
        adj[gen_vnodes, gen_buses_arr] = 1.0

    if n_load:
        load_vnodes    = np.arange(n_bus + n_gen, n_bus + n_gen + n_load)
        load_buses_arr = np.array(reactive_load_buses)
        adj[load_buses_arr, load_vnodes] = 1.0
        adj[load_vnodes, load_buses_arr] = 1.0

    return adj


# ---------------------------------------------------------------------------
# build_flow_matrix_P  (vectorized line/trafo loops)
# ---------------------------------------------------------------------------

def build_flow_matrix_P(net) -> np.ndarray:
    """
    Directed active-power flow matrix.  F_P[i,j] = P flowing from i to j.
    """
    n_bus = len(net.bus)

    active_gen_mask  = net.res_gen['p_mw'] != 0
    active_gen       = net.gen.loc[active_gen_mask]
    active_gen_res   = net.res_gen.loc[active_gen_mask]
    ext_p            = net.res_ext_grid['p_mw'].values
    ext_buses        = net.ext_grid['bus'].values

    gen_buses  = list(ext_buses)        + list(active_gen['bus'].values)
    gen_flows  = list(ext_p)            + list(active_gen_res['p_mw'].values)

    active_load_mask = net.res_load['p_mw'] != 0
    active_load      = net.load.loc[active_load_mask]
    active_load_res  = net.res_load.loc[active_load_mask]
    load_buses = list(active_load['bus'].values)
    load_flows = list(active_load_res['p_mw'].values)

    n_gen   = len(gen_buses)
    n_load  = len(load_buses)
    n_total = n_bus + n_gen + n_load
    F = np.zeros((n_total, n_total))

    # 1. Line flows — vectorised
    if len(net.line):
        from_b  = net.line['from_bus'].values.astype(int)
        to_b    = net.line['to_bus'].values.astype(int)
        p_from  = net.res_line.loc[net.line.index, 'p_from_mw'].values
        p_to    = net.res_line.loc[net.line.index, 'p_to_mw'].values
        F[from_b, to_b] = p_from
        F[to_b, from_b] = p_to

    # 2. Trafo flows — vectorised
    if len(net.trafo):
        hv_b   = net.trafo['hv_bus'].values.astype(int)
        lv_b   = net.trafo['lv_bus'].values.astype(int)
        p_hv   = net.res_trafo.loc[net.trafo.index, 'p_hv_mw'].values
        p_lv   = net.res_trafo.loc[net.trafo.index, 'p_lv_mw'].values
        F[hv_b, lv_b] = p_hv
        F[lv_b, hv_b] = p_lv

    # 3. Generator virtual nodes — vectorised
    if n_gen:
        gen_vnodes    = np.arange(n_bus, n_bus + n_gen)
        gen_buses_arr = np.array(gen_buses)
        gen_abs       = np.abs(gen_flows)
        F[gen_vnodes,    gen_buses_arr] =  gen_abs
        F[gen_buses_arr, gen_vnodes]    = -gen_abs

    # 4. Load virtual nodes — vectorised
    if n_load:
        load_vnodes    = np.arange(n_bus + n_gen, n_bus + n_gen + n_load)
        load_buses_arr = np.array(load_buses)
        load_abs       = np.abs(load_flows)
        F[load_buses_arr, load_vnodes] =  load_abs
        F[load_vnodes,    load_buses_arr] = -load_abs

    return F


# ---------------------------------------------------------------------------
# build_flow_matrix_Q  (vectorized line/trafo loops)
# ---------------------------------------------------------------------------

def build_flow_matrix_Q(net) -> np.ndarray:
    """
    Directed reactive-power flow matrix.  F_Q[i,j] = Q flowing from i to j.
    """
    n_bus = len(net.bus)

    reactive_gen_mask  = net.res_gen['q_mvar'] != 0
    reactive_gen       = net.gen.loc[reactive_gen_mask]
    reactive_gen_res   = net.res_gen.loc[reactive_gen_mask]
    ext_q              = net.res_ext_grid['q_mvar'].values
    ext_buses_q        = net.ext_grid.loc[net.res_ext_grid['q_mvar'] != 0, 'bus'].values

    q_gen_buses = list(ext_buses_q)   + list(reactive_gen['bus'].values)
    q_gen_flows = list(ext_q[net.res_ext_grid['q_mvar'].values != 0]) + \
                  list(reactive_gen_res['q_mvar'].values)

    reactive_load_mask = net.res_load['q_mvar'] != 0
    reactive_load      = net.load.loc[reactive_load_mask]
    reactive_load_res  = net.res_load.loc[reactive_load_mask]
    q_load_buses = list(reactive_load['bus'].values)
    q_load_flows = list(reactive_load_res['q_mvar'].values)

    n_gen   = len(q_gen_buses)
    n_load  = len(q_load_buses)
    n_total = n_bus + n_gen + n_load
    F = np.zeros((n_total, n_total))

    # 1. Line reactive flows — vectorised
    if len(net.line):
        from_b   = net.line['from_bus'].values.astype(int)
        to_b     = net.line['to_bus'].values.astype(int)
        q_from   = net.res_line.loc[net.line.index, 'q_from_mvar'].values
        q_to     = net.res_line.loc[net.line.index, 'q_to_mvar'].values
        F[from_b, to_b] = q_from
        F[to_b, from_b] = q_to

    # 2. Trafo reactive flows — vectorised
    if len(net.trafo):
        hv_b   = net.trafo['hv_bus'].values.astype(int)
        lv_b   = net.trafo['lv_bus'].values.astype(int)
        q_hv   = net.res_trafo.loc[net.trafo.index, 'q_hv_mvar'].values
        q_lv   = net.res_trafo.loc[net.trafo.index, 'q_lv_mvar'].values
        F[hv_b, lv_b] = q_hv
        F[lv_b, hv_b] = q_lv

    # 3. Reactive gen virtual nodes — vectorised
    if n_gen:
        gen_vnodes    = np.arange(n_bus, n_bus + n_gen)
        gen_buses_arr = np.array(q_gen_buses)
        gen_abs       = np.abs(q_gen_flows)
        F[gen_vnodes,    gen_buses_arr] =  gen_abs
        F[gen_buses_arr, gen_vnodes]    = -gen_abs

    # 4. Reactive load virtual nodes — vectorised
    if n_load:
        load_vnodes    = np.arange(n_bus + n_gen, n_bus + n_gen + n_load)
        load_buses_arr = np.array(q_load_buses)
        load_abs       = np.abs(q_load_flows)
        F[load_buses_arr, load_vnodes] =  load_abs
        F[load_vnodes,    load_buses_arr] = -load_abs

    return F


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandapower.networks as pn

    net = pn.case14()
    pp.runpp(net, numba=False)

    adj_P = build_adj_P(net)
    adj_Q = build_adj_Q(net)
    F_P   = build_flow_matrix_P(net)
    F_Q   = build_flow_matrix_Q(net)

    print(f"adj_P shape: {adj_P.shape}")
    print(f"adj_Q shape: {adj_Q.shape}")
    print(f"F_P   shape: {F_P.shape}")
    print(f"F_Q   shape: {F_Q.shape}")

    assert np.all(np.diag(adj_P) == 0), "adj_P diagonal not zero"
    assert np.all(np.diag(adj_Q) == 0), "adj_Q diagonal not zero"

    n_bus = len(net.bus)
    print(f"\nBus-bus P flows non-zero: {np.count_nonzero(F_P[:n_bus, :n_bus])}")
    print(f"Bus-bus Q flows non-zero: {np.count_nonzero(F_Q[:n_bus, :n_bus])}")
    print("\nAll checks passed.")
