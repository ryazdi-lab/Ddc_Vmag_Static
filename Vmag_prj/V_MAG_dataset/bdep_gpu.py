"""
bdep_gpu.py — GPU-accelerated bus dependency matrix computation using PyTorch.
FIXED VERSION: Guarantees 100% exact match with CPU bdep.py.
1. Routes are precomputed on CPU (eliminates GPU tie-breaking differences).
2. maxflow correctly includes the target endpoint.
3. Symmetry is handled perfectly using flattened scatter_add to match CPU logic.
"""
import torch
import numpy as np
from shortest_path_dp import shortest_path_dp_all_sources, route_for_source

@torch.no_grad()
def precompute_routes(Aw: np.ndarray, device: str = 'cuda', 
                      dtype: torch.dtype = torch.float64) -> dict:
    """
    Compute all-pairs shortest paths using CPU DP.
    Using CPU DP guarantees 100% identical routes to bdep.py.
    """
    n = Aw.shape[0]
    steps = n
    
    route_nodes_np = np.full((steps, n, n), -1, dtype=np.int32)
    route_valid_np = np.zeros((steps, n, n), dtype=bool)

    for t in range(1, n + 1):
        J, nxt, t0 = shortest_path_dp_all_sources(Aw, t, steps)
        for s in range(1, n + 1):
            if s == t:
                continue
            _, route_st = route_for_source(J, nxt, s, t0, steps)
            route_0 = [r - 1 for r in route_st]
            
            for p, node in enumerate(route_0):
                route_nodes_np[p, s - 1, t - 1] = node
                route_valid_np[p, s - 1, t - 1] = True

    route_nodes = torch.tensor(route_nodes_np, device=device, dtype=torch.long)
    route_valid = torch.tensor(route_valid_np, device=device, dtype=torch.bool)

    return {
        'route_nodes': route_nodes,
        'route_valid': route_valid,
        'n': n,
        'device': device,
    }

@torch.no_grad()
def bdep_gpu_batched(routes: dict, F_batch, device: str = 'cuda', 
                     dtype: torch.dtype = torch.float64) -> torch.Tensor:
    route_nodes = routes['route_nodes']
    route_valid = routes['route_valid']
    n = routes['n']
    max_len = route_nodes.shape[0]

    if isinstance(F_batch, np.ndarray):
        F_batch = torch.tensor(F_batch, device=device, dtype=dtype)
    elif F_batch.device.type != device:
        F_batch = F_batch.to(device=device, dtype=dtype)

    batch = F_batch.shape[0]
    F_abs = F_batch.abs()

    s_neq_t = ~torch.eye(n, device=device, dtype=torch.bool)
    
    s_idx_1d = torch.arange(n, device=device).view(1, n, 1)
    t_idx_1d = torch.arange(n, device=device).view(1, 1, n)
    s_lt_t = s_idx_1d < t_idx_1d  # Restrict to s < t to match CPU symmetry

    b_idx = torch.arange(batch, device=device).view(-1, 1, 1)
    s_idx_3d = torch.arange(n, device=device).view(1, n, 1).expand(batch, n, n)

    # ================ Pass 1: Compute maxflow ================
    maxflow = torch.zeros(batch, n, n, device=device, dtype=dtype)

    # 1. Source endpoint: |F[s, route[1]]|
    if max_len > 1:
        next_node = route_nodes[1]
        valid_src = route_valid[1] & s_neq_t & s_lt_t.squeeze(0)
        next_safe = next_node.clamp(0, n - 1)
        next_exp = next_safe.unsqueeze(0).expand(batch, n, n)
        flow_src = F_abs[b_idx, s_idx_3d, next_exp]
        maxflow = torch.where(valid_src.unsqueeze(0), torch.maximum(maxflow, flow_src), maxflow)

    # 2. Target endpoint: |F[route[-2], route[-1]]|
    route_lengths = route_valid.sum(dim=0)
    last_idx = (route_lengths - 1).clamp(0, max_len - 1)
    prev_idx = (route_lengths - 2).clamp(0, max_len - 1)

    last_node = route_nodes.gather(0, last_idx.unsqueeze(0)).squeeze(0)
    prev_node = route_nodes.gather(0, prev_idx.unsqueeze(0)).squeeze(0)

    valid_tgt = (route_lengths > 1) & s_neq_t & s_lt_t.squeeze(0)
    last_safe = last_node.clamp(0, n - 1)
    prev_safe = prev_node.clamp(0, n - 1)

    flow_tgt = F_abs[b_idx, prev_safe.unsqueeze(0), last_safe.unsqueeze(0)]
    maxflow = torch.where(valid_tgt.unsqueeze(0), torch.maximum(maxflow, flow_tgt.squeeze(0)), maxflow)

    # 3. Intermediate nodes
    for p in range(1, max_len - 1):
        k = route_nodes[p]
        prev = route_nodes[p - 1]
        next_n = route_nodes[p + 1]
        
        valid = (route_valid[p] & route_valid[p - 1] & route_valid[p + 1] 
                 & s_neq_t & s_lt_t.squeeze(0) 
                 & (k >= 0) & (prev >= 0) & (next_n >= 0))

        k_exp    = k.unsqueeze(0).expand(batch, n, n).clamp(0, n - 1)
        prev_exp = prev.unsqueeze(0).expand(batch, n, n).clamp(0, n - 1)
        next_exp = next_n.unsqueeze(0).expand(batch, n, n).clamp(0, n - 1)

        flow_prev = F_abs[b_idx, k_exp, prev_exp]
        flow_next = F_abs[b_idx, k_exp, next_exp]
        f_stk = torch.maximum(flow_prev, flow_next)
        
        maxflow = torch.where(valid.unsqueeze(0), torch.maximum(maxflow, f_stk), maxflow)

    # ================ Pass 2: Scatter-add into D ================
    maxflow_safe = torch.where(
        maxflow > 0, 
        maxflow, 
        torch.tensor(1.0, device=device, dtype=dtype)
    )
    D = torch.zeros(batch, n, n, device=device, dtype=dtype)

    for p in range(1, max_len - 1):
        k = route_nodes[p]
        prev = route_nodes[p - 1]
        next_n = route_nodes[p + 1]
        
        valid = (route_valid[p] & route_valid[p - 1] & route_valid[p + 1] 
                 & s_neq_t & s_lt_t.squeeze(0) 
                 & (k >= 0) & (prev >= 0) & (next_n >= 0))

        k_exp    = k.unsqueeze(0).expand(batch, n, n).clamp(0, n - 1)
        prev_exp = prev.unsqueeze(0).expand(batch, n, n).clamp(0, n - 1)
        next_exp = next_n.unsqueeze(0).expand(batch, n, n).clamp(0, n - 1)

        flow_prev = F_abs[b_idx, k_exp, prev_exp]
        flow_next = F_abs[b_idx, k_exp, next_exp]
        f_stk = torch.maximum(flow_prev, flow_next)
        ratio = f_stk / maxflow_safe

        ratio = torch.where(
            valid.unsqueeze(0), 
            ratio, 
            torch.tensor(0.0, device=device, dtype=dtype)
        )

        # Scatter to D[s, k]
        D.scatter_add_(2, k_exp, ratio)

        # Scatter to D[t, k] for symmetry using flattened indexing
        # This correctly maps ratio[b, s, t] to D[b, t, k]
        t_exp = t_idx_1d.expand(batch, n, n)
        index_flat = t_exp * n + k_exp
        
        # FIX: Flatten the index and ratio tensors to match D_flat's 2D shape
        D_flat = D.view(batch, -1)
        index_flat_flat = index_flat.view(batch, -1)
        ratio_flat = ratio.view(batch, -1)
        
        D_flat.scatter_add_(1, index_flat_flat, ratio_flat)

    # Zero diagonal: D[s, s] = 0
    diag = torch.arange(n, device=device)
    D[:, diag, diag] = 0.0

    return D

def bdep_gpu(routes: dict, F: np.ndarray, device: str = 'cuda', 
             dtype: torch.dtype = torch.float64) -> np.ndarray:
    """Single F matrix convenience wrapper. Returns numpy array."""
    F_batch = torch.tensor(F, device=device, dtype=dtype).unsqueeze(0)
    D = bdep_gpu_batched(routes, F_batch, device, dtype)
    return D[0].cpu().numpy()

def safe_norm_batch(D_batch: torch.Tensor) -> torch.Tensor:
    """Normalize each (n, n) slice by its max absolute value."""
    mx = D_batch.abs().amax(dim=(1, 2), keepdim=True)
    return torch.where(mx > 0, D_batch / mx, D_batch)

# ---------------------------------------------------------------------------
# Verification against CPU original
# ---------------------------------------------------------------------------
def verify_gpu(Aw: np.ndarray, F: np.ndarray, device: str = 'cuda', 
               dtype: torch.dtype = torch.float64, atol: float = 1e-10):
    """Compare GPU result against original CPU bdep."""
    from bdep import bdep
    D_cpu = bdep(Aw, F)
    routes = precompute_routes(Aw, device, dtype)
    D_gpu = bdep_gpu(routes, F, device, dtype)
    
    max_diff = np.max(np.abs(D_cpu - D_gpu))
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"CPU D sum: {D_cpu.sum():.6f}  |  GPU D sum: {D_gpu.sum():.6f}")
    
    if max_diff < atol:
        print("✅ SUCCESS: GPU and CPU match perfectly!")
    else:
        print("❌ FAIL: GPU and CPU still differ.")
        
    return max_diff < atol

if __name__ == "__main__":
    import pandapower as pp
    import pandapower.networks as pn
    from power_flow_utils import build_adj_P, build_flow_matrix_P
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    print("--- case57 test ---")
    net = pn.case57()
    pp.runpp(net, numba=False)
    Aw57 = build_adj_P(net)[:57, :57]
    F57 = build_flow_matrix_P(net)[:57, :57]
    verify_gpu(Aw57, F57, device, atol=1e-10)