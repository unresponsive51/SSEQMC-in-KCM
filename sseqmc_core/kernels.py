import numpy as np
from numba import njit


# Operator type constants
OP_IDENTITY = 0
OP_DIAG = 1
OP_OFFDIAG = 2


# ============================================================
# Core Compute Kernels (Numba JIT Backend)
# ============================================================

@njit
def fast_Xi_at_site(alpha_flat, site, neighbor_table):
    """Evaluates the kinetic constraint (Xi) at a specific site."""
    cnt = 0
    for k in range(4):
        ni = neighbor_table[site, k]
        cnt += alpha_flat[ni]
    return 1 if cnt == 1 else 0

@njit
def fast_op_weight(alpha_flat, site, op_type, neighbor_table, C, eta, J):
    """Calculates the local operator weight based on the Hamiltonian."""
    if op_type == OP_DIAG:
        return (C + eta) if alpha_flat[site] == 0 else C
    elif op_type == OP_OFFDIAG:
        return J * fast_Xi_at_site(alpha_flat, site, neighbor_table)
    return 1.0

@njit
def fast_toggle_type(op_type):
    """Toggles operator type between diagonal and off-diagonal."""
    if op_type == OP_DIAG: return OP_OFFDIAG
    elif op_type == OP_OFFDIAG: return OP_DIAG
    return op_type

@njit
def fast_propagate_to_p(seq_i, alpha_init, p_target, L):
    """
    Propagates the spin configuration along the imaginary time axis to p_target.
    
    Note:
        alpha_work is the configuration propagated up to (but not including) 
        the operator at index p_target.
    """
    alpha_work = alpha_init.copy()
    for p in range(p_target):
        if seq_i[p, 0] == OP_OFFDIAG:
            site = seq_i[p, 1] * L + seq_i[p, 2]
            alpha_work[site] ^= 1
    return alpha_work

@njit
def fast_segment_update(
    seq_i, alpha_init, nonid, first_op, next_op, neighbor_table, 
    M, L, C, eta, J
):
    """
    Performs a single segment update.
    
    Proposes a flip of operator types (DIAG <-> OFFDIAG) between two successive 
    non-identity operators on the same site. Enforces the kinetic constraint 
    along the segment.
    """
    if len(nonid) == 0:
        return False, 0, 0, 0, 0, False, -1
        
    # 1. Uniformly sample an initial non-identity operator
    p1 = nonid[np.random.randint(len(nonid))]
    t1_old = seq_i[p1, 0]
    if t1_old == OP_IDENTITY: 
        return False, 0, 0, 0, 0, False, -1
        
    i0 = seq_i[p1, 1]
    j0 = seq_i[p1, 2]
    site0 = i0 * L + j0
    
    # 2. Locate the next operator on the same site via static linked list
    p2 = next_op[p1]
    wrap_segment = False
    if p2 == -1:
        p2 = first_op[site0]
        wrap_segment = True
        
    if p1 == p2:
        return False, 0, 0, 0, 0, False, -1
        
    t2_old = seq_i[p2, 0]
    if t2_old == OP_IDENTITY: 
        return False, 0, 0, 0, 0, False, -1
        
    t1_new = fast_toggle_type(t1_old)
    t2_new = fast_toggle_type(t2_old)
    
    # 3. Evaluate local weights prior to the segment
    alpha_before_p1 = fast_propagate_to_p(seq_i, alpha_init, p1, L)
    w1_old = fast_op_weight(alpha_before_p1, site0, t1_old, neighbor_table, C, eta, J)
    w1_new = fast_op_weight(alpha_before_p1, site0, t1_new, neighbor_table, C, eta, J)
    if w1_old <= 0 or w1_new <= 0: 
        return False, 0, 0, 0, 0, False, -1
        
    alpha_old = alpha_before_p1.copy()
    alpha_new = alpha_before_p1.copy()
    
    if t1_old == OP_OFFDIAG: alpha_old[site0] ^= 1
    if t1_new == OP_OFFDIAG:
        if fast_Xi_at_site(alpha_new, site0, neighbor_table) != 1:
            return False, 0, 0, 0, 0, False, -1
        alpha_new[site0] ^= 1
        
    # 4. Scan the segment and enforce kinetic constraints on neighboring sites
    p = (p1 + 1) % M
    while p != p2:
        op_type = seq_i[p, 0]
        if op_type == OP_OFFDIAG:
            site = seq_i[p, 1] * L + seq_i[p, 2]
            alpha_old[site] ^= 1
            
            is_neighbor = False
            for k in range(4):
                if neighbor_table[site0, k] == site:
                    is_neighbor = True
                    break
            
            if is_neighbor:
                if fast_Xi_at_site(alpha_new, site, neighbor_table) != 1:
                    return False, 0, 0, 0, 0, False, -1
            alpha_new[site] ^= 1
        p = (p + 1) % M
        
    # 5. Evaluate weights post-segment and perform Metropolis acceptance test
    w2_old = fast_op_weight(alpha_old, site0, t2_old, neighbor_table, C, eta, J)
    w2_new = fast_op_weight(alpha_new, site0, t2_new, neighbor_table, C, eta, J)
    if w2_old <= 0 or w2_new <= 0: 
        return False, 0, 0, 0, 0, False, -1
        
    ratio = (w1_new * w2_new) / (w1_old * w2_old)
    if ratio >= 1.0 or np.random.rand() < ratio:
        seq_i[p1, 0] = t1_new
        seq_i[p2, 0] = t2_new
        return True, t1_old, t1_new, t2_old, t2_new, wrap_segment, site0
        
    return False, 0, 0, 0, 0, False, -1

@njit
def fast_diagonal_update(seq_i, alpha_flat, M, N_sites, L, beta, C, eta):
    """Performs the diagonal update over the entire operator string."""
    alpha_work = alpha_flat.copy()
    
    n = 0
    for p in range(M):
        if seq_i[p, 0] != OP_IDENTITY:
            n += 1
            
    for p in range(M):
        op_type = seq_i[p, 0]
        
        if op_type == OP_IDENTITY:
            if M - n <= 0:
                continue
            i = np.random.randint(L)
            j = np.random.randint(L)
            site = i * L + j
            
            h_ij = (C + eta) if alpha_work[site] == 0 else C
            if h_ij <= 0: continue
            
            acc_prob = beta * N_sites * h_ij / (M - n)
            if acc_prob >= 1.0 or np.random.rand() < acc_prob:
                seq_i[p, 0] = OP_DIAG
                seq_i[p, 1] = i
                seq_i[p, 2] = j
                n += 1
                
        elif op_type == OP_DIAG:
            site = seq_i[p, 1] * L + seq_i[p, 2]
            h_ij = (C + eta) if alpha_work[site] == 0 else C
            if h_ij <= 0: continue
            
            acc_prob = (M - n + 1) / (beta * N_sites * h_ij)
            if acc_prob >= 1.0 or np.random.rand() < acc_prob:
                seq_i[p, 0] = OP_IDENTITY
                seq_i[p, 1] = 0
                seq_i[p, 2] = 0
                n -= 1
                
        elif op_type == OP_OFFDIAG:
            site = seq_i[p, 1] * L + seq_i[p, 2]
            alpha_work[site] ^= 1

@njit
def fast_loop_sweep(seq_i, alpha_flat, nonid, first_op, next_op, neighbor_table, 
                    M, L, C, eta, J, n_loops):
    """
    Executes a batched sweep of segment updates to minimize Python-C++ 
    boundary crossing overhead (Kernel Fusion).
    """
    loop_accepts = 0
    diag_diff = 0
    offdiag_diff = 0
    
    for _ in range(n_loops):
        for _ in range(20):
            accepted, t1_old, t1_new, t2_old, t2_new, wrap, site0 = fast_segment_update(
                seq_i, alpha_flat, nonid, first_op, next_op, neighbor_table, M, L, C, eta, J
            )
            
            if accepted:
                loop_accepts += 1
                
                if t1_old == OP_DIAG and t1_new == OP_OFFDIAG:
                    diag_diff -= 1; offdiag_diff += 1
                elif t1_old == OP_OFFDIAG and t1_new == OP_DIAG:
                    diag_diff += 1; offdiag_diff -= 1
                    
                if t2_old == OP_DIAG and t2_new == OP_OFFDIAG:
                    diag_diff -= 1; offdiag_diff += 1
                elif t2_old == OP_OFFDIAG and t2_new == OP_DIAG:
                    diag_diff += 1; offdiag_diff -= 1
                    
                if wrap:
                    alpha_flat[site0] ^= 1
                    
                break 
                
    return loop_accepts, diag_diff, offdiag_diff