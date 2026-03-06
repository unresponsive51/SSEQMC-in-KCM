import numpy as np
from typing import Tuple, Dict, List
from .kernels import OP_IDENTITY, OP_DIAG, OP_OFFDIAG
from .kernels import fast_segment_update, fast_diagonal_update, fast_loop_sweep


# ============================================================
# Object-Oriented Frontend
# ============================================================

class RydbergSSE:
    """
    Frontend simulator class for configuring and managing the SSE-QMC state.
    """

    def __init__(self, L: int, beta: float, J: float = 1.0, mu: float = 0.0):
        self.L = L
        self.N = L * L
        self.beta = beta
        self.J = J
        self.mu = mu
        self.eta = -mu  
        self.C = self.eta

        # Operator sequence cutoff
        self.M = max(10, int(2 * beta * self.N))

        # System state arrays
        self.alpha_i = np.zeros((L, L), dtype=np.int8)
        self.seq_i = np.zeros((self.M, 3), dtype=np.int32)

        # Observables and counters
        self.n_operators = 0
        self.n_diag_ops = 0
        self.n_offdiag_ops = 0

        # Precompute nearest neighbor table for fast memory access
        self.neighbor_table = np.zeros((self.N, 4), dtype=np.int32)
        for i in range(L):
            for j in range(L):
                site = i * L + j
                self.neighbor_table[site, 0] = ((i - 1) % L) * L + j
                self.neighbor_table[site, 1] = ((i + 1) % L) * L + j
                self.neighbor_table[site, 2] = i * L + ((j - 1) % L)
                self.neighbor_table[site, 3] = i * L + ((j + 1) % L)

        self._nonid_positions = None 
        self._site_op_positions = None 
        
        # Ensure contiguous memory layout for zero-copy Numba integration
        self.alpha_i = np.ascontiguousarray(self.alpha_i)

    def _get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Restores 2D neighbor coordinates from the 1D table."""
        site = i * self.L + j
        neighbors = []
        for k in range(4):
            n_site = self.neighbor_table[site, k]
            ni = n_site // self.L 
            nj = n_site % self.L  
            neighbors.append((ni, nj))
        return neighbors

    def Xi_at_site(self, alpha_slice: np.ndarray, i: int, j: int) -> int:
        cnt = 0
        for ni, nj in self._get_neighbors(i, j):
            cnt += alpha_slice[ni, nj]
        return 1 if cnt == 1 else 0

    def diag_weight(self, alpha_slice: np.ndarray, i: int, j: int) -> float:
        state = alpha_slice[i, j]
        return (self.C + self.eta) if state == 0 else self.C

    def offdiag_weight(self, alpha_slice: np.ndarray, i: int, j: int) -> float:
        return self.J * self.Xi_at_site(alpha_slice, i, j)

    def _op_weight(self, alpha_slice: np.ndarray, i: int, j: int, op_type: int) -> float:
        if op_type == OP_DIAG:
            return self.diag_weight(alpha_slice, i, j)
        elif op_type == OP_OFFDIAG:
            return self.offdiag_weight(alpha_slice, i, j)
        return 1.0

    @staticmethod
    def _toggle_nonid_type(op_type: int) -> int:
        if op_type == OP_DIAG: return OP_OFFDIAG
        elif op_type == OP_OFFDIAG: return OP_DIAG
        return op_type

    def propagate_alpha_to_p(self, p_target: int) -> np.ndarray:
        """Propagates state in Python space (primarily for boundary checks)."""
        alpha_work = self.alpha_i.copy()
        for p in range(p_target):
            if self.seq_i[p, 0] == OP_OFFDIAG:
                i, j = self.seq_i[p, 1], self.seq_i[p, 2]
                alpha_work[i, j] ^= 1
        return alpha_work

    def initialize_state(self, seed_positions: List[Tuple[int, int]] = None):
        """Sets the initial seed configuration for the simulation."""
        if seed_positions is None:
            seed_positions = [(self.L // 2, self.L // 2)]

        self.alpha_i.fill(0)
        for i, j in seed_positions:
            self.alpha_i[i, j] = 1

        self.seq_i.fill(0)
        self.n_operators = 0
        self.n_diag_ops = 0
        self.n_offdiag_ops = 0

        self._nonid_positions = None
        self._site_op_positions = None

    def diagonal_update(self):
        """Passes execution to the Numba backend for diagonal update."""
        alpha_flat = self.alpha_i.reshape(-1)
        fast_diagonal_update(
            self.seq_i, alpha_flat, self.M, self.N, self.L, self.beta, self.C, self.eta
        )
        self._count_operators()
        self._build_operator_caches()

    def _count_operators(self):
        self.n_diag_ops = int(np.sum(self.seq_i[:, 0] == OP_DIAG))
        self.n_offdiag_ops = int(np.sum(self.seq_i[:, 0] == OP_OFFDIAG))
        self.n_operators = self.n_diag_ops + self.n_offdiag_ops

    def _build_operator_caches(self):
        """
        Constructs the static linked list over 1D NumPy arrays, ensuring 
        O(1) access time for segment updates within the JIT compiler.
        """
        nonid = np.flatnonzero(self.seq_i[:, 0] != OP_IDENTITY).astype(np.int32)
        self._nonid_positions = nonid

        self._first_op_at_site = np.full(self.N, -1, dtype=np.int32)
        self._last_op_at_site = np.full(self.N, -1, dtype=np.int32)
        self._next_op = np.full(self.M, -1, dtype=np.int32)

        for p in nonid:
            i = self.seq_i[p, 1]
            j = self.seq_i[p, 2]
            site_idx = i * self.L + j 

            if self._first_op_at_site[site_idx] == -1:
                self._first_op_at_site[site_idx] = p
            else:
                prev_p = self._last_op_at_site[site_idx]
                self._next_op[prev_p] = p

            self._last_op_at_site[site_idx] = p

    def loop_update_sweep(self, n_loops: int) -> int:
        """
        Executes a full sweep of segment updates using a zero-copy data passing 
        mechanism to the backend.
        """
        if self._nonid_positions is None or self._first_op_at_site is None:
            self._build_operator_caches()

        # Zero-copy view generation
        alpha_flat = self.alpha_i.reshape(-1)

        accepts, diag_diff, offdiag_diff = fast_loop_sweep(
            self.seq_i, alpha_flat, self._nonid_positions, 
            self._first_op_at_site, self._next_op, self.neighbor_table,
            self.M, self.L, self.C, self.eta, self.J, n_loops
        )
        
        self.n_diag_ops += diag_diff
        self.n_offdiag_ops += offdiag_diff
        self.n_operators = self.n_diag_ops + self.n_offdiag_ops
        
        return accepts

    def check_all_sigma_allowed(self) -> bool:
        alpha = self.alpha_i.copy()
        for p in range(self.M):
            if self.seq_i[p, 0] == OP_OFFDIAG:
                i, j = self.seq_i[p, 1], self.seq_i[p, 2]
                if self.Xi_at_site(alpha, i, j) != 1:
                    return False
                alpha[i, j] ^= 1
        return np.array_equal(alpha, self.alpha_i)

    def check_periodicity(self) -> bool:
        alpha_final = self.propagate_alpha_to_p(self.M)
        return np.array_equal(alpha_final, self.alpha_i)

    def adjust_cutoff(self):
        """Dynamically adjusts the truncation length M."""
        if self.n_operators > 0:
            new_M = max(10, int(1.5 * self.n_operators) + 10, self.M)
            if new_M > self.M:
                append_length = new_M - self.M
                append_seq = np.zeros((append_length, 3), dtype=np.int32)
                self.seq_i = np.concatenate([self.seq_i, append_seq], axis=0)
                self.M = new_M

                self._nonid_positions = None
                self._site_op_positions = None

    def rotate_to_middle(self):
        """Performs a cyclic rotation of the operator sequence to minimize autocorrelation."""
        mid = self.M // 2
        alpha_mid = self.propagate_alpha_to_p(mid)
        self.alpha_i = alpha_mid
        self.seq_i = np.roll(self.seq_i, -mid, axis=0)

        self._nonid_positions = None
        self._site_op_positions = None

    def energy_per_site_phys(self) -> float:
        return (self.C + self.eta) - self.n_operators / (self.beta * self.N)

    def density_timeavg_fast(self) -> float:
        """
        Improved estimator for Rydberg state density using time-averaging.
        Algorithmic complexity reduces from O(M * N) to O(N + n_offdiag).
        """
        M = self.M
        N = self.N

        alpha = self.alpha_i.copy()
        total_ones = int(alpha.sum())

        off_idx = np.where(self.seq_i[:, 0] == OP_OFFDIAG)[0]
        if off_idx.size == 0:
            return total_ones / N

        accum = total_ones
        prev = 0

        for p in off_idx:
            accum += (p - prev) * total_ones
            i, j = int(self.seq_i[p, 1]), int(self.seq_i[p, 2])

            if alpha[i, j] == 1:
                total_ones -= 1
            else:
                total_ones += 1
            alpha[i, j] ^= 1

            prev = int(p)

        accum += (M - prev) * total_ones
        return accum / (M * N)

    @staticmethod
    def blocking_binning(samples, min_bins: int = 16):
        """Performs blocking analysis to evaluate the standard error of correlated samples."""
        x = np.asarray(samples, dtype=np.float64)
        n = len(x)

        if n < 2 * min_bins:
            mean = float(x.mean()) if n > 0 else float("nan")
            stderr = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            return [(1, n, mean, stderr)], (1, n, mean, stderr)

        results = []
        bin_size = 1
        while n // bin_size >= min_bins:
            n_bins = n // bin_size
            trimmed = x[:n_bins * bin_size]
            binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
            
            mean = float(binned.mean())
            stderr = float(binned.std(ddof=1) / np.sqrt(n_bins))
            results.append((bin_size, n_bins, mean, stderr))
            bin_size *= 2

        return results, results[-1]

    def measure_observables(self) -> Dict:
        energy_used = -self.n_operators / self.beta
        energy_per_site = self.energy_per_site_phys()
        density = self.density_timeavg_fast()
        n_clusters = self.count_clusters()

        return {
            'energy_used': energy_used,
            'energy_per_site': energy_per_site,
            'density': density,
            'n_operators': self.n_operators,
            'n_diag_ops': self.n_diag_ops,
            'n_offdiag_ops': self.n_offdiag_ops,
            'n_clusters': n_clusters
        }

    def count_clusters(self) -> int:
        """Counts independent connected components of Rydberg excitations."""
        visited = np.zeros((self.L, self.L), dtype=bool)
        n_clusters = 0

        def dfs(i, j):
            stack = [(i, j)]
            while stack:
                ci, cj = stack.pop()
                if visited[ci, cj]:
                    continue
                visited[ci, cj] = True
                for ni, nj in self._get_neighbors(ci, cj):
                    if self.alpha_i[ni, nj] == 1 and not visited[ni, nj]:
                        stack.append((ni, nj))

        for i in range(self.L):
            for j in range(self.L):
                if self.alpha_i[i, j] == 1 and not visited[i, j]:
                    dfs(i, j)
                    n_clusters += 1

        return n_clusters
    
    def run_mc_steps(self, n_sweeps: int = 1000,
                     n_thermalize: int = 100,
                     measure_interval: int = 10,
                     n_loops_per_sweep: int = None,
                     verbose: bool = True) -> Dict:
        """Main Monte Carlo pipeline executing thermalization and measurements."""
        if n_loops_per_sweep is None:
            n_loops_per_sweep = self.N * 5

        measurements = {
            'energy': [], 'energy_per_site': [], 'density': [],
            'n_operators': [], 'n_diag_ops': [], 'n_offdiag_ops': [],
            'n_clusters': [], 'loop_accept_rate': []
        }

        total_loop_attempts = 0
        total_loop_accepts = 0

        for sweep in range(n_sweeps):
            self.diagonal_update()

            loop_accepts = self.loop_update_sweep(n_loops_per_sweep)
            total_loop_attempts += n_loops_per_sweep
            total_loop_accepts += loop_accepts

            if sweep % 100 == 0:
                self.adjust_cutoff()

            self.rotate_to_middle()

            if sweep >= n_thermalize and sweep % measure_interval == 0:
                obs = self.measure_observables()
                measurements['energy'].append(obs['energy_used'])
                measurements['energy_per_site'].append(obs['energy_per_site'])
                measurements['density'].append(obs['density'])
                measurements['n_operators'].append(obs['n_operators'])
                measurements['n_diag_ops'].append(obs['n_diag_ops'])
                measurements['n_offdiag_ops'].append(obs['n_offdiag_ops'])
                measurements['n_clusters'].append(obs['n_clusters'])
                
                loop_rate = loop_accepts / n_loops_per_sweep if n_loops_per_sweep > 0 else 0.0
                measurements['loop_accept_rate'].append(loop_rate)

            if verbose and sweep % (10 * measure_interval) == 0:
                if sweep < n_thermalize:
                    print(f"Thermalization [Warming up...] Sweep {sweep}/{n_thermalize}")
                else:
                    print(
                        f"Sweep {sweep}/{n_sweeps}: "
                        f"E/N = {obs['energy_per_site']:.6f}, "
                        f"ρ = {obs['density']:.4f}, "
                        f"n_ops = {obs['n_operators']} "
                        f"(diag: {obs['n_diag_ops']}, offdiag: {obs['n_offdiag_ops']}), "
                        f"clusters = {obs['n_clusters']}, "
                        f"loop_acc = {loop_rate:.3f}"
                    )

        if verbose:
            overall_loop_rate = total_loop_accepts / total_loop_attempts if total_loop_attempts > 0 else 0.0
            print(f"\nSimulation complete.")
            print(f"Overall loop acceptance rate: {overall_loop_rate:.4f}")
            print(f"Periodicity check: {self.check_periodicity()}")

        e_table, e_best = self.blocking_binning(measurements['energy_per_site'], min_bins=16)
        rho_table, rho_best = self.blocking_binning(measurements['density'], min_bins=16)

        measurements['stats'] = {
            'energy_per_site_mean': e_best[2],
            'energy_per_site_stderr': e_best[3],
            'density_mean': rho_best[2],
            'density_stderr': rho_best[3],
            'energy_blocking_table': e_table,
            'density_blocking_table': rho_table
        }

        if verbose:
            bs, nb, m, se = e_best
            print("\n=== Blocking/Binning (use last line as conservative) ===")
            print(f"E/N : mean = {m:.6f}, stderr = {se:.6f}  (bin_size={bs}, n_bins={nb})")
            bs, nb, m, se = rho_best
            print(f"ρ   : mean = {m:.6f}, stderr = {se:.6f}  (bin_size={bs}, n_bins={nb})")

        return measurements    