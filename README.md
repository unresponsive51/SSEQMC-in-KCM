# Rydberg SSE-QMC Solver

A Stochastic Series Expansion (SSE) Quantum Monte Carlo solver for Rydberg atom arrays.

This program implements an SSE-QMC algorithm with kinetic constraints (facilitation/blockade mechanisms) on a 2D square lattice under Periodic Boundary Conditions (PBC). The codebase adopts a hybrid architecture: a Python-based object-oriented frontend handles state management and physical measurements, while Numba JIT-compiled backends execute the performance-critical Metropolis updates.

## Background & Acknowledgement

This program was developed during the study of the Kinetic Constraint Model (KCM) QMC methods presented in the following research:

> **[Quantum slush state in Rydberg atom arrays](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.132.206503)** (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.132.206503) > *Physical Review Letters 132, 206503 (2024)*

I would like to express my sincere gratitude to the paper's author, **Tengzhou Zhang**, for his guidance and insightful discussions which greatly aided the implementation of this code.

## Implementation Notes

To ensure computational efficiency, the following optimizations are implemented in the JIT backend:

- **Kernel Fusion:** The segment update sweeps are fused into a single JIT-compiled kernel to minimize Python-C++ boundary crossing overhead.
- **Memory Management:** Contiguous memory layouts (`np.ascontiguousarray`) and NumPy views are utilized to allow the backend to mutate the 2D lattice state directly, avoiding data copying overhead (Zero-copy).
- **Data Structures:** Standard Python dictionaries are replaced with a 1D array-based static linked list (`first_op`, `next_op`), ensuring strictly $O(1)$ operator traversal during JIT execution.
- **Error Analysis:** Hierarchical blocking/binning algorithms are included to accurately estimate the standard error of correlated Markov Chain samples.

## Project Structure

```text
Rydberg-SSE-QMC/
├── rydberg_sse/
│   ├── __init__.py
│   ├── kernels.py       # Numba JIT-compiled compute kernels
│   └── simulator.py     # OOP frontend & measurement logic
├── examples/
│   └── run_benchmark.py # Main execution script
├── requirements.txt
└── README.md
```

## Usage

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the simulation:**
```bash
python examples/run_benchmark.py
```