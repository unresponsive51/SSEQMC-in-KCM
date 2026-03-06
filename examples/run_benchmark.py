import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sseqmc_core.simulator import RydbergSSE


def main():
    """Entry point for executing the simulation benchmark."""
    plt.rcParams['font.family'] = 'DejaVu Sans'

    L = 5   
    beta = 32.0
    J = 1.0
    mu = -1.0

    print("=" * 60)
    print("Rydberg Atom Array SSE Simulation")
    print("=" * 60)
    print(f"System size: {L} × {L} = {L*L} sites")
    print(f"Inverse temperature: β = {beta}")
    print(f"Parameters: J = {J}, μ = {mu}, η = {-mu}")
    print("=" * 60)

    simulator = RydbergSSE(L=L, beta=beta, J=J, mu=mu)
    simulator.initialize_state(seed_positions=[(L // 2, L // 2)])

    print("\nStarting Monte Carlo simulation...")

    measurements = simulator.run_mc_steps(
        n_sweeps=100000,
        n_thermalize=10000,
        measure_interval=50,
        n_loops_per_sweep=L * L,
        verbose=True
    )

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(measurements['energy'])
    axes[0, 0].set_xlabel('Measurement')
    axes[0, 0].set_ylabel('Energy (used, incl. shift)')
    axes[0, 0].set_title('Energy Evolution (counter estimator, shifted)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(measurements['density'])
    axes[0, 1].set_xlabel('Measurement')
    axes[0, 1].set_ylabel('Density ρ (time-avg)')
    axes[0, 1].set_title('Rydberg State Density (improved estimator)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(measurements['n_operators'], label='Total', linewidth=2)
    axes[1, 0].plot(measurements['n_diag_ops'], label='Diagonal', linestyle='--')
    axes[1, 0].plot(measurements['n_offdiag_ops'], label='Off-diagonal', linestyle=':')
    axes[1, 0].set_xlabel('Measurement')
    axes[1, 0].set_ylabel('Number of operators')
    axes[1, 0].set_title('Number of Operators')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    im = axes[1, 1].imshow(simulator.alpha_i, cmap='RdBu', interpolation='nearest')
    axes[1, 1].set_title('Final Configuration (alpha_i slice)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im, ax=axes[1, 1], label='State (0=|g⟩, 1=|r⟩)')

    plt.tight_layout()
    plt.savefig('rydberg_sse_results.png', dpi=300)
    print("\nResults saved to rydberg_sse_results.png")

    np.savetxt("final_configuration.txt", simulator.alpha_i, fmt='%d')
    print("Final configuration saved to final_configuration.txt")

    stats = measurements['stats']
    print("\n=== Final estimates with error bars (blocking) ===")
    print(f"E/N = {stats['energy_per_site_mean']:.6f} ± {stats['energy_per_site_stderr']:.6f}")
    print(f"ρ   = {stats['density_mean']:.6f} ± {stats['density_stderr']:.6f}")

if __name__ == "__main__":
    main()