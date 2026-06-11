"""Problem 2: stationary order parameter psi(eta) and transition characterization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vicsek import VicsekParams, estimate_eta_c, initialize_state, polarization, step

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PARAMS = VicsekParams()
ETA_VALUES = np.linspace(0.0, 1.0, 51)
N_SEEDS = 3
BASE_SEED = 7


def run_hysteresis_sweep(
    params: VicsekParams,
    eta_values: np.ndarray,
    *,
    seed: int,
    start_ordered: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if start_ordered:
        positions = rng.uniform(0.0, params.L, size=(params.N, 2))
        angles = np.zeros(params.N)
    else:
        positions, angles = initialize_state(params, rng)

    psi_values = np.zeros_like(eta_values, dtype=float)
    burn_in_steps = 300
    sample_steps = 400
    sample_every = 4

    for idx, eta in enumerate(eta_values):
        for _ in range(burn_in_steps):
            positions, angles = step(positions, angles, params, float(eta), rng)

        samples = []
        for step_idx in range(sample_steps):
            positions, angles = step(positions, angles, params, float(eta), rng)
            if (step_idx + 1) % sample_every == 0:
                samples.append(polarization(angles))

        psi_values[idx] = float(np.mean(samples))

    return psi_values


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    psi_curve, eta_c = estimate_eta_c(
        PARAMS,
        ETA_VALUES,
        n_seeds=N_SEEDS,
        base_seed=BASE_SEED,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ETA_VALUES, psi_curve, "o-", linewidth=1.5, markersize=4, label="psi(eta)")
    if eta_c is not None:
        ax.axvline(eta_c, color="tab:red", linestyle="--", linewidth=1.2, label=f"eta_c ~ {eta_c:.3f}")
    ax.axhline(0.5, color="tab:gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("noise eta")
    ax.set_ylabel("stationary order parameter psi")
    ax.set_title("Order parameter vs noise (N=125, R=1, v=1)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "problem2_psi_vs_eta.png", dpi=150)
    plt.close(fig)

    eta_up = ETA_VALUES
    eta_down = ETA_VALUES[::-1]
    psi_increasing = run_hysteresis_sweep(PARAMS, eta_up, seed=21, start_ordered=True)
    psi_decreasing = run_hysteresis_sweep(PARAMS, eta_down, seed=22, start_ordered=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eta_up, psi_increasing, "o-", label="eta increasing", markersize=4)
    ax.plot(eta_down, psi_decreasing, "s-", label="eta decreasing", markersize=4)
    ax.set_xlabel("noise eta")
    ax.set_ylabel("stationary order parameter psi")
    ax.set_title("Hysteresis check (ordered initial condition)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "problem2_hysteresis.png", dpi=150)
    plt.close(fig)

    print(f"Problem 2 complete. Estimated eta_c ~ {eta_c:.3f}" if eta_c is not None else "Problem 2 complete.")
    print("Outputs written to outputs/")


if __name__ == "__main__":
    main()
