"""Problem 1: simulate dynamics for small and large noise."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vicsek import VicsekParams, initialize_state, run_with_polarization_history

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PARAMS = VicsekParams()
SEED = 42
TOTAL_STEPS = 800
ETA_SMALL = 0.1
ETA_LARGE = 0.9
SNAPSHOT_STEPS = (0, 200, 400, 800)


def plot_snapshots(
    positions: np.ndarray,
    angles: np.ndarray,
    params: VicsekParams,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(positions[:, 0], positions[:, 1], s=18, c="tab:blue", alpha=0.8)

    arrow_scale = 0.2
    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        np.cos(angles),
        np.sin(angles),
        angles="xy",
        scale_units="xy",
        scale=1.0 / arrow_scale,
        width=0.003,
        color="tab:red",
        alpha=0.7,
    )

    ax.set_xlim(0, params.L)
    ax.set_ylim(0, params.L)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_polarization_history(
    histories: dict[str, np.ndarray],
    params: VicsekParams,
    output_path: Path,
) -> None:
    first_history = next(iter(histories.values()))
    time = np.arange(1, first_history.shape[0] + 1) * params.dt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, history in histories.items():
        ax.plot(time, history, label=label, linewidth=1.5)

    ax.set_xlabel("time")
    ax.set_ylabel("P(t)")
    ax.set_title("Instantaneous polarization")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def capture_snapshots_for_eta(
    eta: float,
    params: VicsekParams,
    rng: np.random.Generator,
    initial_positions: np.ndarray,
    initial_angles: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    from vicsek import step

    positions = initial_positions.copy()
    angles = initial_angles.copy()
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {0: (positions.copy(), angles.copy())}

    for step_idx in range(TOTAL_STEPS):
        positions, angles = step(positions, angles, params, eta, rng)
        if step_idx + 1 in SNAPSHOT_STEPS[1:]:
            snapshots[step_idx + 1] = (positions.copy(), angles.copy())

    return snapshots


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    initial_positions, initial_angles = initialize_state(PARAMS, rng)

    histories = {}
    for eta, label in ((ETA_SMALL, "small"), (ETA_LARGE, "large")):
        _, _, history = run_with_polarization_history(
            PARAMS,
            eta,
            TOTAL_STEPS,
            seed=SEED,
            initial_positions=initial_positions,
            initial_angles=initial_angles,
        )
        histories[label] = history

        snapshots = capture_snapshots_for_eta(
            eta,
            PARAMS,
            np.random.default_rng(SEED),
            initial_positions,
            initial_angles,
        )
        for step_idx, (positions, angles) in snapshots.items():
            time = step_idx * PARAMS.dt
            plot_snapshots(
                positions,
                angles,
                PARAMS,
                f"eta={eta}, t={time:.1f}",
                OUTPUT_DIR / f"problem1_eta_{eta}_t_{step_idx}.png",
            )

    plot_polarization_history(
        {f"eta={ETA_SMALL}": histories["small"], f"eta={ETA_LARGE}": histories["large"]},
        PARAMS,
        OUTPUT_DIR / "problem1_polarization_history.png",
    )

    print("Problem 1 complete. Outputs written to outputs/")


if __name__ == "__main__":
    main()
