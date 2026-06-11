"""Train a lightweight surrogate model: f(R, v) -> eta_c from simulation labels."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vicsek import VicsekParams, estimate_eta_c

SURROGATE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SURROGATE_DIR / "outputs"
PROBLEM3_RESULTS = ROOT / "outputs" / "problem3_results.txt"

TARGET_ETA_C = 0.5
MIN_LABELS_FOR_SPLIT = 8
R_BOUNDS = (0.3, 1.5)
V_BOUNDS = (0.5, 1.8)


@dataclass(frozen=True)
class GridConfig:
    name: str
    r_points: int
    v_points: int
    eta_values: np.ndarray
    n_seeds: int
    burn_in_steps: int
    sample_steps: int


QUICK_CONFIG = GridConfig(
    name="quick",
    r_points=5,
    v_points=5,
    eta_values=np.linspace(0.25, 0.75, 9),
    n_seeds=1,
    burn_in_steps=80,
    sample_steps=150,
)

DEFAULT_CONFIG = GridConfig(
    name="default",
    r_points=7,
    v_points=7,
    eta_values=np.linspace(0.2, 0.8, 11),
    n_seeds=1,
    burn_in_steps=100,
    sample_steps=250,
)

ACCURATE_CONFIG = GridConfig(
    name="accurate",
    r_points=7,
    v_points=7,
    eta_values=np.linspace(0.2, 0.8, 13),
    n_seeds=2,
    burn_in_steps=150,
    sample_steps=300,
)


def dataset_path_for(config: GridConfig) -> Path:
    return OUTPUT_DIR / f"surrogate_dataset_{config.name}.csv"


def results_path_for(config: GridConfig) -> Path:
    return OUTPUT_DIR / f"surrogate_model_results_{config.name}.txt"


def heatmap_path_for(config: GridConfig) -> Path:
    return OUTPUT_DIR / f"surrogate_eta_c_heatmap_{config.name}.png"


def make_rv_grid(config: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    R_values = np.linspace(R_BOUNDS[0], R_BOUNDS[1], config.r_points)
    v_values = np.linspace(V_BOUNDS[0], V_BOUNDS[1], config.v_points)
    return R_values, v_values


def seed_for_pair(index: int) -> int:
    return 1000 + index * 17


def generate_dataset(config: GridConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    R_values, v_values = make_rv_grid(config)
    candidate_count = len(R_values) * len(v_values)
    rows_R: list[float] = []
    rows_v: list[float] = []
    rows_eta_c: list[float] = []
    skipped = 0

    print(f"Generating dataset ({config.name}): {config.r_points}x{config.v_points} grid", flush=True)
    pair_index = 0
    for R in R_values:
        for v in v_values:
            params = VicsekParams(R=float(R), v=float(v))
            base_seed = seed_for_pair(pair_index)
            _, eta_c = estimate_eta_c(
                params,
                config.eta_values,
                n_seeds=config.n_seeds,
                base_seed=base_seed,
                burn_in_steps=config.burn_in_steps,
                sample_steps=config.sample_steps,
            )
            pair_index += 1
            if eta_c is None:
                skipped += 1
                print(f"  R={R:.2f}, v={v:.2f} -> no crossing (skipped)", flush=True)
                continue
            rows_R.append(float(R))
            rows_v.append(float(v))
            rows_eta_c.append(float(eta_c))
            print(f"  R={R:.2f}, v={v:.2f} -> eta_c={eta_c:.4f}", flush=True)

    return (
        np.array(rows_R),
        np.array(rows_v),
        np.array(rows_eta_c),
        candidate_count,
        skipped,
    )


def save_dataset(R: np.ndarray, v: np.ndarray, eta_c: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["R", "v", "eta_c"])
        for r_val, v_val, eta_val in zip(R, v, eta_c, strict=True):
            writer.writerow([f"{r_val:.6f}", f"{v_val:.6f}", f"{eta_val:.6f}"])


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1], data[:, 2]


def load_problem3_point() -> tuple[float, float] | None:
    if not PROBLEM3_RESULTS.exists():
        return None

    values: dict[str, float] = {}
    for line in PROBLEM3_RESULTS.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue

        key, raw = line.split("=", 1)
        raw = raw.strip()
        if raw == "None":
            continue

        try:
            values[key.strip()] = float(raw)
        except ValueError:
            continue

    if "optimized_R" in values and "optimized_v" in values:
        return values["optimized_R"], values["optimized_v"]
    return None


def train_models(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[dict[str, dict[str, float]], str, object]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models: dict[str, object] = {
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=100,
            random_state=42,
        ),
    }

    scores: dict[str, dict[str, float]] = {}
    best_name = ""
    best_test_mae = float("inf")
    best_model: object | None = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_mae = mean_absolute_error(y_train, model.predict(X_train))
        test_mae = mean_absolute_error(y_test, model.predict(X_test))
        scores[name] = {"train_mae": train_mae, "test_mae": test_mae}
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            best_name = name
            best_model = model

    assert best_model is not None
    return scores, best_name, best_model


def find_best_on_mesh(
    r_mesh: np.ndarray,
    v_mesh: np.ndarray,
    preds: np.ndarray,
) -> tuple[float, float, float]:
    flat_idx = int(np.argmin(np.abs(preds - TARGET_ETA_C)))
    v_idx, r_idx = np.unravel_index(flat_idx, preds.shape)
    return float(r_mesh[v_idx, r_idx]), float(v_mesh[v_idx, r_idx]), float(preds[v_idx, r_idx])


def plot_heatmap(
    model: object,
    R: np.ndarray,
    v: np.ndarray,
    eta_c: np.ndarray,
    best_name: str,
    surrogate_best: tuple[float, float, float],
    heatmap_path: Path,
) -> None:
    r_lin = np.linspace(R_BOUNDS[0], R_BOUNDS[1], 50)
    v_lin = np.linspace(V_BOUNDS[0], V_BOUNDS[1], 50)
    r_mesh, v_mesh = np.meshgrid(r_lin, v_lin)
    grid = np.column_stack([r_mesh.ravel(), v_mesh.ravel()])
    preds = model.predict(grid).reshape(r_mesh.shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    contour = ax.contourf(r_mesh, v_mesh, preds, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="predicted eta_c")
    if preds.min() <= TARGET_ETA_C <= preds.max():
        ax.contour(r_mesh, v_mesh, preds, levels=[TARGET_ETA_C], colors="white", linewidths=2.0)
    ax.scatter(R, v, c=eta_c, cmap="viridis", edgecolors="black", linewidths=0.4, s=28, label="sim labels")

    problem3 = load_problem3_point()
    if problem3 is not None:
        ax.scatter(
            [problem3[0]],
            [problem3[1]],
            marker="*",
            s=180,
            c="red",
            edgecolors="white",
            linewidths=0.8,
            label=f"Problem 3 (R={problem3[0]:.2f}, v={problem3[1]:.2f})",
            zorder=5,
        )

    ax.scatter(
        [surrogate_best[0]],
        [surrogate_best[1]],
        marker="X",
        s=120,
        c="orange",
        edgecolors="black",
        linewidths=0.6,
        label=f"surrogate best (eta_c~{surrogate_best[2]:.3f})",
        zorder=5,
    )

    ax.set_xlabel("interaction radius R")
    ax.set_ylabel("speed v")
    ax.set_title(f"Surrogate predicted eta_c ({best_name})")
    ax.grid(True, alpha=0.25, color="white")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)


def write_results(
    *,
    results_path: Path,
    config_name: str,
    candidate_grid_size: int,
    valid_dataset_size: int,
    skipped_no_crossing: int,
    scores: dict[str, dict[str, float]],
    best_name: str,
    surrogate_best: tuple[float, float, float],
) -> None:
    lines = [
        f"grid_mode={config_name}",
        f"candidate_grid_size={candidate_grid_size}",
        f"valid_dataset_size={valid_dataset_size}",
        f"skipped_no_crossing={skipped_no_crossing}",
        "note=Labels are stochastic; held-out set is small.",
        "",
    ]
    for name, metric in scores.items():
        lines.append(f"{name}_train_mae={metric['train_mae']:.6f}")
        lines.append(f"{name}_test_mae={metric['test_mae']:.6f}")
    lines.extend(
        [
            "",
            f"best_model={best_name}",
            f"surrogate_best_R={surrogate_best[0]:.6f}",
            f"surrogate_best_v={surrogate_best[1]:.6f}",
            f"surrogate_predicted_eta_c={surrogate_best[2]:.6f}",
        ]
    )
    problem3 = load_problem3_point()
    if problem3 is not None:
        lines.append(f"problem3_optimized_R={problem3[0]:.6f}")
        lines.append(f"problem3_optimized_v={problem3[1]:.6f}")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surrogate f(R, v) -> eta_c")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="5x5 smoke-test grid")
    mode.add_argument("--accurate", action="store_true", help="Heavier 7x7 grid with 2 seeds")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of the mode-specific dataset CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        config = QUICK_CONFIG
    elif args.accurate:
        config = ACCURATE_CONFIG
    else:
        config = DEFAULT_CONFIG

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_path_for(config)
    results_path = results_path_for(config)
    heatmap_path = heatmap_path_for(config)
    candidate_grid_size = config.r_points * config.v_points
    skipped_no_crossing = 0
    must_generate = args.regenerate or args.quick or args.accurate

    if dataset_path.exists() and not must_generate:
        print(f"Loading existing dataset from {dataset_path}", flush=True)
        R, v, eta_c = load_dataset(dataset_path)
        valid_dataset_size = len(eta_c)
    else:
        if dataset_path.exists() and not args.regenerate:
            print("Mode-specific run: regenerating dataset for this grid configuration.", flush=True)
        R, v, eta_c, candidate_grid_size, skipped_no_crossing = generate_dataset(config)
        valid_dataset_size = len(eta_c)
        if valid_dataset_size == 0:
            raise RuntimeError("No valid eta_c labels generated; widen eta sweep or adjust grid.")
        save_dataset(R, v, eta_c, dataset_path)
        print(f"Saved {valid_dataset_size} rows to {dataset_path}", flush=True)

    if valid_dataset_size < MIN_LABELS_FOR_SPLIT:
        raise RuntimeError(
            f"Too few valid eta_c labels ({valid_dataset_size}) for train/test split; "
            f"need at least {MIN_LABELS_FOR_SPLIT}."
        )

    X = np.column_stack([R, v])
    y = eta_c

    scores, best_name, best_model = train_models(X, y)

    r_lin = np.linspace(R_BOUNDS[0], R_BOUNDS[1], 50)
    v_lin = np.linspace(V_BOUNDS[0], V_BOUNDS[1], 50)
    r_mesh, v_mesh = np.meshgrid(r_lin, v_lin)
    grid = np.column_stack([r_mesh.ravel(), v_mesh.ravel()])
    preds = best_model.predict(grid).reshape(r_mesh.shape)
    surrogate_best = find_best_on_mesh(r_mesh, v_mesh, preds)

    plot_heatmap(best_model, R, v, eta_c, best_name, surrogate_best, heatmap_path)
    write_results(
        results_path=results_path,
        config_name=config.name,
        candidate_grid_size=candidate_grid_size,
        valid_dataset_size=valid_dataset_size,
        skipped_no_crossing=skipped_no_crossing,
        scores=scores,
        best_name=best_name,
        surrogate_best=surrogate_best,
    )

    print(f"\nBest model: {best_name}", flush=True)
    print(f"Test MAE: {scores[best_name]['test_mae']:.4f}", flush=True)
    print(f"Surrogate best: R={surrogate_best[0]:.3f}, v={surrogate_best[1]:.3f}", flush=True)
    print(f"Outputs written to {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
