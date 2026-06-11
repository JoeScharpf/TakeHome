"""Problem 3: optimize R and v so the critical noise eta_c is 0.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vicsek import VicsekParams, estimate_eta_c

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
TARGET_ETA_C = 0.5
BASE_SEED = 99
PENALTY = 10.0

COARSE_R = np.array([0.3, 0.5, 0.8, 1.1, 1.5])
COARSE_V = np.array([0.5, 0.8, 1.1, 1.4, 1.8])
COARSE_ETA = np.linspace(0.25, 0.75, 11)

REFINE_SPAN = 0.2
REFINE_POINTS = 5
REFINE_ETA = np.linspace(0.35, 0.65, 11)

FINAL_ETA = np.linspace(0.2, 0.8, 15)


@dataclass
class SearchConfig:
    eta_values: np.ndarray
    n_seeds: int
    burn_in_steps: int
    sample_steps: int


@dataclass
class CandidateResult:
    R: float
    v: float
    eta_c: float | None
    objective: float
    psi_curve: np.ndarray | None = None


def evaluate_candidate(
    R: float,
    v: float,
    config: SearchConfig,
    *,
    base_seed: int = BASE_SEED,
) -> CandidateResult:
    params = VicsekParams(R=R, v=v)
    psi_curve, eta_c = estimate_eta_c(
        params,
        config.eta_values,
        n_seeds=config.n_seeds,
        base_seed=base_seed,
        burn_in_steps=config.burn_in_steps,
        sample_steps=config.sample_steps,
    )
    objective = abs(eta_c - TARGET_ETA_C) if eta_c is not None else PENALTY
    return CandidateResult(R=R, v=v, eta_c=eta_c, objective=objective, psi_curve=psi_curve)


def search_grid(
    R_values: np.ndarray,
    v_values: np.ndarray,
    config: SearchConfig,
    *,
    phase_name: str,
) -> tuple[CandidateResult, list[CandidateResult]]:
    results: list[CandidateResult] = []
    best = CandidateResult(R=float(R_values[0]), v=float(v_values[0]), eta_c=None, objective=np.inf)

    print(f"{phase_name}...")
    for R in R_values:
        for v in v_values:
            result = evaluate_candidate(R, v, config)
            results.append(result)
            print(
                f"  R={R:.2f}, v={v:.2f} -> eta_c={result.eta_c}, objective={result.objective:.4f}",
                flush=True,
            )
            if result.objective < best.objective:
                best = result

    return best, results


def make_refine_grid(best_R: float, best_v: float) -> tuple[np.ndarray, np.ndarray]:
    R_values = np.unique(np.clip(np.linspace(best_R - REFINE_SPAN, best_R + REFINE_SPAN, REFINE_POINTS), 0.1, 2.5))
    v_values = np.unique(np.clip(np.linspace(best_v - REFINE_SPAN, best_v + REFINE_SPAN, REFINE_POINTS), 0.1, 2.5))
    return R_values, v_values


def save_search_log(path: Path, coarse_results: list[CandidateResult], refine_results: list[CandidateResult]) -> None:
    def format_section(title: str, results: list[CandidateResult]) -> str:
        lines = [title]
        ranked = sorted(results, key=lambda item: item.objective)[:5]
        for item in ranked:
            lines.append(
                f"R={item.R:.3f}, v={item.v:.3f}, eta_c={item.eta_c}, objective={item.objective:.4f}"
            )
        return "\n".join(lines)

    content = "\n\n".join(
        [
            format_section("Top coarse candidates:", coarse_results),
            format_section("Top refined candidates:", refine_results),
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def plot_final_result(best: CandidateResult, eta_values: np.ndarray, psi_curve: np.ndarray, eta_c: float | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eta_values, psi_curve, "o-", linewidth=1.5, markersize=4)
    ax.axvline(TARGET_ETA_C, color="tab:gray", linestyle=":", linewidth=1.0, label="target eta_c = 0.5")
    if eta_c is not None:
        ax.axvline(
            eta_c,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=f"estimated eta_c ~ {eta_c:.3f}",
        )
    ax.axhline(0.5, color="tab:gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("noise eta")
    ax.set_ylabel("stationary order parameter psi")
    title = f"Optimized: R={best.R:.3f}, v={best.v:.3f}"
    if eta_c is not None:
        title += f", eta_c≈{eta_c:.3f}"
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "problem3_optimized_psi_vs_eta.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    coarse_config = SearchConfig(
        eta_values=COARSE_ETA,
        n_seeds=1,
        burn_in_steps=100,
        sample_steps=200,
    )
    coarse_best, coarse_results = search_grid(COARSE_R, COARSE_V, coarse_config, phase_name="Coarse grid search")
    print(
        f"\nBest coarse candidate: R={coarse_best.R:.3f}, v={coarse_best.v:.3f}, eta_c={coarse_best.eta_c}",
        flush=True,
    )

    refine_R, refine_V = make_refine_grid(coarse_best.R, coarse_best.v)
    refine_config = SearchConfig(
        eta_values=REFINE_ETA,
        n_seeds=2,
        burn_in_steps=150,
        sample_steps=300,
    )
    refine_best, refine_results = search_grid(refine_R, refine_V, refine_config, phase_name="Refined local grid search")

    best = refine_best if refine_best.objective <= coarse_best.objective else coarse_best
    print(
        f"\nOptimized parameters: R={best.R:.4f}, v={best.v:.4f}, "
        f"estimated eta_c={best.eta_c}, |eta_c - 0.5|={best.objective:.4f}",
        flush=True,
    )

    final_config = SearchConfig(
        eta_values=FINAL_ETA,
        n_seeds=3,
        burn_in_steps=250,
        sample_steps=400,
    )
    final_result = evaluate_candidate(best.R, best.v, final_config)
    final_eta_c = final_result.eta_c
    assert final_result.psi_curve is not None

    plot_final_result(best, FINAL_ETA, final_result.psi_curve, final_eta_c)

    with open(OUTPUT_DIR / "problem3_results.txt", "w", encoding="utf-8") as handle:
        handle.write(f"optimized_R={best.R:.6f}\n")
        handle.write(f"optimized_v={best.v:.6f}\n")
        handle.write(f"estimated_eta_c={final_eta_c:.6f}\n" if final_eta_c is not None else "estimated_eta_c=None\n")
        handle.write(f"objective_abs_error={final_result.objective:.6f}\n")

    save_search_log(OUTPUT_DIR / "problem3_search_log.txt", coarse_results, refine_results)
    print(
        f"Problem 3 complete. Final validation: eta_c={final_eta_c}, "
        f"|eta_c - 0.5|={final_result.objective:.4f}. Outputs written to outputs/"
    )


if __name__ == "__main__":
    main()
