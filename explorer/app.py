"""Streamlit interactive explorer for the Vicsek model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim import cached_compare_dynamics, cached_dynamics, cached_psi_curve
from vicsek import VicsekParams
from viz import (
    make_polarization_compare_figure,
    make_polarization_figure,
    make_psi_curve_figure,
    make_snapshot_figure,
    params_to_tuple,
    polarization_time_axis,
    tuple_to_params,
)

MODES = {
    "Fast": {
        "dynamics_steps": 200,
        "eta_points": 11,
        "n_seeds": 1,
        "burn_in": 200,
        "sample_steps": 400,
    },
    "Accurate": {
        "dynamics_steps": 800,
        "eta_points": 21,
        "n_seeds": 3,
        "burn_in": 400,
        "sample_steps": 800,
    },
}

ETA_COMPARE_SMALL = 0.1
ETA_COMPARE_LARGE = 0.9


def _init_session_state() -> None:
    defaults = {
        "run_id": 0,
        "dynamics_result": None,
        "compare_result": None,
        "phase_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _bump_run_id() -> int:
    st.session_state.run_id += 1
    return st.session_state.run_id


def _trailing_mean(history: np.ndarray, fraction: float = 0.2) -> float:
    n = max(1, int(len(history) * fraction))
    return float(np.mean(history[-n:]))


def _sidebar_params() -> tuple[VicsekParams, dict, float, int, str]:
    st.sidebar.header("Parameters")
    mode = st.sidebar.radio("Mode", list(MODES.keys()), index=0)
    preset = MODES[mode]

    eta = st.sidebar.slider("Noise eta", 0.0, 1.0, 0.5, 0.05)
    R = st.sidebar.slider("Interaction radius R", 0.3, 1.5, 1.0, 0.05)
    v = st.sidebar.slider("Speed v", 0.5, 1.8, 1.0, 0.05)
    N = st.sidebar.slider("Particle count N", 50, 250, 125, 25)
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    if N > 200:
        st.sidebar.warning("N > 200 slows neighbor lookup (O(N^2)).")

    st.sidebar.caption("Fixed: L = 5.0, dt = 0.25 (assignment defaults)")
    params = VicsekParams(N=int(N), L=5.0, R=R, v=v, dt=0.25)
    return params, preset, float(eta), int(seed), mode


def _render_dynamics_tab(params: VicsekParams, preset: dict, eta: float, seed: int) -> None:
    st.subheader("Dynamics")
    col_run, col_compare = st.columns(2)
    with col_run:
        run_clicked = st.button("Run dynamics", type="primary")
    with col_compare:
        compare_clicked = st.button(f"Compare eta={ETA_COMPARE_SMALL} vs {ETA_COMPARE_LARGE}")

    if run_clicked:
        run_id = _bump_run_id()
        params_tuple = params_to_tuple(params)
        positions, angles, history = cached_dynamics(
            params_tuple,
            eta,
            preset["dynamics_steps"],
            seed,
            run_id,
        )
        st.session_state.dynamics_result = {
            "params_tuple": params_tuple,
            "positions": positions,
            "angles": angles,
            "history": history,
            "eta": eta,
            "steps": preset["dynamics_steps"],
        }
        st.session_state.compare_result = None

    if compare_clicked:
        run_id = _bump_run_id()
        params_tuple = params_to_tuple(params)
        time, histories = cached_compare_dynamics(
            params_tuple,
            ETA_COMPARE_SMALL,
            ETA_COMPARE_LARGE,
            preset["dynamics_steps"],
            seed,
            run_id,
        )
        st.session_state.compare_result = {
            "params_tuple": params_tuple,
            "time": time,
            "histories": histories,
        }
        st.session_state.dynamics_result = None

    if st.session_state.compare_result is not None:
        result = st.session_state.compare_result
        fig = make_polarization_compare_figure(
            result["time"],
            result["histories"],
            "Instantaneous polarization (shared initial condition)",
        )
        st.pyplot(fig, clear_figure=True)
        st.caption("Same random initial condition as Problem 1.")
        return

    if st.session_state.dynamics_result is None:
        st.info("Adjust parameters in the sidebar, then click **Run dynamics**.")
        return

    result = st.session_state.dynamics_result
    plot_params = tuple_to_params(result["params_tuple"])
    history = result["history"]
    time = polarization_time_axis(history, plot_params)
    final_time = time[-1]

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("P(t) at final time", f"{history[-1]:.3f}")
    metric_col2.metric("Trailing mean psi (last 20%)", f"{_trailing_mean(history):.3f}")

    plot_col, snap_col = st.columns(2)
    with plot_col:
        fig = make_polarization_figure(
            time,
            history,
            f"Instantaneous polarization (eta={result['eta']})",
        )
        st.pyplot(fig, clear_figure=True)
    with snap_col:
        fig = make_snapshot_figure(
            result["positions"],
            result["angles"],
            plot_params,
            f"Snapshot at t={final_time:.1f}",
        )
        st.pyplot(fig, clear_figure=True)


def _render_phase_tab(params: VicsekParams, preset: dict, seed: int) -> None:
    st.subheader("Phase transition")
    if st.button("Estimate psi(eta) and eta_c"):
        run_id = _bump_run_id()
        params_tuple = params_to_tuple(params)
        eta_values = tuple(np.linspace(0.0, 1.0, preset["eta_points"]).tolist())
        progress = st.progress(0.0, text="Running eta sweep...")
        progress.progress(0.1, text="Running eta sweep (cached after first run)...")
        eta_arr, psi_curve, eta_c = cached_psi_curve(
            params_tuple,
            eta_values,
            preset["n_seeds"],
            preset["burn_in"],
            preset["sample_steps"],
            seed,
            run_id,
        )
        progress.progress(1.0, text="Done.")
        st.session_state.phase_result = {
            "params_tuple": params_tuple,
            "eta_values": eta_arr,
            "psi_curve": psi_curve,
            "eta_c": eta_c,
        }

    if st.session_state.phase_result is None:
        st.info("Click **Estimate psi(eta) and eta_c** to run the sweep.")
        st.markdown(
            "Reference values from batch scripts:\n"
            "- (R=1, v=1) → eta_c ≈ 0.615\n"
            "- (R=0.70, v=0.80) → eta_c ≈ 0.52"
        )
        return

    result = st.session_state.phase_result
    plot_params = tuple_to_params(result["params_tuple"])
    eta_c = result["eta_c"]
    if eta_c is not None:
        st.metric("Estimated eta_c (psi = 0.5)", f"{eta_c:.3f}")
    else:
        st.metric("Estimated eta_c (psi = 0.5)", "N/A")

    fig = make_psi_curve_figure(
        result["eta_values"],
        result["psi_curve"],
        plot_params,
        eta_c,
    )
    st.pyplot(fig, clear_figure=True)
    st.caption("Stochastic estimate; Accurate mode recommended for smoother curves.")


def _render_about_tab() -> None:
    st.subheader("About")
    st.markdown(
        """
The Vicsek model simulates self-propelled particles in a 2D periodic box.
Each particle aligns with its neighbors within radius R, perturbed by noise eta.

**Written report:** see `report.pdf` in the repo root (base assignment deliverable).

**Batch scripts:**
```bash
MPLBACKEND=Agg python problem1_dynamics.py
MPLBACKEND=Agg python problem2_phase.py
MPLBACKEND=Agg python -u problem3_optimize.py
pytest -q
```

Hysteresis plots are generated by `problem2_phase.py` and saved to `outputs/`.
        """
    )


def main() -> None:
    st.set_page_config(page_title="Vicsek Explorer", layout="wide")
    _init_session_state()

    st.title("Vicsek Explorer")
    st.caption("Interactive bonus tool — explore dynamics and phase transitions.")

    params, preset, eta, seed, mode = _sidebar_params()
    st.sidebar.caption(f"Mode: {mode} — {preset['dynamics_steps']} dynamics steps")

    tab_dynamics, tab_phase, tab_about = st.tabs(["Dynamics", "Phase transition", "About"])
    with tab_dynamics:
        _render_dynamics_tab(params, preset, eta, seed)
    with tab_phase:
        _render_phase_tab(params, preset, seed)
    with tab_about:
        _render_about_tab()


if __name__ == "__main__":
    main()
