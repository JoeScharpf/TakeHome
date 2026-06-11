"""Cached simulation wrappers for the Streamlit Vicsek explorer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from viz import tuple_to_params
from vicsek import estimate_eta_c, initialize_state, run_with_polarization_history


@st.cache_data(show_spinner="Running dynamics...")
def cached_dynamics(
    params_tuple: tuple[int, float, float, float, float],
    eta: float,
    total_steps: int,
    seed: int,
    run_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = tuple_to_params(params_tuple)
    positions, angles, history = run_with_polarization_history(
        params,
        eta,
        total_steps,
        seed=seed,
    )
    return positions, angles, history


@st.cache_data(show_spinner="Comparing eta=0.1 vs eta=0.9...")
def cached_compare_dynamics(
    params_tuple: tuple[int, float, float, float, float],
    eta_small: float,
    eta_large: float,
    total_steps: int,
    seed: int,
    run_id: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run two dynamics trajectories from the same initial condition."""
    params = tuple_to_params(params_tuple)
    rng = np.random.default_rng(seed)
    initial_positions, initial_angles = initialize_state(params, rng)

    histories: dict[str, np.ndarray] = {}
    for eta, label in ((eta_small, f"eta={eta_small}"), (eta_large, f"eta={eta_large}")):
        _, _, history = run_with_polarization_history(
            params,
            eta,
            total_steps,
            seed=seed,
            initial_positions=initial_positions,
            initial_angles=initial_angles,
        )
        histories[label] = history

    time = np.arange(1, total_steps + 1) * params.dt
    return time, histories


@st.cache_data(show_spinner="Estimating psi(eta)...")
def cached_psi_curve(
    params_tuple: tuple[int, float, float, float, float],
    eta_values_tuple: tuple[float, ...],
    n_seeds: int,
    burn_in_steps: int,
    sample_steps: int,
    base_seed: int,
    run_id: int,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    params = tuple_to_params(params_tuple)
    eta_values = np.array(eta_values_tuple)
    psi_curve, eta_c = estimate_eta_c(
        params,
        eta_values,
        n_seeds=n_seeds,
        base_seed=base_seed,
        burn_in_steps=burn_in_steps,
        sample_steps=sample_steps,
    )
    return eta_values, psi_curve, eta_c
