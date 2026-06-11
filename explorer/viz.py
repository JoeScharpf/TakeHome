"""Matplotlib figure builders for the Streamlit Vicsek explorer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vicsek import VicsekParams


def params_to_tuple(params: VicsekParams) -> tuple[int, float, float, float, float]:
    return (params.N, params.L, params.R, params.v, params.dt)


def tuple_to_params(t: tuple[int, float, float, float, float]) -> VicsekParams:
    return VicsekParams(N=t[0], L=t[1], R=t[2], v=t[3], dt=t[4])


def polarization_time_axis(history: np.ndarray, params: VicsekParams) -> np.ndarray:
    return np.arange(1, history.shape[0] + 1) * params.dt


def make_snapshot_figure(
    positions: np.ndarray,
    angles: np.ndarray,
    params: VicsekParams,
    title: str,
) -> plt.Figure:
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
    return fig


def make_polarization_figure(
    time: np.ndarray,
    history: np.ndarray,
    title: str,
    *,
    label: str = "P(t)",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(time, history, label=label, linewidth=1.5)
    ax.set_xlabel("time")
    ax.set_ylabel("P(t)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_polarization_compare_figure(
    time: np.ndarray,
    histories: dict[str, np.ndarray],
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, history in histories.items():
        ax.plot(time, history, label=label, linewidth=1.5)
    ax.set_xlabel("time")
    ax.set_ylabel("P(t)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_psi_curve_figure(
    eta_values: np.ndarray,
    psi_curve: np.ndarray,
    params: VicsekParams,
    eta_c: float | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eta_values, psi_curve, "o-", linewidth=1.5, markersize=4, label="psi(eta)")
    if eta_c is not None:
        ax.axvline(eta_c, color="tab:red", linestyle="--", linewidth=1.2, label=f"eta_c ~ {eta_c:.3f}")
    ax.axhline(0.5, color="tab:gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("noise eta")
    ax.set_ylabel("stationary order parameter psi")
    ax.set_title(f"Order parameter vs noise (N={params.N}, R={params.R}, v={params.v})")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
