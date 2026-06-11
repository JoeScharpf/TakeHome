"""Vicsek model simulation for the particle interactions assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VicsekParams:
    N: int = 125
    L: float = 5.0
    R: float = 1.0
    v: float = 1.0
    dt: float = 0.25


def initialize_state(params: VicsekParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positions = rng.uniform(0.0, params.L, size=(params.N, 2))
    angles = rng.uniform(-np.pi, np.pi, size=params.N)
    return positions, angles


def neighbor_mask(positions: np.ndarray, L: float, R: float) -> np.ndarray:
    delta = positions[:, None, :] - positions[None, :, :]
    delta -= L * np.round(delta / L)
    dist_sq = np.sum(delta * delta, axis=2)
    return dist_sq < R * R


def polarization(angles: np.ndarray) -> float:
    return float(np.abs(np.mean(np.exp(1j * angles))))


def step(
    positions: np.ndarray,
    angles: np.ndarray,
    params: VicsekParams,
    eta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one time step using quantities at time t."""
    neighbors = neighbor_mask(positions, params.L, params.R)
    phases = np.exp(1j * angles)

    neighbor_counts = neighbors.sum(axis=1)
    neighbor_sums = np.sum(neighbors * phases[None, :], axis=1)
    align = params.v * neighbor_sums / neighbor_counts

    noise = eta * np.exp(1j * rng.uniform(-np.pi, np.pi, size=params.N))
    new_angles = np.angle(align + noise)

    displacement = params.v * params.dt * np.column_stack([np.cos(angles), np.sin(angles)])
    new_positions = (positions + displacement) % params.L

    return new_positions, new_angles


def run_simulation(
    params: VicsekParams,
    eta: float,
    *,
    burn_in_steps: int = 400,
    sample_steps: int = 800,
    sample_every: int = 4,
    seed: int | None = None,
    initial_positions: np.ndarray | None = None,
    initial_angles: np.ndarray | None = None,
) -> tuple[float, list[float]]:
    """Return stationary order parameter psi and sampled P(t) values."""
    rng = np.random.default_rng(seed)

    if initial_positions is None or initial_angles is None:
        positions, angles = initialize_state(params, rng)
    else:
        positions = initial_positions.copy()
        angles = initial_angles.copy()

    for _ in range(burn_in_steps):
        positions, angles = step(positions, angles, params, eta, rng)

    samples: list[float] = []
    for step_idx in range(sample_steps):
        positions, angles = step(positions, angles, params, eta, rng)
        if (step_idx + 1) % sample_every == 0:
            samples.append(polarization(angles))

    psi = float(np.mean(samples)) if samples else 0.0
    return psi, samples


def run_with_polarization_history(
    params: VicsekParams,
    eta: float,
    total_steps: int,
    *,
    seed: int | None = None,
    initial_positions: np.ndarray | None = None,
    initial_angles: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positions, angles, and P(t) sampled after each step."""
    rng = np.random.default_rng(seed)

    if initial_positions is None or initial_angles is None:
        positions, angles = initialize_state(params, rng)
    else:
        positions = initial_positions.copy()
        angles = initial_angles.copy()

    history = np.empty(total_steps)
    for step_idx in range(total_steps):
        positions, angles = step(positions, angles, params, eta, rng)
        history[step_idx] = polarization(angles)

    return positions, angles, history


def estimate_eta_c(
    params: VicsekParams,
    eta_values: np.ndarray,
    *,
    n_seeds: int = 3,
    base_seed: int = 0,
    burn_in_steps: int = 400,
    sample_steps: int = 800,
    sample_every: int = 4,
) -> tuple[np.ndarray, float | None]:
    """Compute psi(eta) averaged over seeds and estimate eta_c where psi=0.5."""
    psi_curve = np.zeros_like(eta_values, dtype=float)

    for idx, eta in enumerate(eta_values):
        seed_estimates = [
            run_simulation(
                params,
                float(eta),
                burn_in_steps=burn_in_steps,
                sample_steps=sample_steps,
                sample_every=sample_every,
                seed=base_seed + 1000 * idx + seed_offset,
            )[0]
            for seed_offset in range(n_seeds)
        ]
        psi_curve[idx] = float(np.mean(seed_estimates))

    eta_c = interpolate_eta_at_psi(eta_values, psi_curve, target_psi=0.5)
    return psi_curve, eta_c


def estimate_single_psi(
    params: VicsekParams,
    eta: float,
    *,
    n_seeds: int = 2,
    base_seed: int = 0,
    burn_in_steps: int = 200,
    sample_steps: int = 400,
    sample_every: int = 4,
) -> float:
    estimates = [
        run_simulation(
            params,
            eta,
            burn_in_steps=burn_in_steps,
            sample_steps=sample_steps,
            sample_every=sample_every,
            seed=base_seed + seed_offset,
        )[0]
        for seed_offset in range(n_seeds)
    ]
    return float(np.mean(estimates))


def find_eta_c_binary(
    params: VicsekParams,
    *,
    eta_low: float = 0.0,
    eta_high: float = 1.0,
    target_psi: float = 0.5,
    n_seeds: int = 2,
    base_seed: int = 0,
    burn_in_steps: int = 200,
    sample_steps: int = 400,
    max_iter: int = 8,
) -> float | None:
    """Binary-search the noise level where psi crosses target_psi."""
    psi_low = estimate_single_psi(
        params,
        eta_low,
        n_seeds=n_seeds,
        base_seed=base_seed,
        burn_in_steps=burn_in_steps,
        sample_steps=sample_steps,
    )
    psi_high = estimate_single_psi(
        params,
        eta_high,
        n_seeds=n_seeds,
        base_seed=base_seed + 17,
        burn_in_steps=burn_in_steps,
        sample_steps=sample_steps,
    )

    if psi_low < target_psi or psi_high > target_psi:
        return None

    low, high = eta_low, eta_high
    for iteration in range(max_iter):
        mid = 0.5 * (low + high)
        psi_mid = estimate_single_psi(
            params,
            mid,
            n_seeds=n_seeds,
            base_seed=base_seed + 31 + iteration,
            burn_in_steps=burn_in_steps,
            sample_steps=sample_steps,
        )
        if psi_mid > target_psi:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def interpolate_eta_at_psi(
    eta_values: np.ndarray,
    psi_values: np.ndarray,
    target_psi: float = 0.5,
) -> float | None:
    """Linear interpolation for the noise level where psi crosses target_psi."""
    for i in range(len(eta_values) - 1):
        psi_left, psi_right = psi_values[i], psi_values[i + 1]
        eta_left, eta_right = eta_values[i], eta_values[i + 1]

        if psi_left == target_psi:
            return float(eta_left)
        if psi_left > target_psi > psi_right or psi_left < target_psi < psi_right:
            weight = (target_psi - psi_left) / (psi_right - psi_left)
            return float(eta_left + weight * (eta_right - eta_left))

    return None
