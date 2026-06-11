"""Unit tests for core Vicsek simulation invariants."""

import numpy as np

from vicsek import (
    VicsekParams,
    interpolate_eta_at_psi,
    neighbor_mask,
    polarization,
    step,
)


def test_polarization_perfect_alignment():
    angles = np.zeros(125)
    assert np.isclose(polarization(angles), 1.0)


def test_polarization_bounds():
    rng = np.random.default_rng(0)
    angles = rng.uniform(-np.pi, np.pi, size=125)
    p = polarization(angles)
    assert 0.0 <= p <= 1.0


def test_random_headings_low_polarization():
    rng = np.random.default_rng(0)
    angles = rng.uniform(-np.pi, np.pi, size=5000)
    assert polarization(angles) < 0.05


def test_periodic_wrapping():
    """Validate the same modulo convention used in step()."""
    L = 5.0
    positions = np.array(
        [
            [5.2, 2.0],
            [-0.3, 4.9],
            [2.1, 5.7],
        ]
    )
    wrapped = positions % L
    np.testing.assert_allclose(
        wrapped,
        np.array(
            [
                [0.2, 2.0],
                [4.7, 4.9],
                [2.1, 0.7],
            ]
        ),
    )


def test_neighbor_mask_across_boundary():
    positions = np.array([[0.1, 2.0], [4.9, 2.0]])
    mask = neighbor_mask(positions, L=5.0, R=0.5)
    assert mask[0, 1]
    assert mask[1, 0]


def test_position_update_uses_current_heading():
    params = VicsekParams(N=2, L=5.0, R=2.0, v=1.0, dt=0.25)
    positions = np.array(
        [
            [1.0, 1.0],
            [1.1, 1.0],
        ]
    )
    angles = np.array([0.0, np.pi / 2])
    rng = np.random.default_rng(0)

    new_positions, _ = step(positions, angles, params, eta=0.0, rng=rng)

    # Motion must use old headings: right for particle 0, up for particle 1.
    np.testing.assert_allclose(new_positions[0], [1.25, 1.0], atol=1e-6)
    np.testing.assert_allclose(new_positions[1], [1.1, 1.25], atol=1e-6)


def test_positions_remain_inside_box_after_step():
    params = VicsekParams(N=10, L=5.0, R=1.0, v=1.0, dt=0.25)
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, params.L, size=(params.N, 2))
    angles = rng.uniform(-np.pi, np.pi, size=params.N)

    new_positions, new_angles = step(positions, angles, params, eta=0.5, rng=rng)

    assert np.all(new_positions >= 0.0)
    assert np.all(new_positions < params.L)
    assert new_positions.shape == (params.N, 2)
    assert new_angles.shape == (params.N,)


def test_low_noise_preserves_alignment():
    params = VicsekParams(N=5, L=5.0, R=1.0, v=1.0, dt=0.25)
    positions = np.array(
        [
            [1.0, 1.0],
            [1.1, 1.0],
            [0.9, 1.0],
            [1.0, 1.1],
            [1.0, 0.9],
        ]
    )
    angles = np.zeros(params.N)
    rng = np.random.default_rng(0)

    _, new_angles = step(positions, angles, params, eta=0.0, rng=rng)

    assert polarization(new_angles) > 0.99


def test_interpolate_eta_at_psi_crossing():
    eta = np.array([0.4, 0.6])
    psi = np.array([0.8, 0.2])
    assert np.isclose(interpolate_eta_at_psi(eta, psi, target_psi=0.5), 0.5)


def test_interpolate_eta_at_psi_no_crossing():
    eta = np.array([0.2, 0.4, 0.6])
    psi = np.array([0.8, 0.7, 0.6])
    assert interpolate_eta_at_psi(eta, psi, target_psi=0.5) is None
