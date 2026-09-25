"""Unit test for the sponge-layer open boundary — see
docs/superpowers/specs/2026-09-24-air-sph-acoustics-design.md, "Open/absorbing boundary".
"""
import numpy as np
import pytest
import taichi as ti

from air_sph import boundary

ti.init(arch=ti.cpu)


def test_sponge_damps_outer_particle_more_than_inner():
    n = 2
    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=n)
    pos.from_numpy(np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32))
    vel.from_numpy(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))

    center = ti.Vector([0.0, 0.0, 0.0])
    r_start, r_domain, damping_max, dt = 0.5, 1.0, 50.0, 0.001

    for _ in range(20):
        boundary.apply_sponge(pos, vel, n, center, r_start, r_domain, damping_max, dt)

    v = vel.to_numpy()
    assert v[0][0] == pytest.approx(1.0)  # inner particle (r=0 < r_start): untouched
    assert v[1][0] < 1.0                  # outer particle (r=0.9 > r_start): damped


def test_sponge_ramp_shape_quadratic():
    """Verify the damping ramp is quadratic in s, not linear or inverted.
    Tests particles at s=0.25, 0.5, 0.75 and verifies:
    1. Monotonic damping (increasing r → smaller velocity)
    2. Matches closed-form exp(-damping_max * s**2 * dt)
    """
    n = 4
    r_start, r_domain, damping_max, dt = 0.5, 1.0, 50.0, 0.001

    # Position particles at s = 0.25, 0.5, 0.75, 1.0
    # where s = (r - r_start) / (r_domain - r_start)
    radii = np.array([
        r_start + 0.25 * (r_domain - r_start),  # s=0.25, r=0.625
        r_start + 0.5 * (r_domain - r_start),   # s=0.5, r=0.75
        r_start + 0.75 * (r_domain - r_start),  # s=0.75, r=0.875
        r_domain,                                 # s=1.0, r=1.0
    ], dtype=np.float32)

    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=n)
    pos_np = np.array([[r, 0.0, 0.0] for r in radii], dtype=np.float32)
    vel_np = np.ones((n, 3), dtype=np.float32)
    pos.from_numpy(pos_np)
    vel.from_numpy(vel_np)

    center = ti.Vector([0.0, 0.0, 0.0])

    # Single step to verify instantaneous ramp shape
    boundary.apply_sponge(pos, vel, n, center, r_start, r_domain, damping_max, dt)

    v = vel.to_numpy()

    # Expected damping: vel *= exp(-damping_max * s**2 * dt)
    s_values = np.array([0.25, 0.5, 0.75, 1.0])
    expected_vel = np.exp(-damping_max * s_values**2 * dt)

    # Verify monotonic decrease (higher s → lower velocity)
    for i in range(n - 1):
        assert v[i][0] > v[i + 1][0], \
            f"Monotonicity violated: s={s_values[i]} vel={v[i][0]:.6f} " \
            f">= s={s_values[i+1]} vel={v[i+1][0]:.6f}"

    # Verify against closed-form formula (tight tolerance for single step)
    for i, (actual, expected) in enumerate(zip(v[:, 0], expected_vel)):
        assert actual == pytest.approx(expected, rel=1e-5), \
            f"Ramp shape mismatch at s={s_values[i]}: " \
            f"expected {expected:.8f}, got {actual:.8f}"
