"""Unit test for the prescribed-velocity monopole source — see
docs/superpowers/specs/2026-09-24-air-sph-acoustics-design.md, "Driving source".
"""
import numpy as np
import pytest
import taichi as ti

from air_sph import source


def test_monopole_sets_prescribed_velocity_on_source_particles_only():
    n = 2
    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=n)
    is_source = ti.field(dtype=ti.i32, shape=n)

    pos.from_numpy(np.array([[0.01, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32))
    vel.fill(0.0)
    is_source.from_numpy(np.array([1, 0], dtype=np.int32))

    center = ti.Vector([0.0, 0.0, 0.0])
    amplitude, freq, t = 2.0, 100.0, 0.0025  # quarter period at 100 Hz

    source.apply_monopole(pos, vel, is_source, n, center, amplitude, freq, t)

    v = vel.to_numpy()
    expected = amplitude * np.sin(2 * np.pi * freq * t)
    assert v[0][0] == pytest.approx(expected, rel=1e-4)
    assert v[0][1] == pytest.approx(0.0, abs=1e-6)
    assert v[1][0] == pytest.approx(0.0)  # non-source particle untouched


def test_monopole_off_axis_off_center_direction():
    """Test monopole with non-zero center and genuinely non-collinear position.

    This catches two classes of bugs:
    1. Missing center subtraction (using pos instead of pos - center)
    2. Component swapping (y/z swap would be invisible if both are zero)

    The values are chosen to be genuinely non-collinear: normalizing pos alone
    (the buggy path) gives a different unit vector than normalizing (pos - center)
    (the correct path), so a missed-center-subtraction bug will fail this test.
    """
    n = 2
    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=n)
    is_source = ti.field(dtype=ti.i32, shape=n)

    # Source at (1.03, 1.98, 3.05), center at (1.0, 2.0, 3.0)
    # rvec = (0.03, -0.02, 0.05) — all distinct, all nonzero, NOT collinear with center
    pos.from_numpy(np.array([[1.03, 1.98, 3.05], [10.0, 20.0, 30.0]], dtype=np.float32))
    vel.fill(0.0)
    is_source.from_numpy(np.array([1, 0], dtype=np.int32))

    center = ti.Vector([1.0, 2.0, 3.0])
    amplitude, freq, t = 2.0, 100.0, 0.0025

    source.apply_monopole(pos, vel, is_source, n, center, amplitude, freq, t)

    # Manually compute expected: rvec = (0.03, -0.02, 0.05)
    rvec = np.array([0.03, -0.02, 0.05], dtype=np.float32)
    r = np.linalg.norm(rvec)
    direction = rvec / r
    expected_vel = amplitude * np.sin(2 * np.pi * freq * t) * direction

    v = vel.to_numpy()
    # Check all three components independently
    assert v[0][0] == pytest.approx(expected_vel[0], rel=1e-4)
    assert v[0][1] == pytest.approx(expected_vel[1], rel=1e-4)
    assert v[0][2] == pytest.approx(expected_vel[2], rel=1e-4)
    # Non-source particle untouched
    assert v[1][0] == pytest.approx(0.0)


def test_monopole_singularity_guard_at_center():
    """Test monopole singularity guard: source particle exactly at center (r=0).

    When a source particle is at the center (r < 1e-9), the direction vector should
    be set to zero, resulting in zero velocity regardless of amplitude and frequency.
    This covers the fallback branch when division-by-zero would occur.
    """
    n = 1
    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=n)
    is_source = ti.field(dtype=ti.i32, shape=n)

    # Source particle exactly at center
    center_val = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    pos.from_numpy(np.array([center_val], dtype=np.float32))
    vel.fill(0.0)
    is_source.from_numpy(np.array([1], dtype=np.int32))

    center = ti.Vector(center_val.tolist())
    amplitude, freq, t = 2.0, 100.0, 0.0025

    source.apply_monopole(pos, vel, is_source, n, center, amplitude, freq, t)

    v = vel.to_numpy()
    # All three components should be exactly zero (not NaN, not garbage)
    assert v[0][0] == pytest.approx(0.0, abs=1e-9)
    assert v[0][1] == pytest.approx(0.0, abs=1e-9)
    assert v[0][2] == pytest.approx(0.0, abs=1e-9)


def test_smooth_monopole_adds_gaussian_radial_kick():
    """apply_smooth_monopole adds dt * accel * (r_vec / sigma) * exp(-r^2 / 2 sigma^2).

    Radial, zero at the centre, smooth in space, and added to (not written
    over) the existing velocity -- so it drives the air without pinning any
    particles or imposing lattice-scale velocity jumps.
    """
    pts = np.array([[0.0, 0.0, 0.0],        # centre: no push
                    [0.02, 0.0, 0.0],       # r = sigma along +x
                    [0.0, -0.03, 0.04],     # off-axis, r = 2.5 sigma
                    [1.0, 0.0, 0.0]],       # far away: negligible
                   dtype=np.float32)
    n = len(pts)
    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=n)
    pos.from_numpy(pts)
    vel.from_numpy(np.full((n, 3), 0.5, dtype=np.float32))
    center = ti.Vector([0.0, 0.0, 0.0])
    accel, sigma, dt = 100.0, 0.02, 1e-3

    source.apply_smooth_monopole(pos, vel, n, center, accel, sigma, dt)

    r = np.linalg.norm(pts, axis=1)
    expected = 0.5 + dt * accel * pts / sigma * np.exp(-r**2 / (2 * sigma**2))[:, None]
    np.testing.assert_allclose(vel.to_numpy(), expected, rtol=1e-4, atol=1e-7)
