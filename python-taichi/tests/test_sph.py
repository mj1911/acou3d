import numpy as np
import pytest
import taichi as ti

from air_sph import sph

ti.init(arch=ti.cpu)


@ti.kernel
def _eval_w(r: ti.f32, h: ti.f32) -> ti.f32:
    return sph.kernel_w(r, h)


@ti.kernel
def _eval_dwdr(r: ti.f32, h: ti.f32) -> ti.f32:
    return sph.kernel_dwdr(r, h)


def test_kernel_normalizes_to_one():
    # integral over 3D space of W(r,h) d^3r must equal 1 (kernel normalization)
    h = 0.05
    r = np.linspace(1e-6, 2 * h, 4000)
    w = np.array([_eval_w(float(ri), h) for ri in r])
    integral = np.trapz(4 * np.pi * r ** 2 * w, r)
    assert integral == pytest.approx(1.0, abs=0.01)


def test_kernel_zero_beyond_support():
    h = 0.05
    assert _eval_w(2.5 * h, h) == 0.0


def test_kernel_positive_within_support():
    h = 0.05
    assert _eval_w(0.5 * h, h) > 0
    assert _eval_w(1.5 * h, h) > 0


def test_kernel_derivative_matches_finite_difference():
    h = 0.05
    r = 0.6 * h
    eps = 1e-5
    fd = (_eval_w(r + eps, h) - _eval_w(r - eps, h)) / (2 * eps)
    assert _eval_dwdr(r, h) == pytest.approx(fd, rel=1e-2)


from air_sph.grid import Grid
from air_sph.sph import Solver


def _build_lattice(n_per_axis, dx):
    coords = (np.arange(n_per_axis) - (n_per_axis - 1) / 2) * dx
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)


def _make_lattice_solver(n_per_axis, dx):
    h = 1.3 * dx
    mass = sph.RHO0 * dx ** 3
    pos_np = _build_lattice(n_per_axis, dx)
    n = pos_np.shape[0]
    half_extent = (n_per_axis - 1) * dx / 2
    cell_size = 2.0 * h
    n_cells = int(np.ceil((2 * half_extent + 4 * cell_size) / cell_size))
    grid_min = (-half_extent - 2 * cell_size,) * 3
    grid = Grid(n_cells, cell_size, grid_min, max_particles=n)
    solver = Solver(n, h, mass, grid)
    solver.pos.from_numpy(pos_np)
    return solver, grid, pos_np, half_extent, h


def test_density_matches_rho0_in_bulk():
    solver, grid, pos_np, half_extent, h = _make_lattice_solver(n_per_axis=10, dx=0.02)
    n = pos_np.shape[0]

    grid.clear()
    grid.build(solver.pos, n)
    solver.compute_density()

    rho = solver.rho.to_numpy()
    margin = 2 * h
    interior = np.all(np.abs(pos_np) < (half_extent - margin), axis=1)
    assert interior.sum() > 0
    assert rho[interior] == pytest.approx(sph.RHO0, rel=0.1)


def test_pressure_matches_linear_eos():
    solver, grid, pos_np, half_extent, h = _make_lattice_solver(n_per_axis=4, dx=0.02)
    n = pos_np.shape[0]
    rho_np = sph.RHO0 + np.linspace(-0.05, 0.05, n).astype(np.float32)
    solver.rho.from_numpy(rho_np)

    solver.compute_pressure()

    expected = sph.C0 ** 2 * (rho_np - sph.RHO0)
    # atol=0.02 Pa absorbs a small, fixed float32 rounding-floor difference
    # between Taichi's compiled kernel and numpy's independent reference
    # computation (measured ~0.01 Pa, roughly constant across all particles,
    # not scaling with proximity to the rho=rho0 zero crossing) — consistent
    # with the 0.02 Pa "noise floor" threshold used elsewhere in this plan
    # (see Task 11's monopole-radiation probe-detection threshold).
    np.testing.assert_allclose(solver.pressure.to_numpy(), expected, rtol=1e-5, atol=0.02)


def _two_particle_solver(h, dx, extra_grid_margin=0.5):
    mass = sph.RHO0 * dx ** 3
    pos_np = np.array([[0.0, 0.0, 0.0], [dx, 0.0, 0.0]], dtype=np.float32)
    n = 2
    cell_size = 2 * h
    grid = Grid(n_cells=8, cell_size=cell_size, grid_min=(-extra_grid_margin,) * 3, max_particles=n)
    solver = Solver(n, h, mass, grid)
    solver.pos.from_numpy(pos_np)
    grid.clear()
    grid.build(solver.pos, n)
    return solver, grid


def test_pressure_force_obeys_newtons_third_law():
    h, dx = 0.05, 0.03
    solver, grid = _two_particle_solver(h, dx)

    solver.compute_density()
    rho = solver.rho.to_numpy()
    rho[0] *= 1.2  # perturb so pressure differs between the two particles
    solver.rho.from_numpy(rho)
    solver.compute_pressure()
    solver.compute_forces()

    acc = solver.acc.to_numpy()
    np.testing.assert_allclose(solver.mass * acc[0], -solver.mass * acc[1], rtol=1e-5, atol=1e-8)


def test_viscosity_damps_approaching_particles():
    h, dx = 0.05, 0.03
    solver, grid = _two_particle_solver(h, dx)
    solver.vel.from_numpy(np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32))

    solver.compute_density()
    solver.compute_pressure()
    solver.compute_forces()

    acc = solver.acc.to_numpy()
    # particle 0 (at x=0) moving toward particle 1 (at x=dx) should decelerate;
    # particle 1 moving toward particle 0 should likewise decelerate.
    assert acc[0][0] < 0.0
    assert acc[1][0] > 0.0
