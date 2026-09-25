"""System-level validation: a quiescent particle lattice with no source
driving must stay near mechanical equilibrium (the standard SPH "still
box" sanity check) — see docs/superpowers/specs/2026-09-24-air-sph-acoustics-design.md.
"""
import numpy as np
import pytest
import taichi as ti

from air_sph import sph
from air_sph.grid import Grid
from air_sph.sph import Solver


def _build_lattice(n_per_axis, dx):
    coords = (np.arange(n_per_axis) - (n_per_axis - 1) / 2) * dx
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)


def test_still_box_remains_stable():
    n_per_axis, dx = 10, 0.02
    h = 1.3 * dx
    mass = sph.RHO0 * dx ** 3

    pos0 = _build_lattice(n_per_axis, dx)
    n = pos0.shape[0]
    half_extent = (n_per_axis - 1) * dx / 2
    cell_size = 2 * h
    n_cells = int(np.ceil((2 * half_extent + 4 * cell_size) / cell_size))
    grid_min = (-half_extent - 2 * cell_size,) * 3

    grid = Grid(n_cells, cell_size, grid_min, max_particles=n)
    solver = Solver(n, h, mass, grid)
    solver.pos.from_numpy(pos0)
    solver.vel.fill(0.0)
    solver.acc.fill(0.0)

    # Capture each particle's own t=0 SPH density as its personal rest-density
    # reference (see capture_rest_density docstring) -- required for a
    # free-boundary quiescent lattice to actually stay quiescent.
    grid.clear()
    grid.build(solver.pos, n)
    solver.compute_density()
    solver.capture_rest_density()

    # One-time neighbor-list capacity check: build() silently drops particles
    # past max_per_cell, which would corrupt every density/force sum without
    # raising anything. Checked once here rather than per step (it needs a
    # to_numpy copy) -- see Grid.check_no_overflow.
    grid.check_no_overflow()

    dt = 0.3 * h / sph.C0
    n_steps = 300
    for _ in range(n_steps):
        solver.leapfrog_predict(dt)
        grid.clear()
        grid.build(solver.pos, n)
        solver.compute_density()
        solver.compute_pressure()
        solver.compute_forces()
        solver.leapfrog_correct(dt)

    pos_final = solver.pos.to_numpy()
    displacement = np.linalg.norm(pos_final - pos0, axis=1)
    assert displacement.max() < 0.5 * dx

    rho = solver.rho.to_numpy()
    margin = 2 * h
    interior = np.all(np.abs(pos0) < (half_extent - margin), axis=1)
    assert interior.sum() > 0
    assert rho[interior] == pytest.approx(sph.RHO0, rel=0.1)
