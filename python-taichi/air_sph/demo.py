"""CLI entry point: builds the particle lattice, wires the solver, source,
boundary, and (optionally) the real-time viewer into one simulation loop.

Usage:
    python -m air_sph.demo                    # interactive viewer
    python -m air_sph.demo --offline --steps 200
"""
import argparse

import numpy as np
import taichi as ti

from air_sph import boundary, sph, source
from air_sph.grid import Grid
from air_sph.sph import Solver
from air_sph.viewer import Viewer


def build_lattice(n_per_axis, dx):
    coords = (np.arange(n_per_axis) - (n_per_axis - 1) / 2) * dx
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)


def probe_pressure(pos_np, pressure_np, probe_pos, radius):
    d = np.linalg.norm(pos_np - np.array(probe_pos), axis=1)
    mask = d < radius
    if not np.any(mask):
        return 0.0
    return float(pressure_np[mask].mean())


@ti.kernel
def update_colors(pressure: ti.template(), colors: ti.template(), n: ti.i32, scale: ti.f32):
    for i in range(n):
        p = pressure[i] / scale
        p = min(max(p, -1.0), 1.0)
        if p >= 0.0:
            colors[i] = ti.Vector([0.9, 0.9 - 0.7 * p, 0.9 - 0.7 * p])
        else:
            colors[i] = ti.Vector([0.9 + 0.7 * p, 0.9 + 0.7 * p, 0.9])


def build_sim(freq, ppw, n_per_axis):
    dx = (sph.C0 / freq) / ppw
    h = 1.3 * dx
    mass = sph.RHO0 * dx ** 3

    pos0 = build_lattice(n_per_axis, dx)
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
    # reference (see Solver.capture_rest_density docstring) -- required before
    # the source starts driving, or this free-boundary lattice's edge/corner
    # particles will produce a spurious tensile pressure force (see Task 8)
    # that destabilizes the whole simulation. Must happen exactly once,
    # before any driving begins.
    grid.clear()
    grid.build(solver.pos, n)
    solver.compute_density()
    solver.capture_rest_density()

    src_radius = 1.5 * dx
    is_source_np = (np.linalg.norm(pos0, axis=1) < src_radius).astype(np.int32)
    is_source = ti.field(dtype=ti.i32, shape=n)
    is_source.from_numpy(is_source_np)

    return solver, grid, is_source, dx, half_extent


def step(solver, grid, is_source, dt, t, freq, amplitude, center, r_start, r_domain, damping_max):
    solver.leapfrog_predict(dt)
    grid.clear()
    grid.build(solver.pos, solver.n)
    solver.compute_density()
    solver.compute_pressure()
    solver.compute_forces()
    solver.leapfrog_correct(dt)
    source.apply_monopole(solver.pos, solver.vel, is_source, solver.n, center, amplitude, freq, t)
    boundary.apply_sponge(solver.pos, solver.vel, solver.n, center, r_start, r_domain, damping_max, dt)


def init_taichi(offline):
    if offline:
        ti.init(arch=ti.cpu)
        return
    try:
        ti.init(arch=ti.gpu)
    except Exception as exc:
        print(f"GPU backend initialization failed ({exc}); falling back to CPU.")
        ti.init(arch=ti.cpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="run headless, no viewer")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--freq", type=float, default=1000.0)
    parser.add_argument("--ppw", type=int, default=10, help="particles per wavelength")
    parser.add_argument("--n", type=int, default=35, help="particles per axis")
    args = parser.parse_args()

    init_taichi(args.offline)

    solver, grid, is_source, dx, half_extent = build_sim(args.freq, args.ppw, args.n)
    center = ti.Vector([0.0, 0.0, 0.0])
    h = solver.h
    dt = 0.3 * h / sph.C0
    amplitude = 0.05
    r_start = 0.7 * half_extent
    r_domain = half_extent
    damping_max = 200.0

    colors = ti.Vector.field(3, dtype=ti.f32, shape=solver.n) if not args.offline else None
    viewer = Viewer() if not args.offline else None

    t = 0.0
    probe_pos = (0.2, 0.0, 0.0)
    for i in range(args.steps):
        step(solver, grid, is_source, dt, t, args.freq, amplitude, center, r_start, r_domain, damping_max)
        t += dt

        if args.offline:
            if i % 20 == 0:
                pos_np = solver.pos.to_numpy()
                pressure_np = solver.pressure.to_numpy()
                p = probe_pressure(pos_np, pressure_np, probe_pos, 1.5 * dx)
                print(f"t={t:.5f}s probe(0.2,0,0)={p:.4f} Pa")
        else:
            if not viewer.running:
                break
            update_colors(solver.pressure, colors, solver.n, 1.0)
            viewer.render(solver.pos, radius=0.3 * dx, colors_field=colors)


if __name__ == "__main__":
    main()
