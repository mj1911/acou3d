"""CLI entry point: builds the particle lattice, wires the solver, source,
boundary, and (optionally) the real-time viewer into one simulation loop.

Usage:
    python -m air_sph.demo                    # interactive viewer
    python -m air_sph.demo --offline --steps 200
"""
import argparse
import ctypes.util

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
        print(f"WARNING: no particles within {radius:.4g} m of probe {probe_pos} -- "
              "probe reads 0.0 Pa because it is sampling empty space, not silence. "
              "Check the probe position against the domain extent.")
        return 0.0
    return float(pressure_np[mask].mean())


def burst_amplitude(t, freq, base_amplitude, warmup_periods=1, active_cycles=1, silent_cycles=10):
    """Gate the source amplitude into a repeating burst pattern.

    Silent for `warmup_periods` periods, then repeats indefinitely: driven at
    `base_amplitude` for `active_cycles` periods, silent for `silent_cycles`
    periods. `t` (simulation time) is never altered or reset -- gating is a
    pure 0/1 multiplier on top of the source's own continuous sin(2*pi*f*t)
    phase, so every on/off transition lands exactly on a zero-crossing of the
    underlying wave rather than needing separate phase bookkeeping.
    """
    period = 1.0 / freq
    warmup = warmup_periods * period
    if t < warmup:
        return 0.0
    cycle_len = active_cycles + silent_cycles
    cycle_index = int((t - warmup) / period)
    position = cycle_index % cycle_len
    return base_amplitude if position < active_cycles else 0.0


def color_scale(pressure_np):
    """Pick a pressure color scale from the current field.

    A fixed scale saturates: measured pressures in the default configuration
    reach ~39 Pa near the source, so a 1.0 Pa scale clips almost the whole
    domain to solid red/blue and shows no gradient at all. The 95th percentile
    of |pressure| tracks the field's actual range while ignoring the extreme
    few particles inside the source cluster, which would otherwise set a scale
    so large the propagating wave is invisible.

    The percentile alone over-amplifies quiescent periods (startup, and the
    silent gaps in the demo's burst-gated source): with no real signal, the
    field is still all float32 rounding noise from SPH density summation
    (~0.01-0.03 Pa typical, up to ~0.08 Pa observed, amplified through the
    C0^2 term in the linear EOS -- same class of artifact as the pressure
    test's rounding-floor finding, see tests/test_sph.py). A percentile-only
    scale treats that noise as if it were the signal and stretches it to the
    full color range, showing as speckling across the whole particle field.
    Flooring the scale at 0.1 Pa -- comfortably above the measured noise
    ceiling, comfortably below real driven/propagating signal (0.1+ Pa once
    a wave is actually moving through, 14+ Pa near an active source) -- keeps
    quiet periods visually quiet without washing out real signal.

    Only called from the interactive rendering path (once per frame), never
    from the physics loop, so the to_numpy copy behind it is not on the hot
    path.
    """
    _NOISE_FLOOR_PA = 0.1
    scale = float(np.percentile(np.abs(pressure_np), 95.0))
    return scale if scale > _NOISE_FLOOR_PA else _NOISE_FLOOR_PA


@ti.kernel
def update_colors(pressure: ti.template(), colors: ti.template(), n: ti.i32, scale: ti.f32):
    for i in range(n):
        p = pressure[i] / scale
        p = min(max(p, -1.0), 1.0)
        if p >= 0.0:
            colors[i] = ti.Vector([0.9, 0.9 - 0.7 * p, 0.9 - 0.7 * p])
        else:
            colors[i] = ti.Vector([0.9 + 0.7 * p, 0.9 + 0.7 * p, 0.9])


# Taichi's GGUI particle renderer has no alpha/transparency channel, so
# "97% transparent" for near-zero-pressure (white) particles is approximated
# by shrinking their radius instead: 3% of base_radius at p=0, scaling up to
# 20% of base_radius as |p| -> 1 (capped well below full size so even
# saturated/wave particles stay small relative to the domain). This reads
# visually like fading out the quiescent background while keeping the
# propagating wavefront prominent, without needing renderer support that
# doesn't exist.
_MIN_RADIUS_FRACTION = 0.03
_MAX_RADIUS_FRACTION = 0.20


@ti.kernel
def update_radii(pressure: ti.template(), radii: ti.template(), n: ti.i32, scale: ti.f32, base_radius: ti.f32):
    for i in range(n):
        p = abs(pressure[i]) / scale
        p = min(p, 1.0)
        radii[i] = base_radius * (_MIN_RADIUS_FRACTION + (_MAX_RADIUS_FRACTION - _MIN_RADIUS_FRACTION) * p)


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

    # One-time neighbor-list capacity check: build() silently drops particles
    # past max_per_cell, which would corrupt every density/force sum without
    # raising anything. Checked once here rather than per step (it needs a
    # to_numpy copy) -- see Grid.check_no_overflow.
    grid.check_no_overflow()

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
    # Known, measured, accepted ordering imprecision: the prescribed source
    # velocity is written here, AFTER leapfrog_correct, but the next step's
    # leapfrog_predict applies its half-kick (vel += 0.5*dt*acc) before the
    # drift -- so the solver's own acceleration perturbs the prescribed velocity
    # by ~5-12% before it moves the particle. The source is therefore driven
    # slightly off its nominal amplitude. Left as-is deliberately: reordering
    # the integration loop is a far riskier change than the error it would
    # remove. Not an unnoticed bug -- see the final whole-branch review in the
    # plan ledger.
    source.apply_monopole(solver.pos, solver.vel, is_source, solver.n, center, amplitude, freq, t)
    boundary.apply_sponge(solver.pos, solver.vel, solver.n, center, r_start, r_domain, damping_max, dt)


def init_taichi(offline):
    if offline:
        ti.init(arch=ti.cpu)
        return
    # ti.gpu tries CUDA first, which logs "libcuda.so lib not found" on every
    # machine without an NVIDIA driver. Skip it there; other GPUs are unaffected.
    gpu_archs = ti.gpu if ctypes.util.find_library("cuda") else [a for a in ti.gpu if a != ti.cuda]
    try:
        ti.init(arch=gpu_archs)
    except Exception as exc:
        print(f"GPU backend initialization failed ({exc}); falling back to CPU.")
        ti.init(arch=ti.cpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="run headless, no viewer")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--freq", type=float, default=1000.0)
    parser.add_argument("--ppw", type=int, default=10, help="particles per wavelength")
    parser.add_argument("--n", type=int, default=40, help="particles per axis")
    args = parser.parse_args()

    init_taichi(args.offline)

    solver, grid, is_source, dx, half_extent = build_sim(args.freq, args.ppw, args.n)
    center = ti.Vector([0.0, 0.0, 0.0])
    h = solver.h
    dt = 0.3 * h / sph.C0
    amplitude = 0.05
    r_start = 0.7 * half_extent
    r_domain = half_extent
    # Sponge strength must scale with the geometry: the shell is crossed in
    # (r_domain - r_start) / c0 seconds, so the damping rate has to be several
    # times c0 / (r_domain - r_start) to absorb a wave within it. k=5 is the
    # middle of the usable [3, 10] band. A fixed literal (this was 200.0) is
    # ~100x too weak here and leaves a reflecting box rather than a free field;
    # it must be computed, since the domain scales with --n/--freq/--ppw.
    # See tests/test_monopole.py::test_sponge_measurably_absorbs_outgoing_waves.
    damping_max = 5.0 * sph.C0 / (r_domain - r_start)

    colors = ti.Vector.field(3, dtype=ti.f32, shape=solver.n) if not args.offline else None
    radii = ti.field(dtype=ti.f32, shape=solver.n) if not args.offline else None
    base_radius = 0.3 * dx
    viewer = Viewer() if not args.offline else None

    t = 0.0
    # Scaled to the domain rather than a fixed 0.2 m: half_extent depends on
    # --n/--freq/--ppw, so a hardcoded radius can land inside the source's
    # exclusion zone or outside the domain entirely depending on the flags.
    probe_pos = (0.5 * half_extent, 0.0, 0.0)
    print(f"domain half_extent={half_extent:.4f} m, dx={dx:.5f} m, n={solver.n} particles, "
          f"dt={dt:.3e} s, damping_max={damping_max:.1f}")
    for i in range(args.steps):
        gated_amplitude = burst_amplitude(t, args.freq, amplitude)
        step(solver, grid, is_source, dt, t, args.freq, gated_amplitude, center, r_start, r_domain, damping_max)
        t += dt

        if args.offline:
            if i % 20 == 0:
                pos_np = solver.pos.to_numpy()
                pressure_np = solver.pressure.to_numpy()
                p = probe_pressure(pos_np, pressure_np, probe_pos, 1.5 * dx)
                print(f"t={t:.5f}s probe({probe_pos[0]:.3f},0,0)={p:.4f} Pa")
        else:
            if not viewer.running:
                break
            scale = color_scale(solver.pressure.to_numpy())
            update_colors(solver.pressure, colors, solver.n, scale)
            update_radii(solver.pressure, radii, solver.n, scale, base_radius)
            viewer.render(solver.pos, radius=base_radius, colors_field=colors, radii_field=radii)


if __name__ == "__main__":
    main()
