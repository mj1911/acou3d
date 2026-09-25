"""CLI entry point: builds the particle lattice, wires the solver, source,
boundary, and (optionally) the real-time viewer into one simulation loop.

Usage:
    python -m air_sph.demo                    # interactive viewer
    python -m air_sph.demo --slice            # 2D cross-section through the source
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

DEFAULT_FREQ = 500.0       # Hz
DEFAULT_PPW = 20           # particles per wavelength at DEFAULT_FREQ
DEFAULT_N = 40             # particles per axis
DEFAULT_AMPLITUDE = 0.05   # source surface velocity, m/s
WARMUP_PERIODS = 1
ACTIVE_CYCLES = 3
# Source force width, in smoothing lengths (1.5 h ~ 2 particle spacings): wide
# enough to have no lattice-scale content, narrow compared with a wavelength.
SOURCE_SIGMA_H = 1.5


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


def burst_end_time(freq, warmup_periods=WARMUP_PERIODS, active_cycles=ACTIVE_CYCLES):
    """Time at which the first burst's drive stops."""
    return (warmup_periods + active_cycles) / freq


def burst_amplitude(t, freq, base_amplitude, warmup_periods=WARMUP_PERIODS,
                    active_cycles=ACTIVE_CYCLES, silent_cycles=10):
    """Envelope the source amplitude into a repeating, smoothly windowed burst.

    Silent for `warmup_periods` periods, then repeats indefinitely: a
    Hann-windowed burst of `active_cycles` periods, then `silent_cycles`
    periods of silence. `t` is never altered or reset -- the envelope
    multiplies the source's own continuous sin(2*pi*f*t).

    Why a Hann window rather than a hard on/off gate: the SPH medium has a
    frequency ceiling (~c0 / (4.8 dx), where waves stop carrying energy; see
    tests/test_propagation.py). A hard gate sprays energy across a wide band,
    and whatever lands near that ceiling stays trapped around the source and
    rings. The window keeps the burst's spectrum near `freq`.
    """
    period = 1.0 / freq
    warmup = warmup_periods * period
    if t < warmup:
        return 0.0
    cycle_len = (active_cycles + silent_cycles) * period
    tau = (t - warmup) % cycle_len
    active = active_cycles * period
    if tau >= active:
        return 0.0
    return base_amplitude * np.sin(np.pi * tau / active) ** 2


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


@ti.kernel
def slice_radii(in_slice: ti.template(), radii: ti.template(), n: ti.i32, radius: ti.f32):
    """Show only slice particles, at a fixed size; everything else gets radius 0."""
    for i in range(n):
        radii[i] = radius if in_slice[i] == 1 else 0.0


def slice_mask(pos_np, dx):
    """Particles in the single lattice layer nearest z = 0 (the source plane).

    The lattice is centred on the origin, so with an even particle count per
    axis there is no layer at exactly z = 0; the layer at +dx/2 is used. The
    selection is made once: acoustic displacements (~2e-4 dx) never move a
    particle out of its layer.
    """
    z = pos_np[:, 2]
    z_layer = z[np.argmin(np.abs(z - 1e-6 * dx))]
    return np.abs(z - z_layer) < 0.25 * dx


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

    return solver, grid, dx, half_extent


def source_accel(t, freq, amplitude):
    """Peak acceleration of the smooth source force at time t (see
    source.apply_smooth_monopole): amplitude * w * cos(w t), so the air near the
    source moves with velocity of order amplitude * sin(w t). Shared with the
    FDTD reference (air_sph.validation) so both are driven identically."""
    w = 2.0 * np.pi * freq
    return amplitude * w * np.cos(w * t)


def step(solver, grid, dt, t, freq, amplitude, center, r_start, r_domain, damping_max):
    solver.leapfrog_predict(dt)
    grid.clear()
    grid.build(solver.pos, solver.n)
    solver.compute_density()
    solver.compute_pressure()
    solver.compute_forces()
    solver.leapfrog_correct(dt)
    source.apply_smooth_monopole(solver.pos, solver.vel, solver.n, center,
                                 source_accel(t, freq, amplitude), SOURCE_SIGMA_H * solver.h, dt)
    # Impedance-matched absorbing shell: damp velocity and pressure equally.
    boundary.apply_sponge(solver.pos, solver.vel, solver.n, center, r_start, r_domain, damping_max, dt)
    boundary.relax_pressure(solver.pos, solver.rho, solver.rho0, solver.n, center, r_start, r_domain,
                            damping_max, dt)


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
    parser.add_argument("--freq", type=float, default=DEFAULT_FREQ)
    parser.add_argument("--ppw", type=int, default=DEFAULT_PPW, help="particles per wavelength")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="particles per axis")
    parser.add_argument("--slice", action="store_true",
                        help="show only the particle layer through the source, viewed face-on")
    args = parser.parse_args()

    init_taichi(args.offline)

    solver, grid, dx, half_extent = build_sim(args.freq, args.ppw, args.n)
    center = ti.Vector([0.0, 0.0, 0.0])
    h = solver.h
    dt = 0.3 * h / sph.C0
    amplitude = DEFAULT_AMPLITUDE
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
    in_slice = None
    if args.slice and not args.offline:
        mask_np = slice_mask(solver.pos.to_numpy(), dx)
        in_slice = ti.field(dtype=ti.i32, shape=solver.n)
        in_slice.from_numpy(mask_np.astype(np.int32))
        # Face-on along -z; far enough back that the whole slice fits the default 45 deg FOV.
        viewer = Viewer(camera_pos=(0.0, 0.0, 2.6 * half_extent))
    else:
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
        step(solver, grid, dt, t, args.freq, gated_amplitude, center, r_start, r_domain, damping_max)
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
            pressure_np = solver.pressure.to_numpy()
            if in_slice is not None:
                scale = color_scale(pressure_np[mask_np])
                update_colors(solver.pressure, colors, solver.n, scale)
                slice_radii(in_slice, radii, solver.n, 0.5 * dx)
            else:
                scale = color_scale(pressure_np)
                update_colors(solver.pressure, colors, solver.n, scale)
                update_radii(solver.pressure, radii, solver.n, scale, base_radius)
            viewer.render(solver.pos, radius=base_radius, colors_field=colors, radii_field=radii)


if __name__ == "__main__":
    main()
