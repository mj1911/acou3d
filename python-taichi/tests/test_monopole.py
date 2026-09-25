"""System-level validation of wave propagation from a pulsing monopole source
— see docs/superpowers/specs/2026-09-24-air-sph-acoustics-design.md,
"Validation & testing".

SCOPE — what this test does and does not validate (read before tightening
anything here):

It validates that the SPH scheme produces *causally propagating acoustic
waves*:
  1. the disturbance reaches each probe radius at approximately r / c0
     (correct wave speed, causal ordering),
  2. the field at the probe oscillates at the driven source frequency,
  3. the amplitude decreases monotonically with radius and decays no slower
     than pure geometric (1/r) spreading.

It deliberately does NOT validate precise 1/r amplitude scaling. At this
test's resolution (ppw=6, see the comment on `ppw` below), numerical
attenuation of the SPH discretization dominates the amplitude decay -- it is
roughly three orders of magnitude stronger than real air's physical
attenuation -- so the measured falloff is a mixture of geometric spreading and
scheme dissipation, not clean 1/r. Resolving 1/r quantitatively needs the
10-15 particles per wavelength the design spec names as the accuracy floor,
plus decade-span probe radii and a much longer run: a far larger and slower
simulation than this fast unit test. That is documented as deferred future
work in the spec's "Validation & testing" section, not silently skipped.

What remains is a genuinely diagnostic set of checks, not a vacuous one: a
wrong sound speed, a wrong EOS, a sign error, or a wrong driven frequency all
fail these assertions hard (established by this plan's mutation testing).

Probe geometry and measurement method (read before changing anything here):
the source cluster's own kernel support (2h) plus its physical radius
(src_radius) carve out an exclusion zone around the origin -- a probe
placed inside it measures direct one-hop SPH neighbor coupling, not
causal wave propagation, and "arrives" non-causally early. Probe radii
here are chosen to clear that zone with positive margin (verified
numerically). Pressure is sampled by averaging over 6 symmetric
directions (+-x, +-y, +-z) at each radius rather than a single direction
-- this was found empirically to be the dominant fix for run-to-run
measurement noise (a single-direction probe is sensitive to local
particle-placement asymmetry in a way that washes out once averaged over
the sphere), far more so than fine-tuning the exact radii.

Note: averaging over 6 symmetric directions means this test cannot
distinguish a genuinely isotropic monopole source from an anisotropic
one that happens to produce similar direction-averaged readings (e.g.
a source that only drives particles in one hemisphere). Source
direction-vector correctness at the implementation level is covered by
air_sph's source-driver unit tests instead; this test validates wave
propagation physics given a (trusted) source, not source symmetry
itself.
"""
import numpy as np
import pytest
import taichi as ti

from air_sph import sph, boundary, source
from air_sph.grid import Grid
from air_sph.sph import Solver

# Sponge strength coefficient k in damping_max = k * c0 / (r_domain - r_start).
# The sponge must remove an outgoing wave within the shell's thickness; the
# shell is crossed in time (r_domain - r_start) / c0, so the damping rate has
# to be several times c0 / (r_domain - r_start) to matter at all. k in [3, 10]
# is the usable band; k=5 is the middle of it. A fixed absolute damping_max
# (the original code used 200.0) is ~100x too weak at this geometry -- it
# absorbed only ~3% of amplitude per pass, leaving a reflecting box.
# test_sponge_measurably_absorbs_outgoing_waves guards against regressing to
# an inert sponge.
SPONGE_K = 5.0

# Six symmetric sample directions per probe radius, averaged together to
# extract the spherically-symmetric monopole signal cleanly -- see module
# docstring.
_PROBE_DIRECTIONS = np.array([
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
])

# Chosen so that r - src_radius - probe_sample_radius - 2h > 0 for every
# probe (positive kernel-support clearance from the source), while
# staying inside r_start (before sponge damping begins).
_PROBE_RADII = [0.30, 0.335, 0.37]


def _build_lattice(n_per_axis, dx):
    coords = (np.arange(n_per_axis) - (n_per_axis - 1) / 2) * dx
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)


def _probe_pressure(pos_np, pressure_np, r, radius):
    """Average pressure over particles within `radius` of each of the 6
    sample points at distance r from the origin, then average those 6
    readings together."""
    vals = []
    for d in _PROBE_DIRECTIONS:
        dist = np.linalg.norm(pos_np - r * d, axis=1)
        mask = dist < radius
        if np.any(mask):
            vals.append(pressure_np[mask].mean())
    return float(np.mean(vals)) if vals else 0.0


def _make_sim(freq=1000.0, ppw=6, n_per_axis=20):
    """Build the monopole simulation and return it plus the derived geometry.

    Shared by both tests in this module so they exercise the identical
    configuration -- the sponge-effectiveness test must measure the same setup
    the radiation test validates, or it would not guard it.
    """
    dx = (sph.C0 / freq) / ppw
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
    # reference (see Solver.capture_rest_density docstring) -- required before
    # the source starts driving, or this free-boundary lattice's edge/corner
    # particles will produce a spurious tensile pressure force (see Task 8)
    # that swamps the real acoustic signal this test is trying to measure.
    # Must happen exactly once, before any driving begins.
    grid.clear()
    grid.build(solver.pos, n)
    solver.compute_density()
    solver.capture_rest_density()

    # One-time neighbor-list capacity check: build() silently drops particles
    # past max_per_cell, which would corrupt every density/force sum without
    # raising anything. Checked here (not per step -- it needs a to_numpy
    # copy); the lattice is at its densest now, so headroom here holds for the
    # tiny acoustic displacements that follow.
    grid.check_no_overflow()

    # src_radius and probe_sample_radius are 1.0*dx (not the more generous
    # 1.5*dx used elsewhere) specifically to open up enough room between the
    # source's kernel-support exclusion zone and the sponge boundary for
    # probes with positive clearance -- see module docstring.
    src_radius = 1.0 * dx
    is_source_np = (np.linalg.norm(pos0, axis=1) < src_radius).astype(np.int32)
    assert is_source_np.sum() > 0
    is_source = ti.field(dtype=ti.i32, shape=n)
    is_source.from_numpy(is_source_np)

    r_start = 0.7 * half_extent
    r_domain = half_extent

    return dict(
        solver=solver, grid=grid, is_source=is_source, n=n,
        dx=dx, h=h, half_extent=half_extent, freq=freq,
        r_start=r_start, r_domain=r_domain,
        damping_max=SPONGE_K * sph.C0 / (r_domain - r_start),
        dt=0.3 * h / sph.C0,
        amplitude=0.05,  # m/s, small perturbation velocity for the source cluster
        probe_sample_radius=1.0 * dx,
    )


def _run(sim, n_steps, damping_max):
    """Advance the simulation, recording probe pressure histories and the
    pressure field's energy inside the sponge shell."""
    solver, grid = sim["solver"], sim["grid"]
    n, dt, freq = sim["n"], sim["dt"], sim["freq"]
    center = ti.Vector([0.0, 0.0, 0.0])

    histories = {r: [] for r in _PROBE_RADII}
    times = []
    shell_energy = []
    t = 0.0
    for _ in range(n_steps):
        solver.leapfrog_predict(dt)
        grid.clear()
        grid.build(solver.pos, n)
        solver.compute_density()
        solver.compute_pressure()
        solver.compute_forces()
        solver.leapfrog_correct(dt)
        t += dt
        # Known, measured, accepted ordering imprecision: the prescribed source
        # velocity is written here, AFTER leapfrog_correct, but the next step's
        # leapfrog_predict applies its half-kick (vel += 0.5*dt*acc) before the
        # drift -- so the solver's own acceleration perturbs the prescribed
        # velocity by ~5-12% before it moves the particle. The source is
        # therefore driven slightly off its nominal amplitude. Left as-is
        # deliberately: reordering the integration loop is a far riskier change
        # than the error it would remove, and the error is well inside this
        # test's tolerances. Not an unnoticed bug -- see the final whole-branch
        # review in the plan ledger.
        source.apply_monopole(solver.pos, solver.vel, sim["is_source"], n, center,
                             sim["amplitude"], freq, t)
        boundary.apply_sponge(solver.pos, solver.vel, n, center,
                             sim["r_start"], sim["r_domain"], damping_max, dt)

        pos_np = solver.pos.to_numpy()
        pressure_np = solver.pressure.to_numpy()
        times.append(t)
        for r in _PROBE_RADII:
            histories[r].append(_probe_pressure(pos_np, pressure_np, r, sim["probe_sample_radius"]))
        in_shell = np.linalg.norm(pos_np, axis=1) > sim["r_start"]
        shell_energy.append(float(np.sum(pressure_np[in_shell] ** 2)) if np.any(in_shell) else 0.0)

    return (np.array(times),
            {r: np.array(histories[r]) for r in _PROBE_RADII},
            np.array(shell_energy))


def test_monopole_radiation_matches_analytic_free_field():
    # ppw = 6 particles per wavelength. This is BELOW the 10-15 the design spec
    # names as the SPH accuracy floor, and that is the substantive reason this
    # test's amplitude check is a bounded-decay check rather than a precise 1/r
    # fit -- at ppw=6 numerical attenuation dominates geometric spreading (see
    # the SCOPE section of the module docstring and the amplitude assertions
    # below). Keeping the test fast is a secondary benefit, not the main
    # reason. Raising ppw here without also lengthening the run and widening
    # the probe span will not make a precise 1/r assertion pass.
    sim = _make_sim(freq=1000.0, ppw=6, n_per_axis=20)
    freq = sim["freq"]
    times, histories, _ = _run(sim, n_steps=900, damping_max=sim["damping_max"])

    # 900 steps is ~58 periods of the 1 kHz drive. The startup transient needs
    # ~40 periods to leave the domain through the sponge; measuring earlier
    # (the original 400 steps, ~26 periods) samples a still-decaying transient
    # whose dominant spectral content sits well below the driven frequency
    # (measured: ~770 Hz at 26 periods vs ~985 Hz at 58 periods). Measuring in
    # the last quarter of a 900-step run is what makes the frequency assertion
    # below reliable rather than marginal.
    steady_start = int(0.75 * 900)

    # --- arrival time: first probe sample exceeding a small detection threshold ---
    threshold = 0.02  # Pa, well above numerical noise floor
    arrival = {}
    for r in _PROBE_RADII:
        idx = np.argmax(np.abs(histories[r]) > threshold)
        assert idx > 0, f"probe at r={r} never exceeded the detection threshold"
        arrival[r] = times[idx]
    for r in _PROBE_RADII:
        assert arrival[r] == pytest.approx(r / sph.C0, rel=0.25)
    # Margin note (measured, so a future maintainer does not tighten this
    # blindly): detected arrival runs systematically EARLY, by ~16-22% at the
    # outermost probe. That is geometric, not a wave-speed error -- the probe
    # averages particles within probe_sample_radius (1.0*dx) of the sample
    # point, so its inner edge sits up to dx closer to the source and sees the
    # wavefront first: dx/r is ~15% at r=0.37. On top of that, threshold
    # detection quantizes to the step grid, and one step is ~6% of r/c0 here,
    # so the observed value jitters by one step between runs (measured: 0.156
    # and 0.217 across 20 runs, never above 0.217). rel=0.25 therefore holds
    # with only about half a timestep to spare at r=0.37. If this ever starts
    # flaking, the fix is a finer detection (interpolate the threshold crossing
    # between samples) or correcting for the probe's inner-edge offset -- NOT
    # simply widening the tolerance, which would stop it being a wave-speed
    # check.

    # --- amplitude falloff: bounded by geometric spreading, not fitted to it ---
    amp = {r: float(np.max(np.abs(histories[r][steady_start:]))) for r in _PROBE_RADII}
    for r in _PROBE_RADII:
        assert amp[r] > 0.0
    r_near, r_mid, r_far = _PROBE_RADII

    # (1) Monotonic decay. No tolerance needed: a correctly-signed outgoing
    # monopole field must weaken with radius. A bug that inverted the pressure
    # gradient, flattened the field, or turned the sponge into an amplifier
    # violates this immediately.
    assert amp[r_near] > amp[r_mid] > amp[r_far], (
        f"amplitude is not monotonically decreasing with radius: {amp}"
    )

    # (2) Bounded decay rate. Pure geometric spreading gives amp ~ 1/r, i.e.
    # amp(r_near)/amp(r_far) == r_far/r_near. Any additional dissipation --
    # physical attenuation, artificial viscosity, or the scheme's own numerical
    # damping -- can only make the field decay FASTER than geometric, never
    # slower. So geometric spreading is a genuine physical lower bound on the
    # measured ratio, not an empirical fit. The 0.85 factor is measurement
    # noise allowance only (observed ratio/geometric spans 1.25-1.50 across
    # repeated runs, so this bound has wide margin).
    ratio_measured = amp[r_near] / amp[r_far]
    ratio_expected_geometric = r_far / r_near
    assert ratio_measured >= 0.85 * ratio_expected_geometric, (
        f"amplitude decays SLOWER than geometric spreading (ratio {ratio_measured:.3f} vs "
        f"geometric {ratio_expected_geometric:.3f}) -- non-physical, suggests reflection "
        "from an inert boundary or an amplifying source"
    )
    # Generous upper bound: catches a catastrophic collapse (amplitude falling
    # by orders of magnitude between probes) while tolerating the strong
    # numerical attenuation expected at ppw=6.
    assert ratio_measured <= 5.0 * ratio_expected_geometric, (
        f"amplitude collapses far faster than geometric spreading (ratio {ratio_measured:.3f} "
        f"vs geometric {ratio_expected_geometric:.3f}) -- suggests runaway dissipation"
    )

    # --- frequency: dominant spectral peak at the nearest probe's steady window ---
    # Estimated by FFT peak rather than by counting zero crossings. Zero-crossing
    # period estimation over a ~15-period window is intrinsically noisy: the
    # probe signal carries a small DC offset and residual broadband noise, so a
    # single missed or spurious crossing shifts the mean period a lot. Measured
    # directly, that estimator scattered over ~700-1010 Hz run to run and failed
    # this assertion in roughly 1 run in 10 -- while the field's actual dominant
    # frequency was stable at 983-1026 Hz the whole time. The flakiness was in
    # the measurement, not the physics, so the measurement was replaced.
    signal = histories[r_near][steady_start:]
    signal_t = times[steady_start:]
    sample_dt = float(np.mean(np.diff(signal_t)))
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))  # mean removed: ignore DC offset
    bin_freqs = np.fft.rfftfreq(len(signal), d=sample_dt)
    peak = int(np.argmax(spectrum[1:])) + 1  # skip the DC bin
    assert spectrum[peak] > 0.0, "probe signal has no spectral content"
    # Parabolic interpolation across the peak and its neighbors, for sub-bin
    # resolution (bin width here is ~68 Hz, coarse relative to a 1 kHz drive).
    measured_freq = bin_freqs[peak]
    if 0 < peak < len(spectrum) - 1:
        lo, mid, hi = spectrum[peak - 1], spectrum[peak], spectrum[peak + 1]
        denom = lo - 2.0 * mid + hi
        if denom != 0.0:
            offset = 0.5 * (lo - hi) / denom
            if abs(offset) <= 1.0:  # guard against a degenerate fit
                measured_freq = (peak + offset) * (bin_freqs[1] - bin_freqs[0])
    # rel=0.30 unchanged from the original assertion. The estimator is far more
    # accurate than that (measured within ~3% of the drive across repeated
    # runs), so the tolerance is not load-bearing -- it is kept wide because
    # this test's purpose is to catch a wrong driven frequency or wrong wave
    # speed, both of which fail it by a wide margin, not to pin down the
    # spectrum precisely.
    assert measured_freq == pytest.approx(freq, rel=0.30)


def test_sponge_measurably_absorbs_outgoing_waves():
    """The sponge boundary must actually absorb, not merely be wired in.

    This is a "does this component do anything" test. The original code used a
    fixed damping_max=200.0, which is ~100x too weak at this geometry: the
    radiation test above passed identically with the sponge fully disabled, so
    a completely inert absorbing boundary went unnoticed. Comparing the
    acoustic energy that accumulates in the sponge shell with the sponge on
    versus off makes inertness a test failure.
    """
    n_steps = 250  # enough for the wave to reach and fill the shell

    sim_on = _make_sim()
    _, _, energy_on = _run(sim_on, n_steps, damping_max=sim_on["damping_max"])

    # Rebuild from scratch: _run mutates solver state, so the two runs must not
    # share a simulation.
    sim_off = _make_sim()
    assert sim_off["damping_max"] > 0.0, "sponge strength formula produced no damping"
    _, _, energy_off = _run(sim_off, n_steps, damping_max=0.0)

    window = int(0.6 * n_steps)
    mean_on = float(np.mean(energy_on[window:]))
    mean_off = float(np.mean(energy_off[window:]))

    assert mean_on > 0.0 and mean_off > 0.0, "no acoustic energy reached the sponge shell"
    # Measured ratio is ~10-11x on a healthy sponge. The original broken
    # damping_max=200 configuration (before this test existed) measures
    # ~2.9-3.1x -- right on top of a naive 3x threshold, which would only
    # catch that regression about 60% of the time. 6x keeps ~1.8x margin
    # below the healthy value while rejecting the broken one reliably.
    assert mean_off >= 6.0 * mean_on, (
        f"sponge is not measurably absorbing: shell energy with sponge on={mean_on:.3f}, "
        f"off={mean_off:.3f} (ratio {mean_off / mean_on:.2f}x, need >= 6x). "
        "damping_max is probably too weak to absorb a wave within the shell thickness."
    )
