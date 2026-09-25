"""The demo's burst must travel as a compact pulse at close to c0.

test_monopole.py checks the *first arrival* of the wave, which travels at c0
at any resolution. That missed a real problem: SPH kernel smoothing makes the
particle medium dispersive, with a frequency ceiling (~1.27 kHz at 34 mm
spacing, where waves ~4.8 spacings long stop carrying energy). The old demo
drove a hard-gated single 1 kHz cycle right under that ceiling, so most of
the burst's energy crawled outward at ~170 m/s and rang near the source long
after the drive stopped.

The real culprit turned out to be the source, not the drive frequency: the
old source prescribed velocities on the 8 particles nearest the centre, whose
alternating directions put ~99% of the output into SPH waves 2-3 spacings
long, past the dispersion ceiling, carrying energy at ~110-175 m/s. The demo
now drives a smooth Gaussian radial body force (source.apply_smooth_monopole).

These checks run the demo's own pipeline (build_sim, step, burst_amplitude,
demo defaults) and compare it with the FDTD reference (air_sph.validation),
driven by the identical force:
  - each probe's burst must match the reference's in shape and timing
    (windowed cross-correlation), and the lag's change with radius must
    give a speed close to c0;
  - almost no energy may arrive after the burst has passed, i.e. the
    absorbing shell must not reflect (boundary.relax_pressure).

Shells are one particle spacing thick (half-width dx) so each probe averages
enough particles to be stable.
"""
from air_sph import demo, sph, validation

# Inside the sponge (starts ~0.47 m). Probes may sit inside the source's forced
# region: the reference applies the identical force, so the comparison is fair,
# and the wide 0.3 m span keeps the speed estimate well-conditioned.
RADII = (0.10, 0.20, 0.30, 0.40)
R_FAR = RADII[-1]


def test_burst_matches_reference_and_travels_near_c0():
    """SPH's burst must match the FDTD reference's in shape and timing.

    Each probe's burst is cross-correlated against the reference inside the
    drive window, so post-burst tails can't bias it (they did bias the
    earlier energy-centroid version of this test, in both directions).
    Speed comes from how the lag changes with radius. Measured: lags of
    +0.06 to +0.09 ms, correlations >= 0.98.
    """
    freq = demo.DEFAULT_FREQ
    burst_end = demo.burst_end_time(freq)
    t_end = burst_end + R_FAR / sph.C0 + 3.0 / freq
    ts, p, h = validation.sph_monopole(freq, demo.DEFAULT_PPW, demo.DEFAULT_N, RADII, t_end,
                                       demo.DEFAULT_AMPLITUDE)

    def accel(t):
        return demo.source_accel(t, freq, demo.burst_amplitude(t, freq, demo.DEFAULT_AMPLITUDE))

    ts_ref, p_ref = validation.fdtd_monopole(accel, demo.SOURCE_SIGMA_H * h, RADII, t_end,
                                             dx=sph.C0 / freq / 20, half=0.8, sponge=0.3)
    times = []
    for j, rp in enumerate(RADII):
        window = (burst_end - demo.ACTIVE_CYCLES / freq + rp / sph.C0, burst_end + rp / sph.C0)
        lag, corr, _ = validation.burst_lag(ts, p[:, j], ts_ref, p_ref[:, j], window)
        assert corr > 0.95, f"burst waveform at r={rp} m doesn't match the reference (corr {corr:.3f})"
        assert abs(lag) < 0.25e-3, f"burst at r={rp} m lags the reference by {lag * 1e3:.2f} ms"
        times.append(rp / sph.C0 + lag)
    speed = validation.energy_speed(times, RADII)
    assert 0.9 * sph.C0 < speed < 1.1 * sph.C0, (
        f"burst travels at {speed:.0f} m/s, expected close to c0={sph.C0:.0f} m/s")


def test_absorbing_shell_does_not_reflect_the_burst():
    """Energy still arriving after the burst has passed must stay small.

    With the old velocity-only sponge, 5-16% of the probe energy came back as
    reflections from the 0.2 m shell (under a third of a wavelength at
    500 Hz). The same SPH run in a doubled domain with a 0.4 m shell -- where
    reflections can't return in time -- leaves at most ~5%, and the FDTD
    reference ~0%. An impedance-matched shell should reach that level in the
    default domain.
    """
    freq = demo.DEFAULT_FREQ
    radii = (0.20, 0.30, 0.40)
    burst_end = demo.burst_end_time(freq)
    ts, p, _ = validation.sph_monopole(freq, demo.DEFAULT_PPW, demo.DEFAULT_N, radii,
                                       burst_end + 8.0 / freq, demo.DEFAULT_AMPLITUDE)
    for j, rp in enumerate(radii):
        e = p[:, j] ** 2
        tail = e[ts > burst_end + rp / sph.C0 + 0.25 / freq].sum() / e.sum()
        assert tail < 0.06, f"{tail:.1%} of the energy at r={rp} m arrives after the burst"
