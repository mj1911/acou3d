"""The FDTD reference must itself be right before it can judge the SPH scheme."""
import numpy as np
import pytest

from air_sph import demo, validation


def test_energy_arrival_tracks_a_delay():
    ts = np.linspace(0, 0.01, 2001)
    pulse = lambda t0: np.sin(2 * np.pi * 800 * ts) * np.exp(-((ts - t0) / 1e-3) ** 2)
    shift = validation.energy_arrival(ts, pulse(0.006)) - validation.energy_arrival(ts, pulse(0.004))
    assert shift == pytest.approx(0.002, rel=1e-3)


def test_fdtd_reference_burst_travels_at_c0():
    freq = 500.0
    radii = np.array([0.20, 0.25, 0.30, 0.35, 0.40])
    sigma = 0.067

    def accel(t):
        return demo.source_accel(t, freq, demo.burst_amplitude(t, freq, 0.05))

    t_end = demo.burst_end_time(freq) + radii[-1] / validation.C0 + 2.0 / freq
    ts, p = validation.fdtd_monopole(accel, sigma, radii, t_end,
                                     dx=validation.C0 / freq / 20, half=0.8, sponge=0.3)
    arrivals = np.array([validation.energy_arrival(ts, p[:, j]) for j in range(len(radii))])
    assert validation.energy_speed(arrivals, radii) == pytest.approx(validation.C0, rel=0.03)
    # a real spherical wave: r * p stays roughly constant with radius
    r_rms = radii * np.sqrt((p**2).mean(0))
    assert r_rms.max() / r_rms.min() < 1.3
