"""Validation tools: an independent grid-based reference solver and the shared
energy-arrival measurement.

The reference is a 3D linear-acoustics FDTD solver (staggered grid, pure
numpy -- no Taichi, no SPH), driven by the same smooth radial body force as
the SPH demo. Running both through the same probes and the same measurement
separates "the SPH scheme is wrong" from "the measurement or the expectation
is wrong": the reference is the trusted answer.

    python -m air_sph.validation          # compare SPH and FDTD on the demo burst
"""
import argparse

import numpy as np

C0 = 343.0
RHO0 = 1.204


def energy_arrival(ts, p):
    """Energy-weighted arrival time: the centroid of p(t)^2."""
    e = np.asarray(p) ** 2
    return float((np.asarray(ts) * e).sum() / e.sum())


def energy_speed(arrivals, radii):
    """Speed from a straight-line fit of radius against energy-arrival time."""
    return float(np.polyfit(arrivals, radii, 1)[0])


def burst_lag(ts, p, ts_ref, p_ref, window, max_lag=0.5e-3, step=2e-6):
    """Delay of p relative to p_ref over a time window, by cross-correlation.

    Both signals are resampled onto a common grid, the reference is shifted by
    each candidate lag, and the lag with the highest normalised correlation
    inside `window` = (t0, t1) wins. Unlike energy_arrival, anything outside
    the window (e.g. a post-burst tail) cannot bias the result.
    Returns (lag, correlation, rms gain of p over p_ref in the window).
    """
    grid = np.arange(0.0, min(ts[-1], ts_ref[-1]), step)
    a = np.interp(grid, ts, p)
    w = (grid > window[0]) & (grid < window[1])
    best = (-np.inf, 0.0, 0.0)
    for lag in np.arange(-max_lag, max_lag + step / 2, step):
        b = np.interp(grid - lag, ts_ref, p_ref)
        c = (a[w] * b[w]).sum() / np.sqrt((a[w] ** 2).sum() * (b[w] ** 2).sum())
        if c > best[0]:
            best = (c, lag, np.sqrt((a[w] ** 2).sum() / (b[w] ** 2).sum()))
    return best[1], best[0], best[2]


def fdtd_monopole(accel_fn, sigma, radii, t_end, dx, half=1.1, sponge=0.4, shell_halfwidth=None):
    """Run the FDTD reference and return (ts, P), P[i, j] = mean pressure on shell radii[j].

    The source is the same force field as source.apply_smooth_monopole: a body
    acceleration accel_fn(t) * (r_vec / sigma) * exp(-r^2 / (2 sigma^2)).
    A graded sponge in the outer `sponge` metres of the cube absorbs outgoing
    waves. For negligible grid dispersion use dx <= wavelength / 20.
    """
    n = int(round(2 * half / dx))
    dt = 0.5 * dx / (C0 * np.sqrt(3))            # half the 3D CFL limit
    cc = (np.arange(n) + 0.5) * dx - half        # cell centres (pressure)
    fc = np.arange(n + 1) * dx - half            # cell faces (normal velocity)

    def radial(along, a, b):
        return (along / sigma * np.exp(-(along**2 + a**2 + b**2) / (2 * sigma**2))).astype(np.float32)

    fx = radial(fc[:, None, None], cc[None, :, None], cc[None, None, :])
    fy = radial(fc[None, :, None], cc[:, None, None], cc[None, None, :])
    fz = radial(fc[None, None, :], cc[:, None, None], cc[None, :, None])

    x, y, z = np.meshgrid(cc, cc, cc, indexing="ij")
    r = np.sqrt(x**2 + y**2 + z**2)
    s = np.clip((r - (half - sponge)) / sponge, 0.0, 1.0)
    decay = np.exp(-5.0 * C0 / sponge * s**2 * dt).astype(np.float32)

    p = np.zeros((n, n, n), np.float32)
    vx = np.zeros((n + 1, n, n), np.float32)
    vy = np.zeros((n, n + 1, n), np.float32)
    vz = np.zeros((n, n, n + 1), np.float32)
    hw = 0.5 * dx if shell_halfwidth is None else shell_halfwidth
    shells = [np.abs(r - rp) < hw for rp in radii]
    k = dt / (RHO0 * dx)
    ts, rows, t = [], [], 0.0
    while t < t_end:
        a = accel_fn(t)
        vx[1:-1] -= k * (p[1:] - p[:-1])
        vy[:, 1:-1] -= k * (p[:, 1:] - p[:, :-1])
        vz[:, :, 1:-1] -= k * (p[:, :, 1:] - p[:, :, :-1])
        vx += dt * a * fx
        vy += dt * a * fy
        vz += dt * a * fz
        vx[1:-1] *= decay[1:]
        vy[:, 1:-1] *= decay[:, 1:]
        vz[:, :, 1:-1] *= decay[:, :, 1:]
        p -= dt * RHO0 * C0**2 * (vx[1:] - vx[:-1] + vy[:, 1:] - vy[:, :-1] + vz[:, :, 1:] - vz[:, :, :-1]) / dx
        p *= decay
        t += dt
        ts.append(t)
        rows.append([p[m].mean() for m in shells])
    return np.array(ts), np.array(rows)


def sph_monopole(freq, ppw, n_per_axis, radii, t_end, amplitude, shell_halfwidth_dx=1.0):
    """Run the SPH demo pipeline and return (ts, P) like fdtd_monopole.

    Taichi must already be initialised. Also returns the SPH smoothing length
    so callers can size the matching FDTD source.
    """
    import taichi as ti

    from air_sph import demo, sph

    solver, grid, dx, half = demo.build_sim(freq, ppw, n_per_axis)
    r = np.linalg.norm(solver.pos.to_numpy(), axis=1)
    dt = 0.3 * solver.h / sph.C0
    r_start, r_domain = 0.7 * half, half
    damping_max = 5.0 * sph.C0 / (r_domain - r_start)
    center = ti.Vector([0.0, 0.0, 0.0])
    shells = [np.abs(r - rp) < shell_halfwidth_dx * dx for rp in radii]
    ts, rows, t = [], [], 0.0
    while t < t_end:
        amp = demo.burst_amplitude(t, freq, amplitude)
        demo.step(solver, grid, dt, t, freq, amp, center, r_start, r_domain, damping_max)
        t += dt
        ts.append(t)
        pressure = solver.pressure.to_numpy()
        rows.append([pressure[m].mean() for m in shells])
    return np.array(ts), np.array(rows), solver.h


def main():
    from air_sph import demo

    parser = argparse.ArgumentParser(description="Compare the SPH demo with the FDTD reference.")
    parser.add_argument("--freq", type=float, default=demo.DEFAULT_FREQ)
    parser.add_argument("--ppw", type=float, default=demo.DEFAULT_PPW)
    parser.add_argument("--n", type=int, default=demo.DEFAULT_N)
    args = parser.parse_args()

    demo.init_taichi(offline=False)
    radii = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    t_end = demo.burst_end_time(args.freq) + radii[-1] / C0 + 3.0 / args.freq

    ts_s, p_s, h = sph_monopole(args.freq, args.ppw, args.n, radii, t_end, demo.DEFAULT_AMPLITUDE)
    sigma = demo.SOURCE_SIGMA_H * h

    def accel(t):
        return demo.source_accel(t, args.freq, demo.burst_amplitude(t, args.freq, demo.DEFAULT_AMPLITUDE))

    ts_f, p_f = fdtd_monopole(accel, sigma, radii, t_end, dx=C0 / args.freq / 40)

    burst_end = demo.burst_end_time(args.freq)
    print(f"{args.freq:.0f} Hz, source sigma {sigma * 1e3:.0f} mm")
    print("  r (m)   burst lag SPH-FDTD   waveform corr   gain SPH/FDTD   energy after burst (SPH)")
    times = []
    for j, rp in enumerate(radii):
        window = (burst_end - demo.ACTIVE_CYCLES / args.freq + rp / C0, burst_end + rp / C0)
        lag, corr, gain = burst_lag(ts_s, p_s[:, j], ts_f, p_f[:, j], window)
        times.append(rp / C0 + lag)
        e = p_s[:, j] ** 2
        tail = e[ts_s > burst_end + rp / C0 + 0.25 / args.freq].sum() / e.sum()
        print(f"  {rp:.2f}   {lag * 1e3:+10.3f} ms   {corr:12.3f}   {gain:12.2f}   {tail:14.1%}")
    print(f"SPH burst speed from lags: {energy_speed(times, radii):.0f} m/s (c0 = {C0:.0f})")


if __name__ == "__main__":
    main()
