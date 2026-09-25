"""Prescribed-velocity monopole point source: a small cluster of particles
driven with a radially oscillating velocity, approximating a pulsating
point source for validation against analytic monopole radiation.
"""
import taichi as ti


@ti.kernel
def apply_monopole(pos: ti.template(), vel: ti.template(), is_source: ti.template(), n: ti.i32,
                    center: ti.types.vector(3, ti.f32), amplitude: ti.f32, freq: ti.f32, t: ti.f32):
    for i in range(n):
        if is_source[i] == 1:
            rvec = pos[i] - center
            r = rvec.norm()
            direction = ti.Vector([0.0, 0.0, 0.0])
            if r > 1e-9:
                direction = rvec / r
            vel[i] = amplitude * ti.sin(2.0 * ti.math.pi * freq * t) * direction


@ti.kernel
def apply_smooth_monopole(pos: ti.template(), vel: ti.template(), n: ti.i32,
                          center: ti.types.vector(3, ti.f32), accel: ti.f32, sigma: ti.f32, dt: ti.f32):
    """Drive a monopole with a smooth radial body force instead of prescribed velocities.

    Adds dt * accel * (r_vec / sigma) * exp(-r^2 / (2 sigma^2)) to every
    particle's velocity: radial, zero at the centre, and Gaussian-smooth in
    space. With sigma of ~2 particle spacings its spatial spectrum is
    negligible at lattice scale.

    Why not apply_monopole: prescribing velocities on the few particles nearest
    the centre makes neighbouring particles move in opposite directions -- a
    pattern ~1 spacing long. That couples ~99% of the output into SPH waves
    2-3 spacings long, past the dispersion ceiling, which carry energy at
    only ~110-175 m/s (measured). Real, long-wavelength sound got ~1%.
    """
    for i in range(n):
        rvec = pos[i] - center
        r2 = rvec.dot(rvec)
        vel[i] += dt * accel * (rvec / sigma) * ti.exp(-r2 / (2.0 * sigma * sigma))
