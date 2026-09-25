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
