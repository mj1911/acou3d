"""Sponge-layer open/absorbing boundary: damps particle velocity near the
domain edge so outgoing waves attenuate instead of reflecting back, so
results are comparable to a free-field (infinite space) analytic solution.
"""
import taichi as ti


@ti.kernel
def apply_sponge(pos: ti.template(), vel: ti.template(), n: ti.i32,
                  center: ti.types.vector(3, ti.f32), r_start: ti.f32, r_domain: ti.f32,
                  damping_max: ti.f32, dt: ti.f32):
    for i in range(n):
        r = (pos[i] - center).norm()
        if r > r_start:
            s = min((r - r_start) / (r_domain - r_start), 1.0)
            coeff = damping_max * s * s
            vel[i] *= ti.exp(-coeff * dt)
