"""Sponge-layer open/absorbing boundary: damps particle velocity near the
domain edge so outgoing waves attenuate instead of reflecting back, so
results are comparable to a free-field (infinite space) analytic solution.
"""
import taichi as ti


@ti.kernel
def apply_sponge(pos: ti.template(), vel: ti.template(), n: ti.i32,
                  center: ti.types.vector(3, ti.f32), r_start: ti.f32, r_domain: ti.f32,
                  damping_max: ti.f32, dt: ti.f32):
    """Damp velocity in the shell r_start < r <= r_domain, ramping quadratically
    from no damping at r_start to full `damping_max` at r_domain.

    Precondition: `damping_max >= 0`. The damping factor is
    exp(-damping_max * s^2 * dt), so a negative `damping_max` would make the
    exponent positive and *amplify* velocity every step instead of damping it,
    growing without bound. Not enforced here (this is a hot per-step kernel and
    no call site passes a negative value); callers must respect it.

    Choosing `damping_max`: it must be strong enough to absorb an outgoing wave
    within the shell's thickness, or the boundary is inert and the domain
    behaves as a reflecting box. Scale it as k * c0 / (r_domain - r_start) with
    k ~ 3-10 -- see the derivation and the sponge-effectiveness test in
    tests/test_monopole.py.
    """
    for i in range(n):
        r = (pos[i] - center).norm()
        if r > r_start:
            s = min((r - r_start) / (r_domain - r_start), 1.0)
            coeff = damping_max * s * s
            vel[i] *= ti.exp(-coeff * dt)
