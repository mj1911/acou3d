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


@ti.kernel
def relax_pressure(pos: ti.template(), rho: ti.template(), rho0: ti.template(), n: ti.i32,
                   center: ti.types.vector(3, ti.f32), r_start: ti.f32, r_domain: ti.f32,
                   damping_max: ti.f32, dt: ti.f32):
    """Damp pressure in the sponge shell at the same rate apply_sponge damps velocity.

    Pressure is c0^2 * (rho - rho0), so relaxing each particle's rest density
    rho0 toward its current rho at rate s gives dp/dt = ... - s * p. Damping
    pressure and velocity equally keeps the shell's acoustic impedance equal
    to rho0 * c0: the outgoing and incoming wave components stay decoupled,
    so an outgoing wave is absorbed as it travels in instead of partly
    reflecting off the damping gradient (exact at normal incidence, for any
    ramp). Damping velocity alone, as apply_sponge does, is an impedance
    mismatch and reflects.

    Uses the same quadratic ramp and damping_max as apply_sponge; call both.
    """
    for i in range(n):
        r = (pos[i] - center).norm()
        if r > r_start:
            s = min((r - r_start) / (r_domain - r_start), 1.0)
            rho0[i] += (1.0 - ti.exp(-damping_max * s * s * dt)) * (rho[i] - rho0[i])
