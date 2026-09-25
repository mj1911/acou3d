"""Core SPH numerics: cubic-spline kernel, density, pressure, viscosity,
leapfrog integration.

Units are SI throughout (m, kg, s, Pa). The equation of state is the
linearized acoustic form p' = c0^2 * (rho - rho0), valid for the small
pressure perturbations that make up sound (not the nonlinear Tait EOS
used for incompressible-liquid SPH).
"""
import taichi as ti

RHO0 = 1.204        # air density, kg/m^3, 20C
C0 = 343.0          # speed of sound, m/s
ALPHA_VISC = 0.05   # artificial viscosity coefficient (numerical stabilization only)


@ti.func
def kernel_w(r: ti.f32, h: ti.f32) -> ti.f32:
    """Monaghan cubic-spline kernel value at distance r, smoothing length h (3D)."""
    q = r / h
    sigma = 1.0 / (ti.math.pi * h ** 3)
    result = 0.0
    if q < 1.0:
        result = sigma * (1.0 - 1.5 * q ** 2 + 0.75 * q ** 3)
    elif q < 2.0:
        result = sigma * 0.25 * (2.0 - q) ** 3
    return result


@ti.func
def kernel_dwdr(r: ti.f32, h: ti.f32) -> ti.f32:
    """d(W)/dr at distance r (radial derivative, not the gradient vector)."""
    q = r / h
    sigma = 1.0 / (ti.math.pi * h ** 3)
    result = 0.0
    if q < 1.0e-12:
        result = 0.0
    elif q < 1.0:
        result = sigma * (-3.0 * q + 2.25 * q ** 2) / h
    elif q < 2.0:
        result = sigma * (-0.75 * (2.0 - q) ** 2) / h
    return result


@ti.func
def kernel_grad_w(rij, h: ti.f32):
    """Gradient of W with respect to particle i's position, for separation
    vector rij = pos_i - pos_j."""
    r = rij.norm()
    dw = kernel_dwdr(r, h)
    grad = ti.Vector([0.0, 0.0, 0.0])
    if r > 1e-12:
        grad = dw * (rij / r)
    return grad
