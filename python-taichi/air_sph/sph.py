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


@ti.data_oriented
class Solver:
    def __init__(self, n, h, mass, grid):
        self.n = n
        self.h = h
        self.mass = mass
        self.grid = grid
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.acc = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.rho = ti.field(dtype=ti.f32, shape=n)
        self.pressure = ti.field(dtype=ti.f32, shape=n)
        self.rho0 = ti.field(dtype=ti.f32, shape=n)
        self.rho0.fill(RHO0)

    @ti.kernel
    def compute_density(self):
        for i in range(self.n):
            h = self.h
            rho_i = 0.0
            c = self.grid.cell_coord(self.pos[i])
            for di, dj, dk in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                cc = c + ti.Vector([di, dj, dk])
                if 0 <= cc[0] < self.grid.n_cells and 0 <= cc[1] < self.grid.n_cells and 0 <= cc[2] < self.grid.n_cells:
                    cnt = min(self.grid.cell_count[cc[0], cc[1], cc[2]], self.grid.max_per_cell)
                    for s in range(cnt):
                        j = self.grid.cell_particles[cc[0], cc[1], cc[2], s]
                        r = (self.pos[i] - self.pos[j]).norm()
                        if r < 2.0 * h:
                            rho_i += self.mass * kernel_w(r, h)
            self.rho[i] = rho_i

    @ti.kernel
    def capture_rest_density(self):
        """Freeze each particle's current density as its own pressure
        reference. Call once at t=0 for a quiescent, free-boundary lattice:
        edge/corner particles have truncated kernel support and so
        under-estimate density against the global RHO0 (a discretization
        artifact, not a real density gradient); comparing against their own
        captured value instead makes pressure exactly zero everywhere at
        t=0, regardless of that artifact."""
        for i in range(self.n):
            self.rho0[i] = self.rho[i]

    @ti.kernel
    def compute_pressure(self):
        for i in range(self.n):
            self.pressure[i] = C0 ** 2 * (self.rho[i] - self.rho0[i])

    @ti.kernel
    def compute_forces(self):
        for i in range(self.n):
            h = self.h
            a = ti.Vector([0.0, 0.0, 0.0])
            pi_over_rho2 = self.pressure[i] / (self.rho[i] ** 2)
            c = self.grid.cell_coord(self.pos[i])
            for di, dj, dk in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                cc = c + ti.Vector([di, dj, dk])
                if 0 <= cc[0] < self.grid.n_cells and 0 <= cc[1] < self.grid.n_cells and 0 <= cc[2] < self.grid.n_cells:
                    cnt = min(self.grid.cell_count[cc[0], cc[1], cc[2]], self.grid.max_per_cell)
                    for s in range(cnt):
                        j = self.grid.cell_particles[cc[0], cc[1], cc[2], s]
                        if j != i:
                            rij = self.pos[i] - self.pos[j]
                            r = rij.norm()
                            if r < 2.0 * h:
                                gw = kernel_grad_w(rij, h)
                                pj_over_rho2 = self.pressure[j] / (self.rho[j] ** 2)
                                a -= self.mass * (pi_over_rho2 + pj_over_rho2) * gw

                                vij = self.vel[i] - self.vel[j]
                                vr = vij.dot(rij)
                                if vr < 0.0:
                                    rho_bar = 0.5 * (self.rho[i] + self.rho[j])
                                    mu = h * vr / (rij.dot(rij) + 0.01 * h * h)
                                    pi_visc = (-ALPHA_VISC * C0 * mu) / rho_bar
                                    a -= self.mass * pi_visc * gw
            self.acc[i] = a

    @ti.kernel
    def leapfrog_predict(self, dt: ti.f32):
        for i in range(self.n):
            self.vel[i] += 0.5 * dt * self.acc[i]
            self.pos[i] += dt * self.vel[i]

    @ti.kernel
    def leapfrog_correct(self, dt: ti.f32):
        for i in range(self.n):
            self.vel[i] += 0.5 * dt * self.acc[i]
