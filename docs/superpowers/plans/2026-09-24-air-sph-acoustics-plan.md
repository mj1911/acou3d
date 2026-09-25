# Particle-Based Air Acoustics (python-taichi) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3D, GPU-accelerated (Taichi), SPH-based simulation of sound
propagating through air as a particle system, validated against analytic
free-field acoustics, with a real-time interactive viewer.

**Architecture:** Weakly-compressible SPH with a *linearized acoustic*
equation of state (`p' = c0^2 * (rho - rho0)`), a uniform spatial grid for
neighbor search, leapfrog time integration, a prescribed-velocity monopole
source, and a sponge-layer open boundary. Built bottom-up with unit tests on
each numerical piece, then two system-level validation tests (still-box
stability, monopole radiation vs. analytic solution), then a real-time
Taichi GGUI viewer wired into a CLI.

**Tech Stack:** Python, Taichi (GPU-parallel kernels + GGUI), NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-air-sph-acoustics-design.md`

## Global Constraints

- New top-level folder `python-taichi/` at the repo root — independent of
  `python-pyvista/`, no code dependency between them.
- SI units throughout (m, kg, s, Pa).
- `RHO0 = 1.204` kg/m^3, `C0 = 343.0` m/s (air, 20°C) — defined once in
  `air_sph/sph.py`, imported everywhere else that needs them.
- Equation of state is the linear acoustic form `p' = c0^2 * (rho - rho0)`
  — not the reference project's nonlinear Tait EOS.
- No coupling to `python-pyvista`'s driver/cabinet/BEM model in this plan
  (spec Non-goals). Validation is against closed-form analytic acoustics
  only.
- Testing style matches `python-pyvista`: pytest, `pytest.approx` /
  `np.testing.assert_allclose` with explicit quantitative tolerances, no
  mocking of the physics.
- A note on tolerances used below: this is a first implementation of a
  novel numerical solver that cannot be validated by hand ahead of time.
  Tolerances are deliberately generous starting points (documented inline
  with the reasoning); if a test fails outside these bounds, that is a real
  bug to fix, not a tolerance to loosen without understanding why.

---

## Task 1: Project scaffolding

**Files:**
- Create: `python-taichi/README.md`
- Create: `python-taichi/requirements.txt`
- Create: `python-taichi/.gitignore`
- Create: `python-taichi/air_sph/__init__.py`
- Create: `python-taichi/tests/__init__.py`
- Test: `python-taichi/tests/test_setup.py`

**Interfaces:**
- Produces: an importable `air_sph` package; a working Taichi install.

- [ ] **Step 1: Create the directory structure and package files**

```bash
mkdir -p python-taichi/air_sph python-taichi/tests
```

`python-taichi/air_sph/__init__.py`:
```python
"""Particle-based (SPH) air acoustics simulation."""
```

`python-taichi/tests/__init__.py`: empty file.

`python-taichi/.gitignore`:
```
.venv/
venv/
.pytest_cache/
__pycache__/
```

`python-taichi/requirements.txt`:
```
taichi==1.7.3
numpy==2.5.3
pytest==9.1.1
```

`python-taichi/README.md`:
```markdown
# acou3d: Python / Taichi approach

Particle-based (SPH) simulation of sound propagating through air, in 3D,
using Taichi for GPU-parallel kernels and a real-time viewer. Modeled
architecturally on [SebLague/Fluid-Sim](https://github.com/SebLague/Fluid-Sim)
(GPU SPH neighbor search + solver + viewer), but solving linear acoustic
wave propagation in a compressible gas instead of incompressible fluid flow.

See `docs/superpowers/specs/2026-09-24-air-sph-acoustics-design.md` (repo
root) for the full design.

## Setup

\`\`\`bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
\`\`\`

## Run

\`\`\`bash
.venv/bin/python -m air_sph.demo                  # interactive viewer
.venv/bin/python -m air_sph.demo --offline         # headless, prints probe data
.venv/bin/python -m pytest -q
\`\`\`
```

- [ ] **Step 2: Write the failing setup test**

`python-taichi/tests/test_setup.py`:
```python
import taichi as ti

import air_sph


def test_taichi_initializes_on_cpu():
    ti.init(arch=ti.cpu)

    @ti.kernel
    def one() -> ti.i32:
        return 1

    assert one() == 1
```

- [ ] **Step 3: Create venv, install, and run the test**

```bash
cd python-taichi
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/test_setup.py -v
```

Expected: PASS. If `pip install` fails on the `taichi` version pin, adjust
it to the latest version compatible with the installed Python and re-run —
this is expected first-run environment work, not a design change.

- [ ] **Step 4: Commit**

```bash
git add python-taichi/
git commit -m "Scaffold python-taichi subproject"
```

---

## Task 2: SPH cubic-spline kernel

**Files:**
- Create: `python-taichi/air_sph/sph.py`
- Test: `python-taichi/tests/test_sph.py`

**Interfaces:**
- Produces: `sph.RHO0: float`, `sph.C0: float`, `sph.ALPHA_VISC: float`;
  `sph.kernel_w(r: ti.f32, h: ti.f32) -> ti.f32` (`@ti.func`);
  `sph.kernel_dwdr(r: ti.f32, h: ti.f32) -> ti.f32` (`@ti.func`);
  `sph.kernel_grad_w(rij: vector3, h: ti.f32) -> vector3` (`@ti.func`).

- [ ] **Step 1: Write the failing kernel tests**

`python-taichi/tests/test_sph.py`:
```python
import numpy as np
import pytest
import taichi as ti

from air_sph import sph

ti.init(arch=ti.cpu)


@ti.kernel
def _eval_w(r: ti.f32, h: ti.f32) -> ti.f32:
    return sph.kernel_w(r, h)


@ti.kernel
def _eval_dwdr(r: ti.f32, h: ti.f32) -> ti.f32:
    return sph.kernel_dwdr(r, h)


def test_kernel_normalizes_to_one():
    # integral over 3D space of W(r,h) d^3r must equal 1 (kernel normalization)
    h = 0.05
    r = np.linspace(1e-6, 2 * h, 4000)
    w = np.array([_eval_w(float(ri), h) for ri in r])
    integral = np.trapz(4 * np.pi * r ** 2 * w, r)
    assert integral == pytest.approx(1.0, abs=0.01)


def test_kernel_zero_beyond_support():
    h = 0.05
    assert _eval_w(2.5 * h, h) == 0.0


def test_kernel_positive_within_support():
    h = 0.05
    assert _eval_w(0.5 * h, h) > 0
    assert _eval_w(1.5 * h, h) > 0


def test_kernel_derivative_matches_finite_difference():
    h = 0.05
    r = 0.6 * h
    eps = 1e-5
    fd = (_eval_w(r + eps, h) - _eval_w(r - eps, h)) / (2 * eps)
    assert _eval_dwdr(r, h) == pytest.approx(fd, rel=1e-2)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd python-taichi
.venv/bin/python -m pytest tests/test_sph.py -v
```

Expected: FAIL (no `sph` module / no `kernel_w` etc.)

- [ ] **Step 3: Implement the kernel**

`python-taichi/air_sph/sph.py`:
```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_sph.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python-taichi/air_sph/sph.py python-taichi/tests/test_sph.py
git commit -m "Add SPH cubic-spline kernel"
```

---

## Task 3: Spatial grid for neighbor search

**Files:**
- Create: `python-taichi/air_sph/grid.py`
- Test: `python-taichi/tests/test_grid.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Grid(n_cells: int, cell_size: float, grid_min: tuple[float,float,float], max_particles: int, max_per_cell: int = 128)`
  with fields `grid.cell_count[i,j,k]` (`ti.i32`), `grid.cell_particles[i,j,k,slot]` (`ti.i32`);
  methods `grid.clear()`, `grid.build(pos: ti.template(), n: ti.i32)`;
  `@ti.func grid.cell_coord(pos) -> ti.Vector([i,j,k], ti.i32)` (clamped to grid bounds).
  Cell size must equal the kernel support radius (`2*h`) so all neighbors
  within support lie in the particle's own cell or the 26 adjacent cells.

- [ ] **Step 1: Write the failing grid test**

`python-taichi/tests/test_grid.py`:
```python
import numpy as np
import taichi as ti

from air_sph.grid import Grid

ti.init(arch=ti.cpu)


def test_build_places_each_particle_in_exactly_one_cell():
    n = 50
    rng = np.random.default_rng(0)
    pos_np = rng.uniform(-0.4, 0.4, size=(n, 3)).astype(np.float32)

    cell_size = 0.1
    n_cells = 10
    grid_min = (-0.5, -0.5, -0.5)
    grid = Grid(n_cells, cell_size, grid_min, max_particles=n)

    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    pos.from_numpy(pos_np)

    grid.clear()
    grid.build(pos, n)

    counts = grid.cell_count.to_numpy()
    assert counts.sum() == n
    assert counts.max() < grid.max_per_cell


def test_neighbors_of_a_pair_share_or_adjoin_a_cell():
    # two particles closer than cell_size must land in the same or adjacent cell
    cell_size = 0.1
    n_cells = 10
    grid_min = (-0.5, -0.5, -0.5)
    n = 2
    grid = Grid(n_cells, cell_size, grid_min, max_particles=n)

    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    pos.from_numpy(np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32))

    grid.clear()
    grid.build(pos, n)

    c0 = np.array(grid.cell_coord_np(pos.to_numpy()[0]))
    c1 = np.array(grid.cell_coord_np(pos.to_numpy()[1]))
    assert np.all(np.abs(c0 - c1) <= 1)
```

Note: `cell_coord_np` is a small plain-Python helper (not a `@ti.func`) so
the test can compute expected cell coordinates without launching a kernel —
add it alongside the class.

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_grid.py -v
```

Expected: FAIL (no `grid` module)

- [ ] **Step 3: Implement the grid**

`python-taichi/air_sph/grid.py`:
```python
"""Uniform spatial grid for SPH neighbor search.

Cell size must equal the kernel support radius (2h) so that all neighbors
within the support radius of a particle lie in its own cell or the 26
adjacent cells (a 3x3x3 block) — see kernel_w/kernel_grad_w in sph.py.
"""
import numpy as np
import taichi as ti


@ti.data_oriented
class Grid:
    def __init__(self, n_cells, cell_size, grid_min, max_particles, max_per_cell=128):
        self.n_cells = n_cells
        self.cell_size = cell_size
        self.max_per_cell = max_per_cell
        self.grid_min_np = tuple(grid_min)
        self.grid_min = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.grid_min[None] = grid_min
        self.cell_count = ti.field(dtype=ti.i32, shape=(n_cells, n_cells, n_cells))
        self.cell_particles = ti.field(dtype=ti.i32, shape=(n_cells, n_cells, n_cells, max_per_cell))

    @ti.func
    def cell_coord(self, pos):
        rel = (pos - self.grid_min[None]) / self.cell_size
        c = ti.cast(ti.floor(rel), ti.i32)
        return ti.min(ti.max(c, 0), self.n_cells - 1)

    def cell_coord_np(self, pos_np):
        """Plain-Python equivalent of cell_coord, for use outside kernels (tests)."""
        rel = (np.asarray(pos_np) - np.asarray(self.grid_min_np)) / self.cell_size
        c = np.floor(rel).astype(np.int32)
        return np.clip(c, 0, self.n_cells - 1)

    @ti.kernel
    def clear(self):
        for i, j, k in self.cell_count:
            self.cell_count[i, j, k] = 0

    @ti.kernel
    def build(self, pos: ti.template(), n: ti.i32):
        for p in range(n):
            c = self.cell_coord(pos[p])
            slot = ti.atomic_add(self.cell_count[c[0], c[1], c[2]], 1)
            if slot < self.max_per_cell:
                self.cell_particles[c[0], c[1], c[2], slot] = p
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_grid.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python-taichi/air_sph/grid.py python-taichi/tests/test_grid.py
git commit -m "Add spatial grid for SPH neighbor search"
```

---

## Task 4: Particle state and density summation

**Files:**
- Modify: `python-taichi/air_sph/sph.py`
- Modify: `python-taichi/tests/test_sph.py`

**Interfaces:**
- Consumes: `Grid` from Task 3 (`grid.clear()`, `grid.build(pos, n)`,
  `grid.cell_coord`, `grid.cell_count`, `grid.cell_particles`, `grid.n_cells`, `grid.max_per_cell`).
- Produces: `Solver(n: int, h: float, mass: float, grid: Grid)` with fields
  `solver.pos`, `solver.vel`, `solver.acc` (`ti.Vector.field(3, ti.f32, shape=n)`),
  `solver.rho`, `solver.pressure` (`ti.field(ti.f32, shape=n)`); method
  `solver.compute_density()`.

- [ ] **Step 1: Write the failing density test**

Append to `python-taichi/tests/test_sph.py`:
```python
from air_sph.grid import Grid
from air_sph.sph import Solver


def _build_lattice(n_per_axis, dx):
    coords = (np.arange(n_per_axis) - (n_per_axis - 1) / 2) * dx
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)


def _make_lattice_solver(n_per_axis, dx):
    h = 1.3 * dx
    mass = sph.RHO0 * dx ** 3
    pos_np = _build_lattice(n_per_axis, dx)
    n = pos_np.shape[0]
    half_extent = (n_per_axis - 1) * dx / 2
    cell_size = 2.0 * h
    n_cells = int(np.ceil((2 * half_extent + 4 * cell_size) / cell_size))
    grid_min = (-half_extent - 2 * cell_size,) * 3
    grid = Grid(n_cells, cell_size, grid_min, max_particles=n)
    solver = Solver(n, h, mass, grid)
    solver.pos.from_numpy(pos_np)
    return solver, grid, pos_np, half_extent, h


def test_density_matches_rho0_in_bulk():
    solver, grid, pos_np, half_extent, h = _make_lattice_solver(n_per_axis=10, dx=0.02)
    n = pos_np.shape[0]

    grid.clear()
    grid.build(solver.pos, n)
    solver.compute_density()

    rho = solver.rho.to_numpy()
    margin = 2 * h
    interior = np.all(np.abs(pos_np) < (half_extent - margin), axis=1)
    assert interior.sum() > 0
    assert rho[interior] == pytest.approx(sph.RHO0, rel=0.1)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_sph.py -v -k density
```

Expected: FAIL (no `Solver` class)

- [ ] **Step 3: Implement `Solver` and `compute_density`**

Append to `python-taichi/air_sph/sph.py`:
```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_sph.py -v -k density
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python-taichi/air_sph/sph.py python-taichi/tests/test_sph.py
git commit -m "Add SPH density summation"
```

---

## Task 5: Pressure (linear acoustic equation of state)

**Files:**
- Modify: `python-taichi/air_sph/sph.py`
- Modify: `python-taichi/tests/test_sph.py`

**Interfaces:**
- Consumes: `Solver.rho` from Task 4.
- Produces: `solver.compute_pressure()`, filling `solver.pressure[i] = C0**2 * (rho[i] - RHO0)`.

- [ ] **Step 1: Write the failing pressure test**

Append to `python-taichi/tests/test_sph.py`:
```python
def test_pressure_matches_linear_eos():
    solver, grid, pos_np, half_extent, h = _make_lattice_solver(n_per_axis=4, dx=0.02)
    n = pos_np.shape[0]
    rho_np = sph.RHO0 + np.linspace(-0.05, 0.05, n).astype(np.float32)
    solver.rho.from_numpy(rho_np)

    solver.compute_pressure()

    expected = sph.C0 ** 2 * (rho_np - sph.RHO0)
    np.testing.assert_allclose(solver.pressure.to_numpy(), expected, rtol=1e-5)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_sph.py -v -k pressure
```

Expected: FAIL (no `compute_pressure`)

- [ ] **Step 3: Implement `compute_pressure`**

Append inside the `Solver` class in `python-taichi/air_sph/sph.py`:
```python
    @ti.kernel
    def compute_pressure(self):
        for i in range(self.n):
            self.pressure[i] = C0 ** 2 * (self.rho[i] - RHO0)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_sph.py -v -k pressure
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python-taichi/air_sph/sph.py python-taichi/tests/test_sph.py
git commit -m "Add linear acoustic equation of state"
```

---

## Task 6: Pressure-gradient force and artificial viscosity

**Files:**
- Modify: `python-taichi/air_sph/sph.py`
- Modify: `python-taichi/tests/test_sph.py`

**Interfaces:**
- Consumes: `Solver.rho`, `Solver.pressure`, `Solver.vel`, `kernel_grad_w`.
- Produces: `solver.compute_forces()`, filling `solver.acc[i]` with the
  (per-unit-mass) SPH pressure-gradient force plus Monaghan artificial
  viscosity.

- [ ] **Step 1: Write the failing force tests**

Append to `python-taichi/tests/test_sph.py`:
```python
def _two_particle_solver(h, dx, extra_grid_margin=0.5):
    mass = sph.RHO0 * dx ** 3
    pos_np = np.array([[0.0, 0.0, 0.0], [dx, 0.0, 0.0]], dtype=np.float32)
    n = 2
    cell_size = 2 * h
    grid = Grid(n_cells=8, cell_size=cell_size, grid_min=(-extra_grid_margin,) * 3, max_particles=n)
    solver = Solver(n, h, mass, grid)
    solver.pos.from_numpy(pos_np)
    grid.clear()
    grid.build(solver.pos, n)
    return solver, grid


def test_pressure_force_obeys_newtons_third_law():
    h, dx = 0.05, 0.03
    solver, grid = _two_particle_solver(h, dx)

    solver.compute_density()
    rho = solver.rho.to_numpy()
    rho[0] *= 1.2  # perturb so pressure differs between the two particles
    solver.rho.from_numpy(rho)
    solver.compute_pressure()
    solver.compute_forces()

    acc = solver.acc.to_numpy()
    np.testing.assert_allclose(solver.mass * acc[0], -solver.mass * acc[1], rtol=1e-5, atol=1e-8)


def test_viscosity_damps_approaching_particles():
    h, dx = 0.05, 0.03
    solver, grid = _two_particle_solver(h, dx)
    solver.vel.from_numpy(np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32))

    solver.compute_density()
    solver.compute_pressure()
    solver.compute_forces()

    acc = solver.acc.to_numpy()
    # particle 0 (at x=0) moving toward particle 1 (at x=dx) should decelerate;
    # particle 1 moving toward particle 0 should likewise decelerate.
    assert acc[0][0] < 0.0
    assert acc[1][0] > 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_sph.py -v -k "third_law or damps"
```

Expected: FAIL (no `compute_forces`)

- [ ] **Step 3: Implement `compute_forces`**

Append inside the `Solver` class in `python-taichi/air_sph/sph.py`:
```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_sph.py -v -k "third_law or damps"
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python-taichi/air_sph/sph.py python-taichi/tests/test_sph.py
git commit -m "Add SPH pressure-gradient force and artificial viscosity"
```

---

## Task 7: Leapfrog time integration

**Files:**
- Modify: `python-taichi/air_sph/sph.py`
- Modify: `python-taichi/tests/test_sph.py`

**Interfaces:**
- Consumes: `Solver.pos`, `Solver.vel`, `Solver.acc`.
- Produces: `solver.leapfrog_predict(dt: ti.f32)` (half-kick + drift),
  `solver.leapfrog_correct(dt: ti.f32)` (half-kick). Calling `predict`,
  recomputing `acc`, then `correct` implements kick-drift-kick leapfrog.

- [ ] **Step 1: Write the failing integration test**

Append to `python-taichi/tests/test_sph.py`:
```python
def test_leapfrog_matches_constant_acceleration_kinematics():
    n = 1
    grid = Grid(n_cells=4, cell_size=1.0, grid_min=(-2.0, -2.0, -2.0), max_particles=n)
    solver = Solver(n, h=0.1, mass=1.0, grid=grid)
    solver.pos.from_numpy(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    solver.vel.from_numpy(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    solver.acc.from_numpy(np.array([[0.0, -9.8, 0.0]], dtype=np.float32))

    dt = 0.01
    n_steps = 50
    for _ in range(n_steps):
        solver.leapfrog_predict(dt)
        solver.leapfrog_correct(dt)

    t = dt * n_steps
    pos = solver.pos.to_numpy()[0]
    assert pos[0] == pytest.approx(1.0 * t, rel=1e-4)
    assert pos[1] == pytest.approx(0.5 * -9.8 * t ** 2, rel=1e-4)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_sph.py -v -k leapfrog
```

Expected: FAIL (no `leapfrog_predict`/`leapfrog_correct`)

- [ ] **Step 3: Implement leapfrog integration**

Append inside the `Solver` class in `python-taichi/air_sph/sph.py`:
```python
    @ti.kernel
    def leapfrog_predict(self, dt: ti.f32):
        for i in range(self.n):
            self.vel[i] += 0.5 * dt * self.acc[i]
            self.pos[i] += dt * self.vel[i]

    @ti.kernel
    def leapfrog_correct(self, dt: ti.f32):
        for i in range(self.n):
            self.vel[i] += 0.5 * dt * self.acc[i]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_sph.py -v -k leapfrog
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python-taichi/air_sph/sph.py python-taichi/tests/test_sph.py
git commit -m "Add leapfrog time integration"
```

---

## Task 8: Still-box stability system test

This is the spec's first system-level validation milestone: a uniform
lattice with no source driving should stay at rest.

**Files:**
- Create: `python-taichi/tests/test_stability.py`

**Interfaces:**
- Consumes: `Grid` (Task 3), `Solver` with `compute_density`,
  `compute_pressure`, `compute_forces`, `leapfrog_predict`, `leapfrog_correct`
  (Tasks 4-7).

- [ ] **Step 1: Write the failing stability test**

`python-taichi/tests/test_stability.py`:
```python
"""System-level validation: a quiescent particle lattice with no source
driving must stay near mechanical equilibrium (the standard SPH "still
box" sanity check) — see docs/superpowers/specs/2026-09-24-air-sph-acoustics-design.md.
"""
import numpy as np
import pytest
import taichi as ti

from air_sph import sph
from air_sph.grid import Grid
from air_sph.sph import Solver

ti.init(arch=ti.cpu)


def _build_lattice(n_per_axis, dx):
    coords = (np.arange(n_per_axis) - (n_per_axis - 1) / 2) * dx
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)


def test_still_box_remains_stable():
    n_per_axis, dx = 10, 0.02
    h = 1.3 * dx
    mass = sph.RHO0 * dx ** 3

    pos0 = _build_lattice(n_per_axis, dx)
    n = pos0.shape[0]
    half_extent = (n_per_axis - 1) * dx / 2
    cell_size = 2 * h
    n_cells = int(np.ceil((2 * half_extent + 4 * cell_size) / cell_size))
    grid_min = (-half_extent - 2 * cell_size,) * 3

    grid = Grid(n_cells, cell_size, grid_min, max_particles=n)
    solver = Solver(n, h, mass, grid)
    solver.pos.from_numpy(pos0)
    solver.vel.fill(0.0)
    solver.acc.fill(0.0)

    dt = 0.3 * h / sph.C0
    n_steps = 300
    for _ in range(n_steps):
        solver.leapfrog_predict(dt)
        grid.clear()
        grid.build(solver.pos, n)
        solver.compute_density()
        solver.compute_pressure()
        solver.compute_forces()
        solver.leapfrog_correct(dt)

    pos_final = solver.pos.to_numpy()
    displacement = np.linalg.norm(pos_final - pos0, axis=1)
    assert displacement.max() < 0.5 * dx

    rho = solver.rho.to_numpy()
    margin = 2 * h
    interior = np.all(np.abs(pos0) < (half_extent - margin), axis=1)
    assert interior.sum() > 0
    assert rho[interior] == pytest.approx(sph.RHO0, rel=0.1)
```

- [ ] **Step 2: Run the test to verify it fails or is trivially unreachable before this task**

```bash
cd python-taichi
.venv/bin/python -m pytest tests/test_stability.py -v
```

Expected at this point: this test should actually PASS already, since
Tasks 4-7 already implement everything it exercises — this task's "step"
is really the integration checkpoint. If it fails, that means a bug was
introduced in an earlier task; treat it as this task's TDD red state and
fix the root cause in the relevant module (`sph.py`) before proceeding.

- [ ] **Step 3: Fix any issues found, or confirm pass**

No new implementation code is expected here — this test exercises only
code already written in Tasks 2-7. If it's already green, proceed.

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_stability.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python-taichi/tests/test_stability.py
git commit -m "Add still-box stability validation test"
```

---

## Task 9: Sponge-layer open boundary

**Files:**
- Create: `python-taichi/air_sph/boundary.py`
- Create: `python-taichi/tests/test_boundary.py`

**Interfaces:**
- Produces: `apply_sponge(pos: ti.template(), vel: ti.template(), n: ti.i32, center: vector3, r_start: ti.f32, r_domain: ti.f32, damping_max: ti.f32, dt: ti.f32)`
  (`@ti.kernel`) — multiplicatively damps `vel[i]` for particles with
  `|pos[i] - center| > r_start`, ramping up to `damping_max` at `r_domain`.

- [ ] **Step 1: Write the failing boundary test**

`python-taichi/tests/test_boundary.py`:
```python
"""Unit test for the sponge-layer open boundary — see
docs/superpowers/specs/2026-09-24-air-sph-acoustics-design.md, "Open/absorbing boundary".
"""
import numpy as np
import pytest
import taichi as ti

from air_sph import boundary

ti.init(arch=ti.cpu)


def test_sponge_damps_outer_particle_more_than_inner():
    n = 2
    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=n)
    pos.from_numpy(np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32))
    vel.from_numpy(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))

    center = ti.Vector([0.0, 0.0, 0.0])
    r_start, r_domain, damping_max, dt = 0.5, 1.0, 50.0, 0.001

    for _ in range(20):
        boundary.apply_sponge(pos, vel, n, center, r_start, r_domain, damping_max, dt)

    v = vel.to_numpy()
    assert v[0][0] == pytest.approx(1.0)  # inner particle (r=0 < r_start): untouched
    assert v[1][0] < 1.0                  # outer particle (r=0.9 > r_start): damped
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_boundary.py -v
```

Expected: FAIL (no `boundary` module)

- [ ] **Step 3: Implement the sponge boundary**

`python-taichi/air_sph/boundary.py`:
```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_boundary.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python-taichi/air_sph/boundary.py python-taichi/tests/test_boundary.py
git commit -m "Add sponge-layer open boundary"
```

---

## Task 10: Monopole source driver

**Files:**
- Create: `python-taichi/air_sph/source.py`
- Create: `python-taichi/tests/test_source.py`

**Interfaces:**
- Produces: `apply_monopole(pos: ti.template(), vel: ti.template(), is_source: ti.template(), n: ti.i32, center: vector3, amplitude: ti.f32, freq: ti.f32, t: ti.f32)`
  (`@ti.kernel`) — for particles with `is_source[i] == 1`, sets
  `vel[i] = amplitude * sin(2*pi*freq*t) * (radial unit vector from center)`.
  Non-source particles are left untouched.

- [ ] **Step 1: Write the failing source test**

`python-taichi/tests/test_source.py`:
```python
"""Unit test for the prescribed-velocity monopole source — see
docs/superpowers/specs/2026-09-24-air-sph-acoustics-design.md, "Driving source".
"""
import numpy as np
import pytest
import taichi as ti

from air_sph import source

ti.init(arch=ti.cpu)


def test_monopole_sets_prescribed_velocity_on_source_particles_only():
    n = 2
    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=n)
    is_source = ti.field(dtype=ti.i32, shape=n)

    pos.from_numpy(np.array([[0.01, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32))
    vel.fill(0.0)
    is_source.from_numpy(np.array([1, 0], dtype=np.int32))

    center = ti.Vector([0.0, 0.0, 0.0])
    amplitude, freq, t = 2.0, 100.0, 0.0025  # quarter period at 100 Hz

    source.apply_monopole(pos, vel, is_source, n, center, amplitude, freq, t)

    v = vel.to_numpy()
    expected = amplitude * np.sin(2 * np.pi * freq * t)
    assert v[0][0] == pytest.approx(expected, rel=1e-4)
    assert v[0][1] == pytest.approx(0.0, abs=1e-6)
    assert v[1][0] == pytest.approx(0.0)  # non-source particle untouched
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_source.py -v
```

Expected: FAIL (no `source` module)

- [ ] **Step 3: Implement the monopole source**

`python-taichi/air_sph/source.py`:
```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_source.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python-taichi/air_sph/source.py python-taichi/tests/test_source.py
git commit -m "Add monopole source driver"
```

---

## Task 11: Monopole radiation validation system test

This is the spec's key validated deliverable: driven wave propagation
compared to analytic free-field monopole radiation (arrival time ~ r/c0,
amplitude ~ 1/r, frequency match).

**Files:**
- Create: `python-taichi/tests/test_monopole.py`

**Interfaces:**
- Consumes: `Grid`, `Solver` (Tasks 3-7), `boundary.apply_sponge` (Task 9),
  `source.apply_monopole` (Task 10).

- [ ] **Step 1: Write the failing validation test**

`python-taichi/tests/test_monopole.py`:
```python
"""System-level validation: a pulsing monopole source compared to the
analytic free-field monopole radiation solution — see
docs/superpowers/specs/2026-09-24-air-sph-acoustics-design.md, "Validation & testing".

Resolution here (particles-per-wavelength, particle count) is deliberately
lower than what air_sph.demo uses by default, to keep this test's runtime
reasonable; tolerances are correspondingly generous. Tighten both together
if resolution is increased.
"""
import numpy as np
import pytest
import taichi as ti

from air_sph import sph, boundary, source
from air_sph.grid import Grid
from air_sph.sph import Solver

ti.init(arch=ti.cpu)


def _build_lattice(n_per_axis, dx):
    coords = (np.arange(n_per_axis) - (n_per_axis - 1) / 2) * dx
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)


def _probe_pressure(pos_np, pressure_np, probe_pos, radius):
    d = np.linalg.norm(pos_np - np.array(probe_pos), axis=1)
    mask = d < radius
    if not np.any(mask):
        return 0.0
    return float(pressure_np[mask].mean())


def test_monopole_radiation_matches_analytic_free_field():
    freq = 1000.0
    ppw = 6  # particles per wavelength; coarse, for test speed
    n_per_axis = 20
    dx = (sph.C0 / freq) / ppw
    h = 1.3 * dx
    mass = sph.RHO0 * dx ** 3

    pos0 = _build_lattice(n_per_axis, dx)
    n = pos0.shape[0]
    half_extent = (n_per_axis - 1) * dx / 2
    cell_size = 2 * h
    n_cells = int(np.ceil((2 * half_extent + 4 * cell_size) / cell_size))
    grid_min = (-half_extent - 2 * cell_size,) * 3

    grid = Grid(n_cells, cell_size, grid_min, max_particles=n)
    solver = Solver(n, h, mass, grid)
    solver.pos.from_numpy(pos0)
    solver.vel.fill(0.0)
    solver.acc.fill(0.0)

    src_radius = 1.5 * dx
    is_source_np = (np.linalg.norm(pos0, axis=1) < src_radius).astype(np.int32)
    assert is_source_np.sum() > 0
    is_source = ti.field(dtype=ti.i32, shape=n)
    is_source.from_numpy(is_source_np)

    center = ti.Vector([0.0, 0.0, 0.0])
    amplitude = 0.05  # m/s, small perturbation velocity for the source cluster
    r_start = 0.7 * half_extent
    r_domain = half_extent
    damping_max = 200.0

    probe_radii = [0.15, 0.25, 0.35]
    probe_sample_radius = 1.5 * dx
    probe_dir = np.array([1.0, 0.0, 0.0])
    histories = {r: [] for r in probe_radii}
    times = []

    dt = 0.3 * h / sph.C0
    n_steps = 400
    t = 0.0
    for _ in range(n_steps):
        solver.leapfrog_predict(dt)
        grid.clear()
        grid.build(solver.pos, n)
        solver.compute_density()
        solver.compute_pressure()
        solver.compute_forces()
        solver.leapfrog_correct(dt)
        t += dt
        source.apply_monopole(solver.pos, solver.vel, is_source, n, center, amplitude, freq, t)
        boundary.apply_sponge(solver.pos, solver.vel, n, center, r_start, r_domain, damping_max, dt)

        pos_np = solver.pos.to_numpy()
        pressure_np = solver.pressure.to_numpy()
        times.append(t)
        for r in probe_radii:
            histories[r].append(_probe_pressure(pos_np, pressure_np, r * probe_dir, probe_sample_radius))

    times = np.array(times)
    for r in probe_radii:
        histories[r] = np.array(histories[r])

    # --- arrival time: first probe sample exceeding a small detection threshold ---
    threshold = 0.02  # Pa, well above numerical noise floor
    arrival = {}
    for r in probe_radii:
        idx = np.argmax(np.abs(histories[r]) > threshold)
        assert idx > 0, f"probe at r={r} never exceeded the detection threshold"
        arrival[r] = times[idx]
    for r in probe_radii:
        assert arrival[r] == pytest.approx(r / sph.C0, rel=0.25)

    # --- amplitude falloff: peak amplitude in a steady-state window, should scale as 1/r ---
    steady_start = int(0.6 * n_steps)
    amp = {r: float(np.max(np.abs(histories[r][steady_start:]))) for r in probe_radii}
    for r in probe_radii:
        assert amp[r] > 0.0
    r_near, r_far = probe_radii[0], probe_radii[-1]
    ratio_measured = amp[r_near] / amp[r_far]
    ratio_expected = r_far / r_near
    assert ratio_measured == pytest.approx(ratio_expected, rel=0.3)

    # --- frequency: zero-crossing period at the nearest probe's steady window ---
    signal = histories[r_near][steady_start:]
    signal_t = times[steady_start:]
    crossings = np.where(np.diff(np.sign(signal)) > 0)[0]
    assert len(crossings) >= 2, "not enough zero crossings to estimate frequency"
    periods = np.diff(signal_t[crossings])
    measured_freq = 1.0 / np.mean(periods)
    assert measured_freq == pytest.approx(freq, rel=0.2)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd python-taichi
.venv/bin/python -m pytest tests/test_monopole.py -v
```

Expected: FAIL initially only if wiring is wrong (module imports should
already succeed given Tasks 3, 7, 9, 10) — this is the integration
checkpoint for the whole solver. If it fails on a physics assertion, debug
via `systematic-debugging`: instrument `histories` to a CSV/plot rather
than guessing, and check the simpler system tests (Task 8, Task 2-7 unit
tests) still pass first.

- [ ] **Step 3: Fix any issues found, or confirm pass**

No new production module is introduced here — this test exercises
`sph.py`, `grid.py`, `boundary.py`, and `source.py` together for the first
time. Fix root causes in those modules if this fails; do not loosen the
tolerances without first confirming (e.g. via a quick plot of `histories`)
that the physics is qualitatively right and only the tolerance was too
tight.

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_monopole.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python-taichi/tests/test_monopole.py
git commit -m "Add monopole radiation validation test"
```

---

## Task 12: Real-time viewer, CLI, and README

**Files:**
- Create: `python-taichi/air_sph/viewer.py`
- Create: `python-taichi/air_sph/demo.py`
- Modify: `python-taichi/README.md`

**Interfaces:**
- Consumes: everything from Tasks 2-11.
- Produces: `python -m air_sph.demo` (interactive, default) and
  `python -m air_sph.demo --offline --steps N` (headless, prints probe
  pressure history) entry points.

- [ ] **Step 1: Implement the viewer**

`python-taichi/air_sph/viewer.py`:
```python
"""Real-time Taichi GGUI viewer for the particle field, colored by
pressure perturbation."""
import taichi as ti


class Viewer:
    def __init__(self, window_name="Air Acoustics", res=(960, 720)):
        self.window = ti.ui.Window(window_name, res, vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(1.2, 0.9, 1.2)
        self.camera.lookat(0.0, 0.0, 0.0)

    @property
    def running(self):
        return self.window.running

    def render(self, pos_field, radius, colors_field):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.6, 0.6, 0.6))
        self.scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
        self.scene.particles(pos_field, radius=radius, per_vertex_color=colors_field)
        self.canvas.scene(self.scene)
        self.window.show()
```

- [ ] **Step 2: Implement the CLI/demo, tying the solver, source, boundary, and viewer together**

`python-taichi/air_sph/demo.py`:
```python
"""CLI entry point: builds the particle lattice, wires the solver, source,
boundary, and (optionally) the real-time viewer into one simulation loop.

Usage:
    python -m air_sph.demo                    # interactive viewer
    python -m air_sph.demo --offline --steps 200
"""
import argparse

import numpy as np
import taichi as ti

from air_sph import boundary, sph, source
from air_sph.grid import Grid
from air_sph.sph import Solver
from air_sph.viewer import Viewer


def build_lattice(n_per_axis, dx):
    coords = (np.arange(n_per_axis) - (n_per_axis - 1) / 2) * dx
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)


def probe_pressure(pos_np, pressure_np, probe_pos, radius):
    d = np.linalg.norm(pos_np - np.array(probe_pos), axis=1)
    mask = d < radius
    if not np.any(mask):
        return 0.0
    return float(pressure_np[mask].mean())


@ti.kernel
def update_colors(pressure: ti.template(), colors: ti.template(), n: ti.i32, scale: ti.f32):
    for i in range(n):
        p = pressure[i] / scale
        p = min(max(p, -1.0), 1.0)
        if p >= 0.0:
            colors[i] = ti.Vector([0.9, 0.9 - 0.7 * p, 0.9 - 0.7 * p])
        else:
            colors[i] = ti.Vector([0.9 + 0.7 * p, 0.9 + 0.7 * p, 0.9])


def build_sim(freq, ppw, n_per_axis):
    dx = (sph.C0 / freq) / ppw
    h = 1.3 * dx
    mass = sph.RHO0 * dx ** 3

    pos0 = build_lattice(n_per_axis, dx)
    n = pos0.shape[0]
    half_extent = (n_per_axis - 1) * dx / 2
    cell_size = 2 * h
    n_cells = int(np.ceil((2 * half_extent + 4 * cell_size) / cell_size))
    grid_min = (-half_extent - 2 * cell_size,) * 3

    grid = Grid(n_cells, cell_size, grid_min, max_particles=n)
    solver = Solver(n, h, mass, grid)
    solver.pos.from_numpy(pos0)
    solver.vel.fill(0.0)
    solver.acc.fill(0.0)

    src_radius = 1.5 * dx
    is_source_np = (np.linalg.norm(pos0, axis=1) < src_radius).astype(np.int32)
    is_source = ti.field(dtype=ti.i32, shape=n)
    is_source.from_numpy(is_source_np)

    return solver, grid, is_source, dx, half_extent


def step(solver, grid, is_source, dt, t, freq, amplitude, center, r_start, r_domain, damping_max):
    solver.leapfrog_predict(dt)
    grid.clear()
    grid.build(solver.pos, solver.n)
    solver.compute_density()
    solver.compute_pressure()
    solver.compute_forces()
    solver.leapfrog_correct(dt)
    source.apply_monopole(solver.pos, solver.vel, is_source, solver.n, center, amplitude, freq, t)
    boundary.apply_sponge(solver.pos, solver.vel, solver.n, center, r_start, r_domain, damping_max, dt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="run headless, no viewer")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--freq", type=float, default=1000.0)
    parser.add_argument("--ppw", type=int, default=10, help="particles per wavelength")
    parser.add_argument("--n", type=int, default=35, help="particles per axis")
    args = parser.parse_args()

    ti.init(arch=ti.gpu if not args.offline else ti.cpu)

    solver, grid, is_source, dx, half_extent = build_sim(args.freq, args.ppw, args.n)
    center = ti.Vector([0.0, 0.0, 0.0])
    h = solver.h
    dt = 0.3 * h / sph.C0
    amplitude = 0.05
    r_start = 0.7 * half_extent
    r_domain = half_extent

    colors = ti.Vector.field(3, dtype=ti.f32, shape=solver.n) if not args.offline else None
    viewer = Viewer() if not args.offline else None

    t = 0.0
    probe_pos = (0.2, 0.0, 0.0)
    for i in range(args.steps):
        step(solver, grid, is_source, dt, t, args.freq, amplitude, center, r_start, r_domain, r_domain and 200.0)
        t += dt

        if args.offline:
            if i % 20 == 0:
                pos_np = solver.pos.to_numpy()
                pressure_np = solver.pressure.to_numpy()
                p = probe_pressure(pos_np, pressure_np, probe_pos, 1.5 * dx)
                print(f"t={t:.5f}s probe(0.2,0,0)={p:.4f} Pa")
        else:
            if not viewer.running:
                break
            update_colors(solver.pressure, colors, solver.n, 1.0)
            viewer.render(solver.pos, radius=0.3 * dx, colors_field=colors)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Update the README with the finished usage instructions**

The README written in Task 1 already documents `python -m air_sph.demo`
and `--offline`; confirm it still matches these flags (it does — no
change needed unless flag names drifted during implementation, in which
case update the README to match).

- [ ] **Step 4: Run the full test suite, then manually verify the demo**

```bash
cd python-taichi
.venv/bin/python -m pytest -q
.venv/bin/python -m air_sph.demo --offline --steps 100
```

Expected: all tests pass; the offline run prints a probe pressure history
without crashing.

Then manually verify the interactive path (cannot be automated — GUI
window):
```bash
.venv/bin/python -m air_sph.demo
```
Confirm: a window opens showing a 3D particle cloud, particles near the
center visibly oscillate/change color as the source drives them, and a
wavefront-like disturbance is visible spreading outward over the first
second or so. Close the window when done. If the window fails to open or
particles don't visibly respond, that's a bug to fix before considering
this task done — this manual check is the only verification of the
viewer's actual rendering output.

- [ ] **Step 5: Commit**

```bash
git add python-taichi/air_sph/viewer.py python-taichi/air_sph/demo.py python-taichi/README.md
git commit -m "Add real-time viewer and CLI demo"
```

---

## Plan Self-Review Notes

- **Spec coverage:** linear acoustic EOS (Task 5), cubic-spline kernel +
  Taichi-native grid (Tasks 2-3), monopole source + open boundary (Tasks
  9-10), both named validation tests — still-box (Task 8) and monopole
  radiation (Task 11) — real-time GGUI viewer (Task 12), directory layout
  matches the spec exactly (`air_sph/{sph,grid,source,boundary,viewer,demo}.py`,
  `tests/{test_stability,test_monopole}.py` plus the additional unit-test
  files this TDD breakdown needed). No BEM coupling anywhere, per Non-goals.
- **Type/interface consistency:** `Grid(n_cells, cell_size, grid_min, max_particles, max_per_cell=128)`
  and `Solver(n, h, mass, grid)` constructor signatures are identical
  everywhere they're used, across Tasks 3-12. `apply_sponge` and
  `apply_monopole` signatures match between their defining task and their
  use in Task 11's system test and Task 12's `demo.py`.
- **Deviation from the spec's example numbers, flagged explicitly:** the
  spec's Performance Scope section suggested "~100-200 Hz, ~0.5-1m domain"
  as a particle-count estimate. Working through the actual constraint (the
  domain must span multiple wavelengths for probes to sit in the acoustic
  far field, or the monopole validation test is physically meaningless)
  shows those two numbers can't both hold at once — a 150 Hz wave has a
  2.3m wavelength, larger than a 1m domain. Task 11 and Task 12's defaults
  instead use 1000 Hz with a domain sized to fit ~3+ wavelengths, which
  keeps the far-field validation valid and lands at a comparable particle
  count (tens of thousands) to what the spec estimated. This is a
  parameter-level correction within the approved architecture, not a scope
  change.
