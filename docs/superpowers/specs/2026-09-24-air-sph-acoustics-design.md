# Particle-based air acoustics (python-taichi) — Design

## Summary

A new, independent top-level subproject, `python-taichi/`, simulating sound
propagation through air as a 3D particle system — modeled architecturally on
[SebLague/Fluid-Sim](https://github.com/SebLague/Fluid-Sim) (GPU SPH neighbor
search + solver + real-time viewer) but solving a different physics problem:
linear acoustic wave propagation in a compressible gas, rather than an
incompressible/weakly-compressible liquid.

It has no code dependency on `python-pyvista/`. The two subprojects are
independent explorations of 3D acoustic modeling living side by side at the
repo root.

## Goals

- Physically calibrated: real air density and sound speed, validated against
  closed-form analytic acoustics (not just "looks plausible").
- Particle-based (SPH), not grid-based FDTD — matches the spirit of the
  reference project and the "air molecules" framing.
- Real-time interactive 3D viewer, in the same delivery as the validated
  solver — not a stretch goal.

## Non-goals (v1)

- No coupling to `python-pyvista`'s driver/cabinet/BEM model. Validation is
  against analytic free-field solutions only.
- No nonlinear/shock acoustics, no turbulence, no viscosity beyond what's
  needed for numerical stability.
- No attempt at literal per-molecule kinetics (Boltzmann/DSMC) — particles
  are SPH fluid parcels representing a continuum, not individual molecules.
- No requirement to resolve full audible bandwidth in v1 — see Performance
  Scope below.

## Physics model

**Governing equations**: SPH-discretized continuity + momentum equations for
a compressible gas, closed with a **linearized acoustic equation of state**:

```
p' = c0^2 * (rho - rho0)
```

with `rho0 = 1.204 kg/m^3` and `c0 = 343 m/s` (air at 20°C). This is the
linear acoustic wave equation written in SPH form: correct by construction
for small perturbations (which sound is), and it makes the SPH wave speed
equal to `c0` directly — required for the calibration goal.

This is a deliberate departure from the reference project's nonlinear Tait
EOS (used there to model an incompressible liquid with an artificially
lowered/tuned sound speed for numerical convenience). Here, sound speed is
the physical quantity under test, so it must be the real value, not a tuned
numerical parameter.

**Kernel**: standard cubic spline (Monaghan) smoothing kernel.

**Time integration**: leapfrog (kick-drift-kick), with timestep set by the
SPH acoustic CFL condition `dt < C * h / c0` (`C` ~0.3–0.4, `h` = smoothing
length). Because `c0` (343 m/s) is much faster than any liquid-SPH flow
speed, this is the dominant timestep constraint and the main performance
lever (see Performance Scope).

**Driving source**: a small cluster of particles at the domain center given
a prescribed oscillating radial velocity, `v(t) = A * sin(2*pi*f*t)` —
approximating a monopole point source.

**Open/absorbing boundary**: a sponge shell near the domain edge that ramps
up artificial damping on particle velocity, so outgoing waves attenuate
rather than reflecting back into the domain — approximates free-field
(infinite space) so results are comparable to the analytic monopole
solution.

**Neighbor search**: Taichi's native sparse grid (dynamic SNode per cell),
cell size ~ smoothing length `h`. Chosen over porting the reference
project's manual GPU counting-sort/spatial-hash-offset scheme: same
asymptotic cost, far less code, idiomatic Taichi, and doesn't pre-allocate
memory for empty cells (relevant here since air fills a large, roughly
uniform-density domain rather than clustering like a liquid surface).

## Components

| Module | Responsibility |
|---|---|
| `air_sph/sph.py` | Cubic-spline kernel, density summation, pressure (linear EOS), pressure-gradient + viscosity forces, leapfrog integration |
| `air_sph/grid.py` | Taichi sparse uniform grid; rebuilt each step; 27-cell neighbor iteration |
| `air_sph/source.py` | Monopole driver: prescribes velocity on a small particle cluster |
| `air_sph/boundary.py` | Sponge-layer damping near domain edges |
| `air_sph/viewer.py` | Taichi GGUI real-time 3D particle view, colored by pressure perturbation |
| `air_sph/demo.py` | CLI: builds particle lattice, wires solver/source/boundary/viewer together; supports headless/offline mode |

## Data flow (per simulation step)

1. Rebuild spatial grid from current particle positions.
2. SPH kernels: density summation → pressure (linear EOS) → pressure-gradient
   and viscosity forces.
3. Leapfrog integration (velocity half-step, position update, velocity
   half-step).
4. Apply prescribed source velocity to driver particles.
5. Apply sponge-layer damping to particles in the boundary shell.
6. If interactive: push particle positions/pressure to GGUI viewer.
7. Always: record pressure at fixed probe points for validation/analysis.

## Validation & testing

Mirrors `python-pyvista`'s pytest-based, quantitative-tolerance testing
style (e.g. `test_bem.py`'s "within 2%" checks):

1. **Stability / still-box test**: uniform-density particle lattice, no
   source driving. Particles should remain at rest with no spurious
   pressure or velocity growth. Standard SPH sanity check; catches
   kernel/EOS/integration bugs before any wave test is meaningful.
2. **Monopole radiation test**: pulsing source at domain center, probe
   points at several radii along one axis. Checks:
   - wave arrival time at each probe matches `r / c0`
   - pressure amplitude falls off as `1/r`
   - measured frequency matches the driven source frequency
   - all within a small quantitative tolerance (target: a few percent,
     consistent with the existing BEM validation bar)

## Directory layout

```
python-taichi/
  README.md
  requirements.txt
  air_sph/
    __init__.py
    sph.py
    grid.py
    source.py
    boundary.py
    viewer.py
    demo.py
  tests/
    test_stability.py
    test_monopole.py
```

## Performance scope (v1)

SPH accuracy needs roughly 10–15 particles per wavelength. At audible
frequencies (e.g. 1 kHz, wavelength ~0.34 m) in a meter-scale domain, that
implies hundreds of thousands of particles — likely too slow for a
real-time viewer on this machine's hardware (per `python-pyvista/README.md`,
an Intel OpenCL CPU runtime on an i5-7500, no dedicated GPU noted).

**v1 targets a low frequency (~100–200 Hz) and a modest domain (~0.5–1 m)**,
keeping particle counts in the tens of thousands — real-time-viewable and
fast to iterate on. Scaling to higher frequencies or larger domains is
explicitly deferred to a later iteration.

## Open questions / future extensions (not in v1)

- Cross-validating against `python-pyvista`'s BEM far-field results by
  driving this sim with the same baffled-piston motion (deferred by design;
  see Non-goals).
- Higher-frequency / larger-domain performance work (GPU backend tuning,
  adaptive resolution, etc.).
- Reflecting/room boundary conditions (currently open/absorbing only).
