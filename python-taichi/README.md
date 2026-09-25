# acou3d: Python / Taichi approach

Particle-based (SPH) simulation of sound propagating through air, in 3D,
using Taichi for GPU-parallel kernels and a real-time viewer. Modeled
architecturally on [SebLague/Fluid-Sim](https://github.com/SebLague/Fluid-Sim)
(GPU SPH neighbor search + solver + viewer), but solving linear acoustic
wave propagation in a compressible gas instead of incompressible fluid flow.

See `docs/superpowers/specs/2026-09-24-air-sph-acoustics-design.md` (repo
root) for the full design.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python -m air_sph.demo                  # interactive viewer
.venv/bin/python -m air_sph.demo --offline         # headless, prints probe data
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -n auto -q         # same, parallelized across CPU cores
```
