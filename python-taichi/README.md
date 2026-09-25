# acou3d: Python / Taichi approach

Particle-based (SPH) simulation of sound propagating through air, in 3D,
using Taichi for GPU-parallel kernels and a real-time viewer. Modeled
architecturally on [SebLague/Fluid-Sim](https://github.com/SebLague/Fluid-Sim)
(GPU SPH neighbor search + solver + viewer), but solving linear acoustic
wave propagation in a compressible gas instead of incompressible fluid flow.

See `docs/superpowers/specs/2026-09-24-air-sph-acoustics-design.md` (repo
root) for the full design.

## Setup

All commands below are run from this directory (`python-taichi/`), not the
repository root — the `air_sph` package is resolved relative to it.

```bash
cd python-taichi
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
cd python-taichi   # if you are not already here
.venv/bin/python -m air_sph.demo                  # interactive viewer
.venv/bin/python -m air_sph.demo --offline         # headless, prints probe data
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -n auto -q         # same, parallelized across CPU cores
```
