# acou3d: Python / PyVista approach

Speaker enclosure modelling with gmsh geometry, Thiele-Small models and PyVista visualisation.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Tested on Python 3.14. The pip `vtk` wheel bundles its own libraries, so the system VTK isn't used.

BEM assembly uses bempp-cl's OpenCL backend when `pyopencl` finds a device, which is
about 6.5x faster than the numba fallback on this machine (Intel OpenCL CPU runtime,
i5-7500). Both backends already use every core, so extra threads won't help.

## Run

```bash
.venv/bin/python -m acou3d_py.demo                  # interactive, vented box, frequency slider
.venv/bin/python -m acou3d_py.demo --box sealed
.venv/bin/python -m acou3d_py.demo --out renders/   # off-screen PNGs
.venv/bin/python -m acou3d_py.demo --bem            # BEM radiation, third octaves 100 Hz to 2 kHz
.venv/bin/python -m pytest -q
```

## Layout

| Module | What it does |
|---|---|
| `acou3d_py/ts.py` | Thiele-Small driver, sealed/vented transfer functions, port length, SPL |
| `acou3d_py/geometry.py` | Parametric cabinet in gmsh (driver recess, port bore), surface mesh with region tags |
| `acou3d_py/directivity.py` | Baffled-piston directivity balloon (analytic) and balloon meshing |
| `acou3d_py/bem.py` | bempp-cl exterior radiation (Burton-Miller), far field, cached frequency sweeps |
| `acou3d_py/demo.py` | CLI tying it together |

## BEM mode (`--bem`)

- The driver (the recess floor) vibrates at 1 m/s; the rest of the cabinet is rigid.
- The cabinet is coloured by surface pressure (dB relative to its maximum), and the balloon
  is the full-sphere far field normalised to on-axis.
- The chart adds "incl. baffle step": the Thiele-Small response plus the BEM on-axis gain
  relative to half-space radiation (-6 dB at low frequency, a diffraction bump near 1 kHz).
- The mesh is sized for 6 elements per wavelength at `--bem-fmax` (default 2 kHz).
  The first run solves about 14 frequencies in roughly 90 s. Results are cached in `.cache/`,
  keyed on the geometry and frequencies, so later runs start in about 2 s.
- It's validated in `tests/test_bem.py`: an interior point source is reproduced within 2%,
  and the low-frequency baffle gain is -6 dB within 0.3 dB.

## Limits / next steps

- **The port is modelled as rigid.** Driving the port's end with the Thiele-Small port
  velocity would add its contribution near fb.
- BEM stops at `--bem-fmax`. Higher frequencies need a finer mesh, and dense BEM cost
  grows with N^2. The analytic piston balloon (no `--bem`) still covers up to 20 kHz.
- The Thiele-Small model ignores voice-coil inductance and cone breakup.
- Interior standing waves and cabinet modes need FEM (scikit-fem or FEniCSx).
- `EXAMPLE_DRIVER` is a generic 6.5" woofer, not a real product.
