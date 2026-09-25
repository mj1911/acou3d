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

- Physically calibrated: real air density and sound speed (no tuned numerical
  sound speed), with wave propagation validated quantitatively against
  closed-form analytic acoustics rather than judged by eye. As built, that
  validation covers causal propagation at `c0` and the driven frequency;
  precise 1/r amplitude scaling is deferred — see "Validation & testing".
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

**Driving source**: a smooth radial body force centred on the domain,
`dv/dt = a(t) * (r_vec / sigma) * exp(-r^2 / (2 sigma^2))` with
`sigma = 1.5 h` (~2 particle spacings) — a monopole that no particle is
pinned to. The demo shapes `a(t)` into repeating Hann-windowed bursts
(3 cycles on, 10 off).

*Revised 2026-09-25.* The original source prescribed the velocity of the
~8 particles nearest the centre, `v(t) = A * sin(2*pi*f*t)`. Neighbouring
source particles then move in opposite directions, a pattern ~1 spacing long,
and measurement showed ~99% of the output went into SPH waves 2–3 spacings
long. Those lie past the kernel's dispersion ceiling (frequency peaks at
~c0 / (4.8 dx), where group speed falls to zero) on the branch where
frequency falls again, and carry energy at ~110–175 m/s: measured energy
speeds matched that branch's predicted group speed within ~7% at 300, 500
and 800 Hz. Real sound got ~1%. Making the pinned sphere larger did not help
(its hard edge is still lattice-scale). The smooth force has negligible
spatial content at lattice scale, and energy now travels at ~c0 in all
directions (see the propagation test below). `source.apply_monopole` is kept,
with its unit tests, but the demo no longer uses it.

**Open/absorbing boundary** *(revised 2026-09-25: now impedance-matched; see
below)*: a sponge shell near the domain edge that ramps
up artificial damping on particle velocity, so outgoing waves attenuate
rather than reflecting back into the domain — approximates free-field
(infinite space) so results are comparable to the analytic monopole
solution.

**Neighbor search**: a **dense fixed-capacity uniform grid** — a
`(n_cells, n_cells, n_cells)` cell-count array plus a
`(n_cells, n_cells, n_cells, max_per_cell)` particle-index array, rebuilt each
step — with **`cell_size = 2h`** (the full kernel support radius) paired with a
**3×3×3 neighbor stencil**.

This departs from the originally-planned sparse grid (dynamic SNode per cell,
`cell_size ~ h`), for two reasons found while building it:

- *Dense beats sparse here.* Sparsity pays off when particles cluster and most
  cells are empty. Air fills the domain at roughly uniform density, so nearly
  every cell is occupied and sparse bookkeeping buys almost nothing while
  costing indirection and code. Memory is bounded and small in practice — a few
  MB at typical particle counts — even though it is O(n_cells³ × max_per_cell)
  rather than proportional to the occupied-cell count.
- *`cell_size = 2h` beats `cell_size = h`.* With cells the size of the kernel
  support, every neighbor within `2h` is guaranteed to lie in the particle's own
  cell or one of the 26 adjacent ones, so a 3×3×3 = 27-cell stencil suffices.
  With `cell_size = h` the same guarantee needs a 5×5×5 = 125-cell stencil.
  Fewer, larger cells means fewer cell visits and less loop overhead, at the
  cost of scanning some particles beyond `2h` that are then rejected by the
  explicit distance test.

This still avoids porting the reference project's manual GPU
counting-sort/spatial-hash-offset scheme: same asymptotic cost, far less code,
idiomatic Taichi.

The fixed capacity is the one sharp edge: `build()` drops particles past
`max_per_cell` silently. `Grid.check_no_overflow()` exists to detect that and is
called once after the initial build in the demo and in the tests (not per step —
it needs a device-to-host copy).

## Components

| Module | Responsibility |
|---|---|
| `air_sph/sph.py` | Cubic-spline kernel, density summation, pressure (linear EOS), pressure-gradient + viscosity forces, leapfrog integration |
| `air_sph/grid.py` | Dense fixed-capacity uniform grid (`cell_size = 2h`); rebuilt each step; 27-cell (3×3×3) neighbor iteration; `check_no_overflow()` capacity sanity check |
| `air_sph/source.py` | Monopole drivers: smooth radial body force (used by the demo); legacy prescribed-velocity cluster |
| `air_sph/boundary.py` | Sponge-layer damping near domain edges |
| `air_sph/viewer.py` | Taichi GGUI real-time 3D particle view, colored by pressure perturbation |
| `air_sph/validation.py` | Independent 3D FDTD reference solver (pure numpy) driven by the same source; shared energy-arrival measurement; `python -m air_sph.validation` compares it with the SPH demo |
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
7. Record pressure at fixed probe points for validation/analysis. In the demo
   CLI this is available via `--offline` (which prints probe readings
   periodically); the validation tests record probe histories every step.
   Probe recording is not performed on the interactive viewer path, where the
   per-frame device-to-host copy would compete with the frame rate.

## Validation & testing

Mirrors `python-pyvista`'s pytest-based, quantitative-tolerance testing
style (e.g. `test_bem.py`'s "within 2%" checks):

1. **Stability / still-box test**: uniform-density particle lattice, no
   source driving. Particles should remain at rest with no spurious
   pressure or velocity growth. Standard SPH sanity check; catches
   kernel/EOS/integration bugs before any wave test is meaningful.
2. **Monopole radiation test**: pulsing source at domain center, probe points
   at several radii, each sampled by averaging over 6 symmetric directions.
   Validates that the scheme produces causally propagating acoustic waves:
   - **wave arrival time** at each probe matches `r / c0` (within 25%) —
     confirms the wave travels at the physical sound speed, in causal order
   - **measured frequency** matches the driven source frequency — the probe
     field's dominant spectral peak (by FFT, measured within ~3% of the drive)
     is asserted within a deliberately wide 30%, which confirms the source
     drives the field it is supposed to
   - **amplitude decay is monotonic in radius, and no slower than geometric
     (1/r) spreading** — geometric spreading is a genuine physical *lower*
     bound on the decay rate, since any additional dissipation can only make
     the field decay faster

3. **Sponge-effectiveness test**: the absorbing boundary must measurably
   absorb, not merely be wired in. Acoustic energy accumulating in the sponge
   shell is compared with the sponge enabled versus disabled, and must differ
   by a large factor (measured ~10×; asserted ≥3×). This exists because an
   earlier version used a damping coefficient ~100× too weak, leaving a
   technically-correct but completely inert boundary that no other test
   noticed.

4. **Propagation tests** (`test_propagation.py`) run the demo's own
   pipeline and defaults against the FDTD reference driven by the identical
   force, checking what the first-arrival test above cannot see:
   - *Burst matches the reference*: at 0.10–0.40 m, each probe's burst is
     cross-correlated with the reference's inside the drive window, so tails
     can't bias it. It requires correlation > 0.95, |lag| < 0.25 ms, and a speed
     from the lags within 0.9–1.1 c0. Measured: lags +0.06 to +0.09 ms,
     correlation ≥ 0.98. The old pinned-cluster source fails it (correlation
     down to 0.54, 189 m/s).
   - *Shell doesn't reflect*: under 6% of the energy at 0.20–0.40 m may arrive
     after the burst has passed. Measured 1.2–2.2%; 5–16% with the old
     velocity-only sponge.

   An earlier version timed energy centroids over the whole record. Post-burst
   tails biased it in both directions (375 m/s with the reflecting sponge,
   221 m/s with the matched one, for the same main burst), which is why it
   now compares waveforms.

**SPH amplitude excess (2026-09-25).** At 0.25–0.40 m the SPH burst is
13–23% louder than the FDTD reference driven by the same force. Near the
source (0.10 m) the two agree. Two effects combine:

- *Reflections* (a ±5% ripple). In the default domain, the SPH r·p rises
  toward the absorbing shell and dips just before it. In a doubled domain
  (shell at 0.95 m) this ripple disappears and r·p is flat, as a spherical
  wave's should be.
- *Dispersion* (the steady +13–15%). It remains in the doubled domain.
  For a force-driven source in a dispersive medium, the radiated
  far-field amplitude scales as c0 / v_g, where v_g is the group velocity:
  the same injected power builds up more amplitude when energy moves away
  more slowly. For SPH (kernel-smoothed density and pressure gradient,
  ω = c0 k sinc⁴(kh/2)), a residue calculation gives the gain as
  c0 / v_g × S(k_sph) / S(k0), where S is the source's spatial spectrum at the
  SPH and true wavenumbers. Near 560 Hz, v_g ≈ 0.894 c0. Predicted
  band-averaged gain 1.146–1.147; measured 1.134, 1.147, 1.158 at 0.30,
  0.35 and 0.40 m. Per frequency (450–650 Hz) it agrees within ~5%, the
  noise level of the short records.

This is a property of the discretisation, not a bug, and it vanishes with
resolution. Predicted excess at 560 Hz: +12% at 20 particles per wavelength
(the demo), +5% at 30 (216k particles for the same domain), +2.7% at 40
(~490k), +1.2% at 60 (~1.6M). Timing is unaffected: the burst still arrives
within ~0.1 ms of the reference.

**Impedance-matched absorbing shell (2026-09-25).** Damping only velocity
changes the shell's acoustic impedance, so outgoing waves partly reflect off
the damping gradient, and at 500 Hz the 0.2 m shell is under a third of a
wavelength thick. `boundary.relax_pressure` now also damps pressure at the
same rate, by relaxing each shell particle's rest density toward its current
density. With p and v damped equally, the outgoing and incoming wave
components stay decoupled, so there's no reflection at normal incidence for
any ramp (the same principle as the FDTD reference's sponge). The post-burst
tail fell from 5–16% to 1.2–2.2%.

   **Timing against the reference (2026-09-25).** Energy-arrival times for
   SPH trail the FDTD reference by a near-constant ~0.75 ms at 0.25–0.40 m.
   That is not a propagation delay. Cross-correlating the main burst alone
   gives a lag of only ~0.04–0.11 ms (default domain) and ~0.08 ms (with a
   waveform correlation of 0.995 in a doubled domain), consistent with SPH
   dispersion at these wavelengths. The energy centroid is pulled late by a
   post-burst tail that the reference doesn't have: 5–16% of the probe
   energy in the default domain, falling to 1–5% when the absorbing shell is
   0.4 m thick (as in the reference) instead of 0.2 m. So the tail is mostly
   reflection from the default shell, which is under a third of a wavelength
   thick at 500 Hz. The residual 1–5% tail and a ~12–16% higher SPH
   amplitude at 0.2–0.4 m were open questions. The tail was later fixed by the
   impedance-matched shell, and the amplitude was explained (see "SPH amplitude
   excess" below).

**Deferred (not delivered in v1): precise 1/r amplitude-scaling validation.**
The original target above — 1/r falloff "within a few percent, consistent with
the existing BEM validation bar" — is *not* met and is not asserted. At the
resolution these fast tests run (6 particles per wavelength), numerical
attenuation of the SPH discretization dominates geometric spreading; it is
roughly three orders of magnitude stronger than real air's physical
attenuation, so measured falloff is a mixture of spreading and scheme
dissipation rather than clean 1/r. The probe span usable at this domain size
(between the source's kernel-support exclusion zone and the sponge shell) spans
only ~23% expected amplitude change, smaller than the measurement scatter.
Rather than assert a tolerance loose enough to be vacuous, v1 asserts the
bounded-decay property above, which is physically meaningful and reliable.

Recipe for a future properly-resolved version, should quantitative 1/r
validation be wanted (this is a documented gap, not a silent one):

- probe radii spanning close to a decade, not ~23%, so the expected 1/r signal
  is far larger than measurement scatter
- **≥10–15 particles per wavelength**, the accuracy floor this spec already
  names under "Performance scope" — below it, scheme dissipation, not physics,
  sets the amplitude profile
- a verified-effective sponge boundary (see the sponge-effectiveness test), so
  the measurement is of a free field rather than a resonant box
- measurement after **~40+ driven periods**, once the startup transient has
  left the domain. This matters more than it sounds: measured at ~26 periods
  the probe's dominant frequency reads ~770 Hz for a 1000 Hz drive, versus
  ~985 Hz at ~58 periods — the "steady state" is not steady early on
- marked as a **slow/GPU-only test**, separate from the fast CI suite, since
  the particle count and step count needed put it well outside unit-test
  runtime

Note the tolerances above (25%, 30%) are deliberately looser than
`python-pyvista`'s BEM "within 2%" bar. They are set by measurement-estimator
scatter at this resolution, not by uncertainty about the physics, and they
remain sharp enough to be diagnostic: this plan's mutation testing confirmed
that a wrong sound speed, wrong EOS, sign error, or wrong driven frequency all
fail them by a wide margin.

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
    validation.py
  tests/
    conftest.py          # session-scoped ti.init fixture, shared by all tests
    test_setup.py
    test_sph.py
    test_grid.py
    test_source.py
    test_boundary.py
    test_stability.py
    test_monopole.py
    test_propagation.py
    test_validation.py   # the FDTD reference itself travels at c0
```

## Performance scope (v1)

SPH accuracy needs roughly 10–15 particles per wavelength. This constraint is
much cheaper at 1 kHz than first estimated here, so v1 targets 1 kHz directly.

The arithmetic: at 1 kHz the wavelength is ~0.343 m, so 10 particles per
wavelength means a particle spacing of ~0.034 m. A ~1 m domain is then ~29
particles per axis, i.e. **~29³ ≈ 24k particles** — tens of thousands, not
hundreds of thousands. (An earlier draft of this section claimed "hundreds of
thousands" and recommended backing off to ~100–200 Hz; that was an arithmetic
error, and the implementation correctly never followed the recommendation.)

**As built and validated, v1 runs at 1 kHz.** The demo CLI defaults to 1 kHz
with `--ppw 10 --n 35`, which is 42,875 particles in a ~1.17 m domain,
measured at **~0.7 s per 100 steps on CPU** (no GPU) — comfortable to iterate
on, and faster again on the GPU backend the viewer path selects. The timestep,
not the particle count, is the dominant cost driver: the acoustic CFL condition
`dt < C·h/c0` gives `dt ≈ 3.9e-5 s` here, so a millisecond of simulated time is
~26 steps.

*Revised 2026-09-25:* the demo now defaults to **500 Hz at 20 particles per
wavelength** (`--n 40`, 64,000 particles), the same 34 mm spacing and cost as
1 kHz at 10. Its dispersion ceiling is ~1.27 kHz, and a 1 kHz drive sits too
close to it: 1 kHz energy travels at only ~0.7 c0, and a burst's upper
spectrum reaches frequencies where energy barely moves. The resulting ringing
near the source was measured at 1,323 Hz, against a predicted ceiling of 1,270 Hz.
At 500 Hz, the burst's spectrum stays well below the ceiling. The cost is
wider rings (0.69 m wavelength in a 1.34 m domain).

Genuinely deferred: substantially larger domains, and the much higher
resolution/step count that quantitative 1/r amplitude validation would need
(see "Validation & testing"). Resolving the full audible bandwidth remains a
non-goal — the spacing requirement scales linearly with frequency, so 10 kHz
would cost ~1000× the particles of 1 kHz for the same domain.

## Open questions / future extensions (not in v1)

- Cross-validating against `python-pyvista`'s BEM far-field results by
  driving this sim with the same baffled-piston motion (deferred by design;
  see Non-goals).
- Higher-frequency / larger-domain performance work (GPU backend tuning,
  adaptive resolution, etc.).
- Reflecting/room boundary conditions (currently open/absorbing only).
