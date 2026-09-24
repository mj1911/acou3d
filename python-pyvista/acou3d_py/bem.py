"""Exterior acoustic radiation of the enclosure with bempp-cl.

Solves the Helmholtz exterior Neumann problem on the closed cabinet surface
with the Burton-Miller formulation (no spurious interior resonances):

    (K - I/2 - eta W) p = (V + eta (I/2 + K')) dp/dn,   eta = i/k

Pressure ``p`` is P1 (continuous), the normal derivative is DP0 so the
driver's velocity can jump to zero at its rim. Time convention e^{-iwt},
outgoing kernel e^{ikr}/(4 pi r), normals point out of the cabinet.

Rule of thumb: keep at least 6 elements per wavelength, i.e.
f <= c / (6 * mesh_size).
"""
import hashlib
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

os.environ.setdefault("BEMPP_LOG_LEVEL", "error")
import bempp_cl.api as bempp  # noqa: E402
from bempp_cl.api.operators.boundary import helmholtz, sparse  # noqa: E402
from bempp_cl.api.operators.far_field import helmholtz as far_field  # noqa: E402

from .geometry import REGION_DRIVER, Enclosure, build_mesh  # noqa: E402
from .ts import C_AIR  # noqa: E402

RHO_AIR = 1.2  # kg/m^3
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

warnings.filterwarnings("ignore", message="splu converted its input")


def _use_opencl_if_available() -> None:
    """OpenCL assembly is ~6x faster than numba on CPU; fall back silently."""
    try:
        import pyopencl
        if any(p.get_devices() for p in pyopencl.get_platforms()):
            bempp.DEFAULT_DEVICE_INTERFACE = "opencl"
    except Exception:
        pass


_use_opencl_if_available()


def max_frequency(mesh_size: float, elements_per_wavelength: float = 6) -> float:
    return C_AIR / (elements_per_wavelength * mesh_size)


def to_grid(mesh: pv.PolyData) -> bempp.Grid:
    """bempp grid from a closed triangle mesh; domain index = ``region`` tag.

    Triangles are re-wound so normals point outward, as bempp assumes.
    """
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False,
                                consistent_normals=True, auto_orient_normals=True,
                                split_vertices=False)
    tris = mesh.faces.reshape(-1, 4)[:, 1:]
    return bempp.Grid(mesh.points.T.astype(np.float64), tris.T.astype(np.uint32),
                      domain_indices=mesh.cell_data["region"].astype(np.uint32))


@dataclass
class Solution:
    freq: float
    grid: bempp.Grid
    pressure: "bempp.GridFunction"  # P1 surface pressure
    dpdn: "bempp.GridFunction"      # DP0 normal derivative (boundary data)

    @property
    def k(self) -> float:
        return 2 * np.pi * self.freq / C_AIR

    def surface(self) -> pv.PolyData:
        """Surface mesh with complex vertex pressure in ``"p"``."""
        v, e = self.grid.vertices, self.grid.elements
        faces = np.hstack([np.full((e.shape[1], 1), 3), e.T]).ravel()
        mesh = pv.PolyData(v.T.copy(), faces)
        mesh.point_data["p"] = self.pressure.evaluate_on_vertices()[0]
        return mesh

    def far_field(self, directions: np.ndarray) -> np.ndarray:
        """Complex far-field pattern for unit ``directions`` of shape (3, N)."""
        p1, dp0 = self.pressure.space, self.dpdn.space
        dlp = far_field.double_layer(p1, directions, self.k)
        slp = far_field.single_layer(dp0, directions, self.k)
        return (dlp * self.pressure - slp * self.dpdn)[0]


def dp0_space(grid: bempp.Grid):
    return bempp.function_space(grid, "DP", 0)


def solve(dpdn: "bempp.GridFunction", freq: float) -> Solution:
    """Solve for surface pressure given normal-derivative data on a DP0 space."""
    k = 2 * np.pi * freq / C_AIR
    dp0 = dpdn.space
    p1 = bempp.function_space(dp0.grid, "P", 1)
    eta = 1j / k
    dlp = helmholtz.double_layer(p1, p1, p1, k)
    hyp = helmholtz.hypersingular(p1, p1, p1, k)
    ident_p1 = sparse.identity(p1, p1, p1)
    slp = helmholtz.single_layer(dp0, p1, p1, k)
    adlp = helmholtz.adjoint_double_layer(dp0, p1, p1, k)
    ident_dp0 = sparse.identity(dp0, p1, p1)

    lhs = dlp - 0.5 * ident_p1 - eta * hyp
    rhs = (slp + eta * (0.5 * ident_dp0 + adlp)) * dpdn
    pressure = bempp.linalg.lu(lhs, rhs)
    return Solution(freq, dp0.grid, pressure, dpdn)


def solve_driver(grid: bempp.Grid, freq: float, velocity: float = 1.0) -> Solution:
    """Driver surface moves with normal ``velocity`` (m/s); everything else rigid."""
    value = 1j * 2 * np.pi * freq * RHO_AIR * velocity  # Euler: dp/dn = i w rho v_n
    driver = REGION_DRIVER

    @bempp.complex_callable
    def fun(x, n, domain_index, result):
        result[0] = value if domain_index == driver else 0.0

    return solve(bempp.GridFunction(dp0_space(grid), fun=fun), freq)


def sphere_directions(n_theta: int = 73, n_phi: int = 72):
    """Full-sphere (theta, phi) grids and unit directions (3, n_theta * n_phi).

    theta is measured from +z, the baffle normal.
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return th, ph, dirs.reshape(3, -1)


def halfspace_axis_farfield(freq: float, sd: float, velocity: float = 1.0) -> float:
    """|far-field| on axis of a baffled piston radiating into half space."""
    return 2 * np.pi * freq * RHO_AIR * velocity * sd / (2 * np.pi)


@dataclass
class Sweep:
    """Pre-solved results over a set of frequencies, cheap to browse."""
    freqs: np.ndarray
    surface: pv.PolyData      # point_data "p_<i>" = complex surface pressure
    theta: np.ndarray
    phi: np.ndarray
    patterns: np.ndarray      # (n_freq, n_theta, n_phi) complex far field
    baffle_gain: np.ndarray   # on-axis |far field| relative to half space

    def nearest(self, f: float) -> int:
        return int(np.argmin(np.abs(np.log(self.freqs / f))))


def third_octaves(f_lo: float, f_hi: float) -> np.ndarray:
    """ISO 266 nominal third-octave centres in [f_lo, f_hi]."""
    nominal = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                        400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                        5000, 6300, 8000, 10000, 12500, 16000, 20000])
    return nominal[(nominal >= f_lo) & (nominal <= f_hi)]


def driver_area(grid: bempp.Grid) -> float:
    """Area of the driver-tagged elements (the faceted disc, not pi r^2)."""
    return float(grid.volumes[grid.domain_indices == REGION_DRIVER].sum())


def sweep(enc: Enclosure, freqs, mesh_size: float,
          use_cache: bool = True, progress=print) -> Sweep:
    """Solve the driver radiation problem at each frequency (cached on disk)."""
    freqs = np.asarray(freqs, float)
    key = hashlib.sha1(json.dumps([asdict(enc), freqs.tolist(), mesh_size],
                                  default=str).encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"sweep-{key}.npz"
    th, ph, dirs = sphere_directions()
    axis = np.array([[0.0], [0.0], [1.0]])

    if use_cache and cache.exists():
        d = np.load(cache)
        return Sweep(freqs, _surface(d["points"], d["faces"], d["pressures"]),
                     th, ph, d["patterns"], d["baffle_gain"])

    grid = to_grid(build_mesh(enc, mesh_size=mesh_size))
    sd = driver_area(grid)
    progress(f"BEM: {grid.number_of_elements} elements, {len(freqs)} frequencies, "
             f"backend {bempp.DEFAULT_DEVICE_INTERFACE}")
    pressures, patterns, gain = [], [], []
    for i, f in enumerate(freqs):
        progress(f"  solving {f:.0f} Hz ({i + 1}/{len(freqs)})")
        sol = solve_driver(grid, f)
        pressures.append(sol.pressure.evaluate_on_vertices()[0])
        patterns.append(sol.far_field(dirs).reshape(th.shape))
        gain.append(abs(sol.far_field(axis)[0]) / halfspace_axis_farfield(f, sd))

    points, faces = grid.vertices.T.copy(), grid.elements.T.copy()
    pressures, patterns, gain = np.array(pressures), np.array(patterns), np.array(gain)
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cache, points=points, faces=faces, pressures=pressures,
                        patterns=patterns, baffle_gain=gain)
    return Sweep(freqs, _surface(points, faces, pressures), th, ph, patterns, gain)


def _surface(points: np.ndarray, faces: np.ndarray, pressures: np.ndarray) -> pv.PolyData:
    mesh = pv.PolyData(points, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    for i, p in enumerate(pressures):
        mesh.point_data[f"p_{i}"] = p
    return mesh
