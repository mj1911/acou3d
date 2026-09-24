import numpy as np
import pytest

from acou3d_py import bem
from acou3d_py.geometry import Enclosure, build_mesh
from acou3d_py.ts import C_AIR

ENC = Enclosure(0.24, 0.40, 0.30, driver_radius=0.065)


@pytest.fixture(scope="module")
def grid():
    return bem.to_grid(build_mesh(ENC, mesh_size=0.04))


def test_interior_point_source_is_reproduced(grid):
    """Neumann data of a source inside the box must give back that source outside."""
    f, x0 = 600.0, np.array([0.1, 0.15, 0.12])
    k = 2 * np.pi * f / C_AIR
    s0, s1, s2 = x0

    @bem.bempp.complex_callable
    def dpdn(x, n, domain_index, result):
        d0, d1, d2 = x[0] - s0, x[1] - s1, x[2] - s2
        r = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        result[0] = (np.exp(1j * k * r) / (4 * np.pi * r) * (1j * k - 1 / r)
                     * (d0 * n[0] + d1 * n[1] + d2 * n[2]) / r)

    sol = bem.solve(bem.bempp.GridFunction(bem.dp0_space(grid), fun=dpdn), f)

    r = np.linalg.norm(grid.vertices.T - x0, axis=1)
    exact = np.exp(1j * k * r) / (4 * np.pi * r)
    surf = sol.pressure.evaluate_on_vertices()[0]
    assert np.linalg.norm(surf - exact) / np.linalg.norm(exact) < 0.02

    _, _, dirs = bem.sphere_directions(7, 8)
    ff_exact = np.exp(-1j * k * (dirs.T @ x0)) / (4 * np.pi)
    assert np.abs(sol.far_field(dirs) - ff_exact).max() < 0.02 * np.abs(ff_exact).max()


def test_baffle_step(grid):
    """On-axis: ~-6 dB vs half space at low frequency, ~0 dB once ka >> 1."""
    sd = bem.driver_area(grid)
    low, high = (bem.solve_driver(grid, f) for f in (60.0, 1400.0))
    axis = np.array([[0.0], [0.0], [1.0]])
    gain_db = [20 * np.log10(abs(s.far_field(axis)[0]) / bem.halfspace_axis_farfield(s.freq, sd))
               for s in (low, high)]
    assert gain_db[0] == pytest.approx(-6.0, abs=0.3)
    assert gain_db[1] == pytest.approx(0.0, abs=2.5)


def test_normals_point_outward(grid):
    # divergence theorem: sum of (centroid . normal * area) = 3 * volume > 0
    v, e = grid.vertices, grid.elements
    a, b, c = v[:, e[0]], v[:, e[1]], v[:, e[2]]
    signed = np.einsum("ij,ij->j", a, np.cross(b - a, c - a, axis=0)).sum() / 6
    assert signed > 0


def test_sweep_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(bem, "CACHE_DIR", tmp_path)
    args = (ENC, [200.0, 400.0], 0.05)
    first = bem.sweep(*args, progress=lambda _: None)
    second = bem.sweep(*args, progress=lambda _: pytest.fail("cache miss"))
    assert len(list(tmp_path.glob("sweep-*.npz"))) == 1
    np.testing.assert_allclose(second.patterns, first.patterns)
    np.testing.assert_allclose(second.baffle_gain, first.baffle_gain)
    np.testing.assert_allclose(second.surface.point_data["p_1"], first.surface.point_data["p_1"])
    assert np.iscomplexobj(second.surface.point_data["p_1"])
