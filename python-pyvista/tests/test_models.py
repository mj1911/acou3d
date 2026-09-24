import numpy as np
import pytest

from acou3d_py import ts
from acou3d_py.directivity import piston_directivity
from acou3d_py.geometry import REGION_DRIVER, REGION_PORT, Enclosure, build_mesh

DRV = ts.EXAMPLE_DRIVER


def test_sealed_alignment_formula():
    # Vb = Vas -> compliance ratio 1 -> both scale by sqrt(2)
    fc, qtc = ts.sealed_alignment(DRV, DRV.vas)
    assert fc == pytest.approx(DRV.fs * np.sqrt(2))
    assert qtc == pytest.approx(DRV.qts * np.sqrt(2))


def test_sealed_response_at_fc_equals_qtc():
    fc, qtc = ts.sealed_alignment(DRV, 0.01)
    assert abs(ts.sealed_response(DRV, 0.01, np.array([fc]))[0]) == pytest.approx(qtc)


def test_responses_flat_in_passband_and_rolloff_slopes():
    hi = np.array([5000.0])
    assert abs(ts.sealed_response(DRV, 0.02, hi)[0]) == pytest.approx(1, abs=1e-3)
    assert abs(ts.vented_response(DRV, 0.02, 40, hi)[0]) == pytest.approx(1, abs=1e-3)
    # asymptotic slopes: 12 dB/oct sealed, 24 dB/oct vented
    lo = np.array([2.0, 4.0])
    s = 20 * np.log10(np.abs(ts.sealed_response(DRV, 0.02, lo)))
    v = 20 * np.log10(np.abs(ts.vented_response(DRV, 0.02, 40, lo)))
    assert s[1] - s[0] == pytest.approx(12.04, abs=0.3)
    assert v[1] - v[0] == pytest.approx(24.08, abs=0.6)


def test_port_length_matches_imperial_rule_of_thumb():
    # Lv[cm] = 23562.5 d^2 / (fb^2 Vb[L]) - 0.732 d, with c = 344 m/s
    d_cm, fb, vb_l = 5.0, 40.0, 30.0
    expected_cm = 23562.5 * d_cm**2 / (fb**2 * vb_l) - 0.732 * d_cm
    got_cm = ts.port_length(vb_l / 1000, fb, d_cm / 200) * 100
    assert got_cm == pytest.approx(expected_cm, rel=0.01)


def test_piston_directivity():
    theta = np.array([0.0, np.pi / 2])
    assert piston_directivity(theta, 100, 0.06)[0] == 1.0
    assert piston_directivity(theta, 100, 0.06)[1] > 0.99   # omni when ka << 1
    assert piston_directivity(theta, 10000, 0.06)[1] < 0.2  # beams when ka >> 1


def test_mesh_regions_present():
    enc = Enclosure(0.24, 0.40, 0.30, port_radius=0.025, port_length=0.12)
    mesh = build_mesh(enc, mesh_size=0.02)
    regions = set(np.unique(mesh.cell_data["region"]))
    assert {REGION_DRIVER, REGION_PORT} <= regions


def test_region_areas_match_geometry():
    r_d, r_p, l_p = 0.065, 0.025, 0.12
    enc = Enclosure(0.24, 0.40, 0.30, driver_radius=r_d, port_radius=r_p, port_length=l_p)
    mesh = build_mesh(enc, mesh_size=0.02)
    area = mesh.compute_cell_sizes()["Area"]
    region = mesh.cell_data["region"]
    assert area[region == REGION_DRIVER].sum() == pytest.approx(np.pi * r_d**2, rel=0.02)
    bore = 2 * np.pi * r_p * l_p + np.pi * r_p**2
    assert area[region == REGION_PORT].sum() == pytest.approx(bore, rel=0.05)  # faceting


def test_port_too_long_rejected():
    enc = Enclosure(0.24, 0.40, 0.30, port_radius=0.025, port_length=0.5)
    with pytest.raises(ValueError):
        build_mesh(enc)
