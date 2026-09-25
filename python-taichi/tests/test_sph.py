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
