"""Thiele-Small lumped-element models for sealed and vented enclosures.

Units are SI throughout (m, m^2, m^3, Hz). Voice-coil inductance and
cone breakup are ignored, so results are valid in the piston band only.
"""
from dataclasses import dataclass

import numpy as np

C_AIR = 343.0  # speed of sound, m/s


@dataclass(frozen=True)
class Driver:
    fs: float    # free-air resonance, Hz
    qts: float   # total Q
    qes: float   # electrical Q
    vas: float   # equivalent compliance volume, m^3
    sd: float    # effective cone area, m^2

    @property
    def radius(self) -> float:
        return float(np.sqrt(self.sd / np.pi))

    def spl_1w1m(self) -> float:
        """Reference-efficiency sensitivity in dB SPL @ 1 W / 1 m (half space)."""
        eta0 = 4 * np.pi**2 / C_AIR**3 * self.fs**3 * self.vas / self.qes
        return 112.1 + 10 * np.log10(eta0)


# Generic 6.5" woofer used for demos; not a specific commercial product.
EXAMPLE_DRIVER = Driver(fs=45.0, qts=0.38, qes=0.42, vas=22e-3, sd=132e-4)


def sealed_alignment(drv: Driver, vb: float) -> tuple[float, float]:
    """Return (fc, Qtc) of a sealed box of net volume ``vb``."""
    k = np.sqrt(1 + drv.vas / vb)
    return drv.fs * k, drv.qts * k


def sealed_response(drv: Driver, vb: float, f: np.ndarray) -> np.ndarray:
    """Complex 2nd-order high-pass transfer function of a sealed box."""
    fc, qtc = sealed_alignment(drv, vb)
    s = 1j * f / fc
    return s**2 / (s**2 + s / qtc + 1)


def vented_response(drv: Driver, vb: float, fb: float, f: np.ndarray,
                    ql: float = 7.0) -> np.ndarray:
    """Complex 4th-order transfer function of a vented box (Small, 1973).

    ``ql`` is the box leakage Q; 7 is a typical real-world value.
    """
    alpha = drv.vas / vb
    h = fb / drv.fs
    qt = drv.qts
    a1 = (ql + h * qt) / (np.sqrt(h) * ql * qt)
    a2 = (h + (alpha + 1 + h**2) * ql * qt) / (h * ql * qt)
    a3 = (h * ql + qt) / (np.sqrt(h) * ql * qt)
    s = 1j * f / np.sqrt(drv.fs * fb)  # normalised to T0 = 1/sqrt(ws*wb)
    return s**4 / (s**4 + a1 * s**3 + a2 * s**2 + a3 * s + 1)


def port_length(vb: float, fb: float, port_radius: float) -> float:
    """Physical length of a round port tuning ``vb`` to ``fb``.

    End correction assumes one flanged and one free end (1.463 r).
    """
    sv = np.pi * port_radius**2
    return C_AIR**2 * sv / ((2 * np.pi * fb) ** 2 * vb) - 1.463 * port_radius


def spl(drv: Driver, h: np.ndarray, watts: float = 1.0) -> np.ndarray:
    """On-axis SPL at 1 m for transfer function ``h`` and input power."""
    return drv.spl_1w1m() + 10 * np.log10(watts) + 20 * np.log10(np.abs(h))
