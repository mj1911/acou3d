"""Analytic directivity of a rigid circular piston in an infinite baffle.

This is a placeholder for a BEM solution (e.g. bempp-cl): it ignores
the finite baffle, so there is no edge diffraction and nothing behind it.
"""
import numpy as np
import pyvista as pv
from scipy.special import j1

from .ts import C_AIR


def piston_directivity(theta: np.ndarray, freq: float, radius: float) -> np.ndarray:
    """|D(theta)|, normalised to 1 on axis. ``theta`` is the off-axis angle."""
    x = 2 * np.pi * freq / C_AIR * radius * np.sin(theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(np.abs(x) < 1e-9, 1.0, 2 * j1(x) / x)
    return np.abs(d)


def balloon(freq: float, radius: float, floor_db: float = -40.0,
            n_theta: int = 91, n_phi: int = 180) -> pv.StructuredGrid:
    """Front-hemisphere directivity balloon, axis along +z.

    Point radius is (level_dB - floor_db) / -floor_db, so on-axis is 1 and
    the floor collapses to the origin. Level in dB is stored as ``"dB"``.
    """
    theta = np.linspace(0, np.pi / 2, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    db = 20 * np.log10(np.maximum(piston_directivity(th, freq, radius), 1e-12))
    return balloon_from_levels(th, ph, db, floor_db)


def balloon_from_levels(th: np.ndarray, ph: np.ndarray, db: np.ndarray,
                        floor_db: float = -40.0) -> pv.StructuredGrid:
    """Balloon surface from levels in dB (0 dB -> radius 1) on a (theta, phi) grid."""
    db = np.maximum(db, floor_db)
    r = (db - floor_db) / -floor_db
    grid = pv.StructuredGrid(r * np.sin(th) * np.cos(ph),
                             r * np.sin(th) * np.sin(ph),
                             r * np.cos(th))
    grid["dB"] = db.ravel(order="F")
    return grid
