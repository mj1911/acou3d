"""Parametric enclosure geometry built with gmsh's OpenCASCADE kernel.

Coordinates: the box spans [0, W] x [0, H] x [0, D] with the front baffle
at z = D, facing +z. The solid is the *outer* shape of the cabinet; the
driver recess and port bore are cut into it so the surface mesh is usable
as a BEM radiating surface later on.
"""
from dataclasses import dataclass

import gmsh
import numpy as np
import pyvista as pv

REGION_CABINET, REGION_DRIVER, REGION_PORT = 0, 1, 2
DRIVER_RECESS = 0.005  # flange recess depth; its floor stands in for the cone


@dataclass(frozen=True)
class Enclosure:
    width: float
    height: float
    depth: float
    wall: float = 0.018
    driver_radius: float = 0.065
    driver_center: tuple[float, float] | None = None  # (x, y); default upper third
    port_radius: float | None = None                  # None -> sealed box
    port_length: float = 0.0
    port_center: tuple[float, float] | None = None    # default lower quarter

    @property
    def net_volume(self) -> float:
        """Internal air volume, ignoring driver and port displacement."""
        t2 = 2 * self.wall
        return (self.width - t2) * (self.height - t2) * (self.depth - t2)

    def _driver_xy(self):
        return self.driver_center or (self.width / 2, self.height * 0.65)

    def _port_xy(self):
        return self.port_center or (self.width / 2, self.height * 0.22)

    def validate(self) -> None:
        if self.port_radius is not None and self.port_length > self.depth - self.wall:
            raise ValueError(
                f"port length {self.port_length:.3f} m does not fit in depth "
                f"{self.depth:.3f} m; enlarge the box or shrink the port")


def build_mesh(enc: Enclosure, mesh_size: float = 0.01) -> pv.PolyData:
    """Mesh the enclosure surface; cell data ``"region"`` tags each triangle."""
    enc.validate()
    W, H, D = enc.width, enc.height, enc.depth
    recess = DRIVER_RECESS

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("enclosure")
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, W, H, D)
        dx, dy = enc._driver_xy()
        cuts = [(3, occ.addCylinder(dx, dy, D - recess, 0, 0, recess * 2,
                                    enc.driver_radius))]
        if enc.port_radius is not None:
            px, py = enc._port_xy()
            cuts.append((3, occ.addCylinder(px, py, D - enc.port_length, 0, 0,
                                            enc.port_length * 2, enc.port_radius)))
        occ.cut([(3, box)], cuts)
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.model.mesh.generate(2)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        index = {t: i for i, t in enumerate(tags)}
        points = coords.reshape(-1, 3)
        _, _, node_tags = gmsh.model.mesh.getElements(dim=2)
        tris = np.array([index[t] for t in node_tags[0]]).reshape(-1, 3)
    finally:
        gmsh.finalize()

    faces = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
    mesh = pv.PolyData(points, faces)
    mesh.cell_data["region"] = _classify(enc, mesh.cell_centers().points)
    return mesh


def _classify(enc: Enclosure, c: np.ndarray) -> np.ndarray:
    """Tag cells by centroid: driver = recess floor only, port = bore walls + end."""
    region = np.full(len(c), REGION_CABINET)
    eps = 1e-6
    z = c[:, 2]
    dx, dy = enc._driver_xy()
    in_driver = np.hypot(c[:, 0] - dx, c[:, 1] - dy) <= enc.driver_radius + eps
    on_floor = np.abs(z - (enc.depth - DRIVER_RECESS)) < eps
    region[in_driver & on_floor] = REGION_DRIVER
    if enc.port_radius is not None:
        px, py = enc._port_xy()
        in_port = np.hypot(c[:, 0] - px, c[:, 1] - py) <= enc.port_radius + eps
        in_bore = (z > enc.depth - enc.port_length - eps) & (z < enc.depth - eps)
        region[in_port & in_bore] = REGION_PORT
    return region
