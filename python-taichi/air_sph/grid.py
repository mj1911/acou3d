"""Uniform spatial grid for SPH neighbor search.

Cell size must equal the kernel support radius (2h) so that all neighbors
within the support radius of a particle lie in its own cell or the 26
adjacent cells (a 3x3x3 block) — see kernel_w/kernel_grad_w in sph.py.
"""
import numpy as np
import taichi as ti


@ti.data_oriented
class Grid:
    def __init__(self, n_cells, cell_size, grid_min, max_particles, max_per_cell=128):
        self.n_cells = n_cells
        self.cell_size = cell_size
        self.max_per_cell = max_per_cell
        self.grid_min_np = tuple(grid_min)
        self.grid_min = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.grid_min[None] = grid_min
        self.cell_count = ti.field(dtype=ti.i32, shape=(n_cells, n_cells, n_cells))
        self.cell_particles = ti.field(dtype=ti.i32, shape=(n_cells, n_cells, n_cells, max_per_cell))

    @ti.func
    def cell_coord(self, pos):
        rel = (pos - self.grid_min[None]) / self.cell_size
        c = ti.cast(ti.floor(rel), ti.i32)
        return ti.min(ti.max(c, 0), self.n_cells - 1)

    def cell_coord_np(self, pos_np):
        """Plain-Python equivalent of cell_coord, for use outside kernels (tests)."""
        rel = (np.asarray(pos_np) - np.asarray(self.grid_min_np)) / self.cell_size
        c = np.floor(rel).astype(np.int32)
        return np.clip(c, 0, self.n_cells - 1)

    @ti.kernel
    def clear(self):
        for i, j, k in self.cell_count:
            self.cell_count[i, j, k] = 0

    @ti.kernel
    def build(self, pos: ti.template(), n: ti.i32):
        for p in range(n):
            c = self.cell_coord(pos[p])
            slot = ti.atomic_add(self.cell_count[c[0], c[1], c[2]], 1)
            if slot < self.max_per_cell:
                self.cell_particles[c[0], c[1], c[2], slot] = p
