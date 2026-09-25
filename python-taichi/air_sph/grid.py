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
        # NOTE: `max_particles` is currently unused and vestigial -- it sizes
        # nothing here, because the per-cell particle lists are sized by
        # max_per_cell and the cell grid by n_cells. It is kept in the
        # signature deliberately, as API documentation: callers state the
        # particle count they intend to use with this grid, which makes the
        # intended capacity explicit at every construction site. It is not
        # load-bearing, so do not rely on it to bound anything.
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
            # Reproducibility note: this atomic_add races -- which slot a
            # particle lands in within its cell depends on the order the
            # parallel-for threads happen to reach this line, so the
            # per-cell particle ordering is not reproducible run to run.
            # compute_density/compute_forces then sum float32 neighbor
            # contributions in that varying order, and float32 addition is
            # not associative, so bit-identical inputs can yield
            # bit-different results across runs (amplified over many
            # integration steps).
            #
            # This is a reproducibility property, not a correctness bug: the
            # differences are last-bits rounding, not qualitative -- dynamics
            # stay bounded and physically equivalent. Independently verified
            # in this project's Task 11 investigation and again by the final
            # whole-branch review. Tests that need run-to-run stable
            # measurements must average over samples (as test_monopole.py's
            # 6-direction probe averaging does) rather than assume
            # bit-reproducibility.
            slot = ti.atomic_add(self.cell_count[c[0], c[1], c[2]], 1)
            if slot < self.max_per_cell:
                self.cell_particles[c[0], c[1], c[2], slot] = p

    def check_no_overflow(self):
        """Assert no cell received more particles than `max_per_cell`.

        `build()` silently drops particles past slot `max_per_cell` (the
        bounds check there has no else-branch), which would quietly shrink
        neighbor lists and corrupt density/force sums with no visible error.
        This makes that failure mode detectable.

        Deliberately NOT called from the per-step simulation loop: it needs a
        `.to_numpy()` device-to-host copy of the whole cell-count array, far
        too slow to do every step. Call it once after the initial `build()` --
        the lattice is densest and most uniform at t=0, and the small
        acoustic displacements this project models (verified max ~2e-4*dx)
        cannot push a cell over capacity later if it had headroom then.
        """
        peak = int(self.cell_count.to_numpy().max())
        assert peak <= self.max_per_cell, (
            f"cell_count reached {peak} > max_per_cell={self.max_per_cell} -- particles "
            "were silently dropped from the neighbor search; raise max_per_cell"
        )
