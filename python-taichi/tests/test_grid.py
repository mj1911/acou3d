import numpy as np
import taichi as ti

from air_sph.grid import Grid

ti.init(arch=ti.cpu)


def test_build_places_each_particle_in_exactly_one_cell():
    n = 50
    rng = np.random.default_rng(0)
    pos_np = rng.uniform(-0.4, 0.4, size=(n, 3)).astype(np.float32)

    cell_size = 0.1
    n_cells = 10
    grid_min = (-0.5, -0.5, -0.5)
    grid = Grid(n_cells, cell_size, grid_min, max_particles=n)

    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    pos.from_numpy(pos_np)

    grid.clear()
    grid.build(pos, n)

    counts = grid.cell_count.to_numpy()
    assert counts.sum() == n
    assert counts.max() < grid.max_per_cell


def test_neighbors_of_a_pair_share_or_adjoin_a_cell():
    # two particles closer than cell_size must land in the same or adjacent cell
    cell_size = 0.1
    n_cells = 10
    grid_min = (-0.5, -0.5, -0.5)
    n = 2
    grid = Grid(n_cells, cell_size, grid_min, max_particles=n)

    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    pos.from_numpy(np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32))

    grid.clear()
    grid.build(pos, n)

    c0 = np.array(grid.cell_coord_np(pos.to_numpy()[0]))
    c1 = np.array(grid.cell_coord_np(pos.to_numpy()[1]))
    assert np.all(np.abs(c0 - c1) <= 1)
