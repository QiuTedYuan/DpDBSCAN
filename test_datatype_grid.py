import math

import numpy as np

from datatype_grid import GridSpace

grid_space = GridSpace(2, np.array([0., 0.]), np.array([9.9, 9.9]), math.sqrt(2), 1)


def test_num_grids():
    assert grid_space.width == 1.
    assert grid_space.num_grids == 100


def test_neighbor_offset():
    assert len(grid_space.neighbor_offsets) == 21


def test_is_valid():
    assert grid_space.is_valid((0, 0))
    assert not grid_space.is_valid((-1, 0))
    assert not grid_space.is_valid((10, 10))


def test_encode_decode():
    assert grid_space.encode_as_key((0, 1)) == 1
    assert grid_space.encode_as_key((1, 0)) == 10
    assert grid_space.decode_from_key(10) == (1, 0)
    assert grid_space.decode_from_key(1) == (0, 1)


def test_low_point():
    assert grid_space.get_low_point_of_grid((1, 1)).tolist() == [1., 1.]
    assert grid_space.get_grid_of_point(np.array([1.1, 1.2])) == (1, 1)


def test_dimension():
    gs = GridSpace(4, np.array([0., 0., -1., -5.]), np.array([3.9, 4.9, 5.9, -4.1]), 1, 1)
    assert gs.width == 0.5
    assert gs.grids_per_dim.tolist() == [8, 10, 14, 2]
