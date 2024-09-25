import math
from functools import lru_cache
from itertools import product

import numpy as np

from datatype_point import Point

GridCoord = tuple[int, ...]
GridKey = np.int64


# Denotes a d-dim grid space over [low, high]
class GridSpace:
    def __init__(self, dim: int, low: Point, high: Point, alpha: float, grid_scale: float):
        self.dim = dim
        self.low = low
        self.high = high
        self.alpha = alpha
        self.grid_scale = grid_scale
        self.width = grid_scale * alpha / math.sqrt(dim)
        self.grids_per_dim = tuple(np.add(np.floor_divide(np.subtract(high, low), self.width), 1).astype(dtype=int))
        self.num_grids: int = np.prod(self.grids_per_dim)
        self.neighbor_offsets = GridSpace.__neighbor_offsets(self.dim, radius=self.alpha / self.width)

        self.encode_as_key = lru_cache(maxsize=None)(self.__encode_as_key)
        self.decode_from_key = lru_cache(maxsize=None)(self.__decode_from_key)

    def is_valid(self, coord: GridCoord) -> bool:
        return all(0 <= c < bound for c, bound in zip(coord, self.grids_per_dim))

    def __decode_from_key(self, key: GridKey) -> GridCoord:
        coord = [0] * self.dim
        for i in range(self.dim).__reversed__():
            dim_max = self.grids_per_dim[i]
            coord[i] = key % dim_max
            key = key // dim_max
        return tuple(coord)

    def __encode_as_key(self, g: GridCoord) -> GridKey:
        key: GridKey = np.int64(0)
        for coord, dim_max in zip(g, self.grids_per_dim):
            key *= dim_max
            key += coord
        return key

    # convert point location to grid
    # for each dimension, the coordinate for location x is (x - low) // grid_width
    def get_grid_of_point(self, p: Point) -> GridCoord:
        return tuple(np.floor_divide(np.subtract(p, self.low), self.width).astype(dtype=int))

    def get_low_point_of_grid(self, coord: GridCoord) -> Point:
        return np.add(np.multiply(coord, self.width), self.low)

    # Given a d-dim radius 1 grid, compute its min distance with respect to the grid vec(0)
    @staticmethod
    @lru_cache(maxsize=None)
    def dist(coord: GridCoord) -> float:
        # compute the lower/upper bounding points
        sum_square: float = 0
        for c in coord:
            # dimensional difference between [0, 1) and [c, c+1)
            if c + 1 < 0:
                sum_square += 1. * (c + 1) * (c + 1)
            elif c > 1:
                sum_square += 1. * (c - 1) * (c - 1)
            else:
                sum_square += 0.
        return math.sqrt(sum_square)

    # Given a d-dim radius 1 grid, compute the offsets of all radius-neighboring grids
    @staticmethod
    def __neighbor_offsets(dim: int, radius: float) -> set[GridCoord]:
        max_offset = math.ceil(radius)
        res: set[GridCoord] = set()
        # each dimension ranges from is -max_offset-1 to max_offset
        for neighbor_grid in product(range(-max_offset - 1, max_offset + 1), repeat=dim):
            distance = GridSpace.dist(neighbor_grid)
            if distance < radius:
                res.add(neighbor_grid)
        return res
