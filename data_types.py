import math
from itertools import product
from typing import TypeAlias

import numpy as np
import numpy.typing

# ################## Point ##################
# Point(s) refers to high-dimensional points
Point: TypeAlias = numpy.typing.NDArray
Points: TypeAlias = numpy.typing.NDArray
Labels: TypeAlias = numpy.typing.NDArray[numpy.int32]


def get_dim(p: Point) -> int:
    return len(p)


def dist(p: Point, q: Point) -> float:
    sum_square: float = 0.
    for i in range(get_dim(p)):
        sum_square += 1. * (p[i] - q[i]) * (p[i] - q[i])
    return math.sqrt(sum_square)


# ################## Grid ##################
# The index and coordinates of a d-dim grid
# The coordinate is equivalent to the low point when radius is 1
GridIndex: TypeAlias = int
GridCoord: TypeAlias = numpy.typing.NDArray[numpy.int32]
# A histogram maps grid index to a frequency
Histogram: TypeAlias = np.array


# Compute minimum distance of two axis-parallel d-dim grids of width 1
def min_dist(g: GridCoord, h: GridCoord) -> float:
    # compute the lower/upper bounding points
    g_low: Point = g
    g_high: Point = np.add(g, 1)
    h_low: Point = h
    h_high: Point = np.add(h, 1)
    dim: int = get_dim(g_low)
    sum_square: float = 0
    for i in range(dim):
        x, y, x_, y_ = g_low[i], g_high[i], h_low[i], h_high[i]
        if x_ < x:
            x, y, x_, y_ = x_, y_, x, y
        # distance between [x, y) and [x_, y_)
        if x_ <= y:
            sum_square += 0
        else:
            sum_square += 1. * (y - x_) * (y - x_)
    return math.sqrt(sum_square)


# Given a d-dim radius 1 grid, compute the offsets of all radius-neighboring grids
def neighbor_offsets(dim: int, radius: float) -> set[GridCoord]:
    max_offset = math.ceil(radius)
    res: set[GridCoord] = set()
    # each dimension ranges from is -max_offset-1 to max_offset
    for grid in product(range(-max_offset - 1, max_offset + 1), repeat=dim):
        distance = min_dist(grid, np.zeros(dim, dtype=int))
        if distance < radius:
            res.add(grid)
    return res


# Denotes a d-dim grid space over [0, u)
# alpha_dbscan: DBSCAN radius, s_approx: approximation factor
# grid width = s * alpha / sqrt(dim), will be rescaled to 1
class GridHelper:
    def __init__(self, dim: int, univ: float, alpha: float, s: float):
        self.dim = dim
        self.univ = univ
        self.alpha = alpha
        self.s = s
        self.width = s * alpha / math.sqrt(dim)
        self.grids_per_dim: int = math.ceil(univ / self.width)
        self.num_grids: int = int(math.pow(self.grids_per_dim, self.dim))
        self.neighbor_offsets = neighbor_offsets(self.dim, radius=self.alpha / self.width)

    def is_valid(self, coord: GridCoord) -> bool:
        for i in range(self.dim):
            if coord[i] < 0 or coord[i] >= self.grids_per_dim:
                return False
        return True

    def get_coord(self, idx: GridIndex) -> GridCoord:
        coord: GridCoord = np.zeros(self.dim, dtype=int)
        for i in range(self.dim).__reversed__():
            coord[i] = (idx % self.grids_per_dim)
            idx = idx // self.grids_per_dim
        return coord

    def get_index(self, coord: GridCoord) -> GridIndex:
        idx: GridIndex = 0
        for i in range(self.dim):
            idx *= self.grids_per_dim
            idx += coord[i]
        return idx

    def get_grid(self, p: Point) -> GridCoord:
        return np.floor_divide(p, self.width)

    def num_grids(self) -> int:
        return int(math.pow(self.grids_per_dim, self.dim))

    def count_points(self, pts: Points) -> Histogram:
        res: Histogram = np.zeros(self.num_grids, dtype=float)
        for p in pts:
            g: GridCoord = self.get_grid(p)
            res[int(self.get_index(g))] += 1
        return res

    def compute_upper_bound_hist(self, count_hist: Histogram) -> Histogram:
        assert len(count_hist) == self.num_grids
        sum_hist: Histogram = np.zeros(self.num_grids, dtype=float)
        for current_idx in range(self.num_grids):
            current_coord: GridCoord = self.get_coord(current_idx)
            for offset in self.neighbor_offsets:
                neighbor_coord: GridCoord = np.add(current_coord, offset)
                if self.is_valid(neighbor_coord):
                    neighbor_idx: int = self.get_index(neighbor_coord)
                    sum_hist[current_idx] += count_hist[neighbor_idx]
        return sum_hist


assert len(neighbor_offsets(2, math.sqrt(2))) == 21
