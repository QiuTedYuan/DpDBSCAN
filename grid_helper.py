# helper functions for grids
import math

import numpy as np


# distance between points (x1, y1) and (x2, y2)
def dist_point(x1, y1, x2, y2) -> float:
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


# min distance between rectangles (x1, y1) -- (x1b, y1b) to (x2, y2) -- (x2b, y2b)
def dist_rect(x1, y1, x1b, y1b, x2, y2, x2b, y2b) -> float:
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist_point(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist_point(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist_point(x1b, y1, x2, y2b)
    elif right and top:
        return dist_point(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:  # rectangles intersect
        return 0


class GridHelper:
    # d = dimension
    # u = size of each domain
    # alpha = DBSCAN factor
    # eta = constant factor
    # w = grid width
    def __init__(self, u: float, alpha: float, eta: float):
        self.d = 2
        self.u = u
        self.alpha = alpha
        self.eta = eta
        self.w = eta * alpha / math.sqrt(self.d)
        self.grid_each_dim = math.ceil(u / self.w)
        self.num_grids = self.grid_each_dim * self.grid_each_dim
        self.neighbor_offsets = self.compute_neighbor_offsets(eta, self.d)

    def loc2grid(self, x: float, y: float) -> (int, int):
        return math.floor(x / self.w), math.floor(y / self.w)

    def idx2loc_bot_left(self, idx: int) -> (float, float):
        x, y = self.idx2grid(idx)
        return x * self.w, y * self.w

    def grid2idx(self, x: int, y: int) -> int:
        return x * self.grid_each_dim + y

    def idx2grid(self, idx: int) -> (int, int):
        return idx // self.grid_each_dim, idx % self.grid_each_dim

    def is_valid_grid(self, x: int, y: int):
        return 0 <= x < self.grid_each_dim and 0 <= y < self.grid_each_dim

    def get_neighbor_sum_hist(self, hist: np.array) -> np.array:
        assert len(hist) == self.num_grids
        sum_hist = np.zeros(self.num_grids)
        for idx in range(self.num_grids):
            x, y = self.idx2grid(idx)
            for offset in self.neighbor_offsets:
                x_ = x + offset[0]
                y_ = y + offset[1]
                if self.is_valid_grid(x_, y_):
                    sum_hist[idx] += hist[self.grid2idx(x_, y_)]
        return sum_hist



    # the alpha-neighborhood of a grid with width w = eta * alpha / sqrt(d)
    # equivalently, the (sqrt(d) / eta)-neighborhood of unit grids
    @staticmethod
    def compute_neighbor_offsets(eta: float, d: int) -> list:
        scaled_dist = math.sqrt(d) / eta
        max_offset = math.ceil(scaled_dist)
        res = []
        # leftmost is (-max_offset,0) -- (-max_offset+1, 0)
        # rightmost is (max_offset, 0) -- (max_offset+1, 0)
        for x_offset in range(-max_offset, max_offset + 1):
            for y_offset in range(-max_offset, max_offset + 1):
                rect_dist = dist_rect(0, 0, 1, 1, x_offset, y_offset, x_offset + 1, y_offset + 1)
                if rect_dist < scaled_dist:
                    res.append((x_offset, y_offset))
        return res


assert len(GridHelper.compute_neighbor_offsets(1, 2)) == 21