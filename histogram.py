import numpy as np
from unionfind import unionfind

from datatype_grid import GridSpace, GridKey, GridCoord
from datatype_point import Points, PointLabels
from noise import Noises, NoiseGenerator


class Histogram:
    def __init__(self):
        self._inner = {}

    def get_by_key(self, key: GridKey) -> float:
        return self._inner.get(key, 0)

    def increment(self, key: GridKey, amt: float):
        self._inner[key] = self._inner.get(key, 0) + amt

    def max_freq(self) -> float:
        return max(abs(max(self._inner.values())), abs(min(self._inner.values())))

    def size(self):
        return len(self._inner)

    def items(self):
        return self._inner.items()

    def keys(self):
        return self._inner.keys()

    @classmethod
    def build_from_pts(cls, pts: Points, grid_helper: GridSpace):
        res = cls()
        for p in pts.get():
            coord = grid_helper.get_grid_of_point(p)
            res.increment(grid_helper.encode_as_key(coord), 1)
        return res


class MockHistogram(Histogram):

    def get_by_key(self, x: GridCoord) -> float:
        return 1

    def max_freq(self) -> float:
        return 1


class SumHistogram(Histogram):
    @classmethod
    def build_from_counts(cls, counts: Histogram, grid_space: GridSpace):
        res = cls()
        for grid_key, freq in counts._inner.items():
            current_coord: GridCoord = grid_space.decode_from_key(grid_key)
            for offset in grid_space.neighbor_offsets:
                neighbor_coord: GridCoord = tuple(np.add(current_coord, offset).astype(dtype=int))
                if grid_space.is_valid(neighbor_coord):
                    res.increment(grid_space.encode_as_key(neighbor_coord), freq)
        return res


# a noisy histogram only storing counts > threshold
class NoisyHistogram(Histogram):
    # currently we use O(u) time to visit every cell, can be improved to O(n)
    @classmethod
    def naive_build(cls, histogram: Histogram, noise_gen: NoiseGenerator, universe: int):
        res = cls()
        noises = noise_gen.generate(universe)
        for key, noise in enumerate(noises):
            noisy_freq: float = histogram.get_by_key(key) + noise
            res.increment(key, noisy_freq)
        return res

    @classmethod
    def linear_time_build(cls, histogram: Histogram, noise_gen: NoiseGenerator, universe: int, gamma):
        res = cls()
        noises = noise_gen.generate(histogram.size())
        idx = 0
        for key, freq in histogram.items():
            noisy_freq = freq + noises[idx]
            idx += 1
            if noisy_freq >= gamma:
                res.increment(key, noisy_freq)
        empty_count = universe - histogram.size()
        p = 0.5 * noise_gen.tail_bound(gamma)
        m = noise_gen.generate_binomial(empty_count, p)
        for idx in range(m):
            j = np.random.randint(universe)
            while j in histogram.keys() or j in res.keys():
                j = np.random.randint(universe)
            noisy_freq = noise_gen.generate_large(gamma)
            res.increment(j, noisy_freq)
        return res


def max_diff(h1: Histogram, h2: Histogram) -> float:
    m = 0.
    for key, freq in h1.items():
        m = max(m, abs(h2.get_by_key(key) - freq))
    for key in set(h2.keys()) - set(h1.keys()):
        m = max(m, abs(h1.get_by_key(key)))
    return m


class GridLabels:
    def __init__(self):
        self._inner = {}

    @classmethod
    def label_all(cls, hist: Histogram):
        res = cls()
        for key, freq in hist.items():
            res._inner[key] = 0
        return res

    def values(self):
        return self._inner.values()

    def items(self):
        return self._inner.items()

    @classmethod
    def label_high_freq(cls, hist: Histogram, threshold):
        res = cls()
        cnt = 0
        for key, freq in hist.items():
            if freq >= threshold:
                res._inner[key] = cnt
                cnt += 1
        return res

    def merge_neighbors(self, grid_space: GridSpace) -> int:
        uf = unionfind(grid_space.num_grids)
        for key, label in self._inner.items():
            current_coord: GridCoord = grid_space.decode_from_key(key)
            for offset in grid_space.neighbor_offsets:
                neighbor_coord: GridCoord = tuple(np.add(current_coord, offset))
                neighbor_key = grid_space.encode_as_key(neighbor_coord)
                if grid_space.is_valid(neighbor_coord) and self._inner.get(neighbor_key, -1) != -1:
                    uf.unite(key, neighbor_key)
        # normalize
        unique_labels = {}
        cnt = 0
        for key in self._inner:
            uf_label = uf.find(key)
            if uf_label not in unique_labels:
                unique_labels[uf_label] = cnt
                cnt += 1
            self._inner[key] = unique_labels[uf_label]
        return cnt

    # non-dp
    def obtain_point_labels(self, pts: Points, grid_space: GridSpace) -> PointLabels:
        res: PointLabels = np.zeros(pts.get_size(), dtype=int)
        for idx, p in enumerate(pts.get()):
            grid = grid_space.get_grid_of_point(p)
            res[idx] = self._inner.get(grid_space.encode_as_key(grid), -1)
        return res

