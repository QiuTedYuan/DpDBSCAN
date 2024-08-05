# generate test data

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons


# normalize values in range [low, high) to [0, u)
class Normalizer:
    def __init__(self, low: float, high: float, u: float):
        assert high > low
        self.low = low
        self.high = high
        self.u = u

    def normalize_value(self, v: float) -> float:
        assert self.low <= v < self.high
        return (v - self.low) / (self.high - self.low) * self.u

    @staticmethod
    def normalize_pts(pts: np.ndarray, u: float) -> np.ndarray:
        mins = pts.min(0)
        maxs = pts.max(0)
        x_normalizer = Normalizer(mins[0], maxs[0] * 1.01, u)
        y_normalizer = Normalizer(mins[1], maxs[1] * 1.01, u)
        res: list = []
        for pt in pts:
            res.append((x_normalizer.normalize_value(pt[0]),
                        y_normalizer.normalize_value(pt[1])))
        return np.asarray(res)


class Generator:
    n_samples: int
    random_state: int

    def __init__(self, n_samples, random_state):
        self.n_samples = n_samples
        self.random_state = random_state

    def generate_blobs(self, u: float) -> (np.ndarray, np.array):
        centers = [[1, 1], [-1, -1], [1.5, -1.5]]
        pts, label = make_blobs(n_samples=self.n_samples,
                                centers=centers,
                                cluster_std=[0.4, 0.1, 0.75],
                                random_state=self.random_state)
        return Normalizer.normalize_pts(pts, u), label

    def generate_circles(self, u: float) -> (np.ndarray, np.array):
        pts, label = make_circles(n_samples=self.n_samples,
                                  factor=0.5,
                                  noise=0.05,
                                  random_state=self.random_state)
        return Normalizer.normalize_pts(pts, u), label

    def generate_moons(self, u: float) -> (np.ndarray, np.array):
        pts, label = make_moons(n_samples=self.n_samples,
                                noise=0.05,
                                random_state=self.random_state)
        return Normalizer.normalize_pts(pts, u), label
