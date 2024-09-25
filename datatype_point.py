import numpy as np
from nptyping import NDArray, Shape, typing_
from sklearn.cluster import DBSCAN

Point = NDArray[Shape["N, D"], typing_.Number]
PointLabel = int
PointLabels = NDArray[Shape["N"], typing_.Int]


class Points:

    def __init__(self, inner: Point):
        self.inner = inner

    def get(self):
        return self.inner

    def get_dim(self):
        return self.inner.shape[1]

    def get_size(self):
        return self.inner.shape[0]

    def get_ranges(self) -> (Point, Point):
        return self.inner.min(0), self.inner.max(0)

    @staticmethod
    def dist(p: Point, q: Point) -> float:
        return np.linalg.norm(p - q)

    @staticmethod
    def __sorted_list_difference(full_list, partial_list) -> list:
        i, j = 0, 0
        result = []
        while i < len(full_list) and j < len(partial_list):
            if full_list[i] < partial_list[j]:
                result.append(full_list[i])
                i += 1
            elif full_list[i] > partial_list[j]:
                j += 1
            else:  # full_list[i] == partial_list[j]
                i += 1
                j += 1
        while i < len(full_list):
            result.append(full_list[i])
            i += 1
        return result

    # by default, DBSCAN labels boarder points arbitrarily, which we mark as noises instead
    @staticmethod
    def compute_dbscan_labels(dbs: DBSCAN) -> PointLabels:
        labels = dbs.labels_.copy()
        core_points = dbs.core_sample_indices_
        non_core_points = Points.__sorted_list_difference(range(0, len(labels)), core_points)
        for idx in non_core_points:
            labels[idx] = -1
        return labels
