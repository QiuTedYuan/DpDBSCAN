# generate data
from abc import ABC, abstractmethod
from glob import glob

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from datatype_point import Points, PointLabels


class DataProvider(ABC):
    pts: Points = None
    labels: PointLabels = None
    params: dict = {"alpha": 0.5,
                    "min_samples": 10,
                    "grid_scale": 1.}

    @abstractmethod
    def has_true_labels(self) -> bool:
        return False

    def get_data(self) -> Points:
        return self.pts

    def get_true_labels(self) -> PointLabels:
        return self.labels

    def get_params(self) -> dict:
        return self.params

    # geometric data have been pre-processed for distance calculation
    # original (x, y) => (x * scale[0], y * scale[1])
    def get_scales(self) -> np.array:
        return np.ones(self.pts.get_dim())


# examples from ski-learn
# see https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
class SimpleDataProvider(DataProvider):
    def has_true_labels(self) -> bool:
        return True

    def __init__(self, pts, labels, params):
        self.pts = pts
        self.labels = labels
        self.params = params

    @classmethod
    def toy(cls):
        pts = Points(np.array([[-1, 0], [-1, 0], [0, 0], [1, 0], [1, 0], [0, -1], [0, 1]]))
        labels = np.array([-1, -1, 0, -1, -1, -1, -1])
        params = {"alpha": 1, "min_samples": 7, "grid_scale": 0.1}
        return cls(pts, labels, params)

    @classmethod
    def circles(cls, n_samples, seed):
        x, labels = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
        pts = Points(StandardScaler().fit_transform(x))
        params = {"alpha": 0.2, "min_samples": 10, "grid_scale": 1}
        return cls(pts, labels, params)

    @classmethod
    def moons(cls, n_samples, seed):
        x, labels = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
        pts = Points(StandardScaler().fit_transform(x))
        params = {"alpha": 0.2, "min_samples": 7, "grid_scale": 1}
        return cls(pts, labels, params)

    @classmethod
    def blobs(cls, n_samples, seed):
        x, labels = datasets.make_blobs(n_samples=n_samples, centers=[[1, 1], [-1, -1], [1.5, -1.5]],
                                        cluster_std=[0.4, 0.1, 0.75], random_state=seed)
        pts = Points(StandardScaler().fit_transform(x))
        params = {"alpha": 0.2, "min_samples": 7, "grid_scale": 1}
        return cls(pts, labels, params)


# each longitude (x) will be scaled by a factor of cos(latitude), to approximate the real distance
# this differs from using km by a scale factor of km_per_latitude()
class LongitudeLatitudeDataProvider(DataProvider):
    def __init__(self, long_lat, params, latitude):
        self.params = params
        self.latitude = latitude
        # keep latitude and map longitude to (longitude * cos(latitude))
        self.scales = [np.cos(latitude / 180. * np.pi), 1.]
        self.long_lat = long_lat
        self.pts = Points(np.array(list(map(lambda xy: [xy[0] * self.scales[0], xy[1]], self.long_lat))))

    @staticmethod
    def km_per_latitude() -> float:
        return np.pi * 6371. / 180.

    def km_per_longitude(self) -> float:
        return np.pi * 6371. * np.cos(self.latitude) / 180.

    def has_true_labels(self) -> bool:
        return False

    def get_scales(self) -> list[float]:
        return self.scales

    @classmethod
    def crash(cls):
        df = pd.read_csv("datasets/crashes_240928.csv")
        data = df[["LONGITUDE", "LATITUDE"]].to_numpy()
        long_lat = data[((data[:, 0] > -74.2) & (data[:, 0] < -73) & (data[:, 1] > 40) & (data[:, 1] < 41))]
        params = {"alpha": 0.1 / cls.km_per_latitude(), "min_samples": 300, "grid_scale": 1}
        return cls(long_lat, params, 40)

    @classmethod
    def cabs(cls):
        files = glob("datasets/cabspottingdata/new_*.txt")
        end_points = list()
        for file in files:
            trajectory = np.genfromtxt(file, dtype=[float, float, int, int], names=None)
            for point in trajectory:
                end_points.append(np.array([point[1], point[0]]))

        data = np.array(end_points)
        long_lat = data[((data[:, 0] <= -122.3) & (data[:, 0] >= -122.6) &
                         (data[:, 1] >= 37.55) & (data[:, 1] <= 37.85))]
        params = {"alpha": 0.02 / cls.km_per_latitude(), "min_samples": 500, "grid_scale": 1, "linear": True}
        return cls(long_lat, params, 37.8)

    @classmethod
    def cabs_tiny(cls):
        files = glob("datasets/cabspottingdata/new_*.txt")
        end_points = list()
        for file in files:
            trajectory = np.genfromtxt(file, dtype=[float, float, int, int], names=None)
            loaded = False
            for point in trajectory:
                if not loaded and point[2] == 1:
                    loaded = True
                    end_points.append(np.array([point[1], point[0]]))
                elif loaded and point[2] == 0:
                    loaded = False
                    end_points.append(np.array([point[1], point[0]]))

        data = np.array(end_points)
        long_lat = data[((data[:, 0] <= -122.38) & (data[:, 0] >= -122.48) &
                         (data[:, 1] >= 37.72) & (data[:, 1] <= 37.82))]
        params = {"alpha": 0.05 / cls.km_per_latitude(), "min_samples": 1000, "grid_scale": 1}
        return cls(long_lat, params, 37.8)


class ArffDataProvider(DataProvider):
    @staticmethod
    def normalize_enum(x) -> int:
        if x == b'noise':
            return -1
        else:
            return x

    def __init__(self, path, params):
        data, meta = arff.loadarff('clustering-benchmark/src/main/resources/datasets/artificial/' + path)
        df = pd.DataFrame(data)
        self.pts = Points(df.iloc[:, :-1].to_numpy())
        self.labels = df.iloc[:, -1].map(ArffDataProvider.normalize_enum).astype(int).to_numpy()
        self.params = params

    def has_true_labels(self) -> bool:
        return True

    @classmethod
    def cluto_t4(cls):
        params = {"alpha": 9, "min_samples": 11, "grid_scale": 1}
        return cls('cluto-t4-8k.arff', params)

    @classmethod
    def cluto_t5(cls):
        params = {"alpha": 9, "min_samples": 20, "grid_scale": 1}
        return cls('cluto-t5-8k.arff', params)

    @classmethod
    def cluto_t7(cls):
        params = {"alpha": 12, "min_samples": 20, "grid_scale": 1}
        return cls('cluto-t7-10k.arff', params)


class PrinterParams:
    def __init__(self, dpi: int, ext: str, fig_size: (int, int) = (10, 10),
                 draw_label: bool = True, draw_edge: bool = True, font_size: int = 12, marker_size: int = 4):
        self.fig_size = fig_size
        self.draw_label = draw_label
        self.draw_edge = draw_edge
        self.dpi = dpi
        self.ext = ext
        self.marker_size = marker_size
        self.font_size = font_size


# data_provider, draw_label, draw_edge, fig_size
def get_data(name: str, dpi, ext) -> (DataProvider, PrinterParams):
    default_params = PrinterParams(dpi, ext)
    cluto_params = PrinterParams(dpi, ext, fig_size=(10, 6))
    real_params = PrinterParams(dpi, ext, draw_label=False, draw_edge=False)
    match name:
        case 'toy':
            return SimpleDataProvider.toy(), default_params
        case 'moons':
            return SimpleDataProvider.moons(2000, 30), default_params
        case 'blobs':
            return SimpleDataProvider.blobs(2000, 30), default_params
        case 'circles':
            return SimpleDataProvider.circles(2000, 30), default_params
        case 'cluto_t4':
            return ArffDataProvider.cluto_t4(), cluto_params
        case 'cluto_t5':
            return ArffDataProvider.cluto_t5(), cluto_params
        case 'cluto_t7':
            return ArffDataProvider.cluto_t7(), cluto_params
        case 'crash':
            return LongitudeLatitudeDataProvider.crash(), real_params
        case 'cabs_tiny':
            return LongitudeLatitudeDataProvider.cabs_tiny(), real_params
        case 'cabs':
            return LongitudeLatitudeDataProvider.cabs(), real_params
        case _:
            raise Exception("Unsupported dataset")
