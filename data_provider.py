# generate test data
from abc import ABC, abstractmethod

import h5py
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from datatype_point import Points, PointLabels


class DataProvider(ABC):

    params = None

    @abstractmethod
    def get_data(self) -> Points:
        pass

    @abstractmethod
    def get_true_labels(self) -> PointLabels:
        pass

    def set_params(self, params):
        self.params = params

    def get_params(self) -> dict:
        if self.params is None:
            self.set_params({"alpha": 0.5, "min_samples": 10, "grid_scale": 1})
        return self.params


class ToyDataProvider(DataProvider):
    def get_data(self) -> Points:
        return Points(np.array([[-1, 0], [-1, 0], [0, 0], [1, 0], [1, 0], [0, -1], [0, 1]]))

    def get_true_labels(self) -> PointLabels:
        return np.array([-1, -1, 0, -1, -1, -1, -1])

    def get_params(self) -> dict:
        return {"alpha": 1, "min_samples": 7, "grid_scale": 0.1}


# examples from ski-learn (n_samples=500, seed=30)
# see https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
class SimpleDataProvider(DataProvider):
    default_base = {
        "quantile": 0.3,
        "alpha": 0.3,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 3,
        "n_clusters": 3,
        "min_samples": 7,
        "xi": 0.05,
        "min_cluster_size": 0.1,
        "allow_single_cluster": True,
        "hdbscan_min_cluster_size": 15,
        "hdbscan_min_samples": 3,
        "random_state": 42,
        "grid_scale": 0.8,
    }

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 2000
    seed = 30
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    rng = np.random.RandomState(seed)
    no_structure = rng.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )

    datasets = {
        "circles": (noisy_circles,
                    {
                        "damping": 0.77,
                        "preference": -240,
                        "quantile": 0.2,
                        "n_clusters": 2,
                        "min_samples": 5,
                        "xi": 0.08,
                    },
                    ),
        "moons": (noisy_moons,
                  {
                      "damping": 0.75,
                      "preference": -220,
                      "n_clusters": 2,
                      "alpha": 0.2,
                      "min_samples": 7,
                      "xi": 0.1,
                  },
                  ),
        "varied": (varied,
                   {
                       "alpha": 0.18,
                       "n_neighbors": 2,
                       "min_samples": 5,
                       "xi": 0.01,
                       "min_cluster_size": 0.2,
                   },
                   ),
        "aniso": (
            aniso,
            {
                "alpha": 0.15,
                "n_neighbors": 2,
                "min_samples": 5,
                "xi": 0.1,
                "min_cluster_size": 0.2,
            },
        ),
        "blobs": (blobs, {"alpha": 0.25, "min_samples": 15, "xi": 0.1, "min_cluster_size": 0.2}),
        "no_structure": (no_structure, {}),
    }

    def __init__(self, shape):
        ((x, self.labels), algo_params) = self.datasets[shape]
        self.pts = Points(StandardScaler().fit_transform(x))
        self.params = self.default_base.copy()
        self.params.update(algo_params)

    def get_data(self) -> Points:
        return self.pts

    def get_params(self) -> dict:
        return self.params

    def get_true_labels(self) -> PointLabels:
        return self.labels


class CsvDataProvider(DataProvider):
    def __init__(self, path, delimiter=' ', label_path=None):
        self.pts = Points(np.genfromtxt('datasets/' + path, delimiter=delimiter))
        self.labels = None if label_path is None else np.genfromtxt('datasets/' + label_path, dtype=int)
        self.params = None

    def get_data(self) -> Points:
        return self.pts

    def set_params(self, params):
        self.params = params

    def get_params(self) -> dict:
        if self.params is None:
            self.params = {"alpha": 0.5, "min_samples": 10, "grid_scale": 0.5}
        return self.params

    def get_true_labels(self) -> PointLabels:
        return self.labels

    @classmethod
    def banana(cls):
        res = cls("banana", ' ', 'banana_label')
        res.set_params({"alpha": 0.03, "min_samples": 12, "grid_scale": 0.5})
        return res

    @classmethod
    def cluto(cls):
        res = cls("cluto", ' ', 'cluto_label')
        res.set_params({"alpha": 10, "min_samples": 12, "grid_scale": 1})
        return res

    @classmethod
    def hypercube(cls):
        res = cls("hypercube", ' ', 'hypercube_label')
        res.set_params({"alpha": 0.5, "min_samples": 10, "grid_scale": 0.7})
        return res

class CsvWithHeaderDataProvider(DataProvider):
    def __init__(self, path, columns, pred):
        df = pd.read_csv("datasets/" + path)
        self.pts = Points(df[columns].dropna().query(pred).to_numpy())

    def get_data(self) -> Points:
        return self.pts

    def get_true_labels(self) -> PointLabels:
        return np.zeros(self.pts.get_size())

    @classmethod
    def collisions(cls):
        res = cls("collisions_20240923.csv", ["LONGITUDE", "LATITUDE"],
                  "LONGITUDE > -74.2 and LONGITUDE < -73 and LATITUDE > 40 and LATITUDE < 41")
        res.set_params({"alpha": 0.001, "min_samples": 300, "grid_scale": 1})
        return res


class H5DataProvider(DataProvider):
    def __init__(self, path):
        file = h5py.File('datasets/' + path, 'r')
        keys = file.keys()
        self.data = {}
        for key in keys:
            self.data[key] = file[key]
        self.pts = Points(self.data["DBSCAN"][:])
        self.labels = self.data["Clusters"][:]
        self.params = None

    def get_data(self) -> Points:
        return self.pts

    def get_params(self) -> dict:
        if self.params is None:
            self.params = {"alpha": 0.5, "min_samples": 10, "grid_scale": 0.5}
        return self.params

    def get_true_labels(self) -> PointLabels:
        return self.labels

    @classmethod
    def twitter_small(cls):
        res = cls("twitterSmall.h5.h5")
        res.set_params({"alpha": 0.01, "min_samples": 40, "grid_scale": 0.8})
        return res

    @classmethod
    def berman_small(cls):
        res = cls("bremenSmall.h5.h5")
        res.set_params({"alpha": 100, "min_samples": 312, "grid_scale": 1.25})
        return res


class ArffDataProvider(DataProvider):
    def __init__(self, path):
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        self.pts = Points(df.iloc[:, :-1].to_numpy())
        self.labels = (df.iloc[:, -1].map(lambda x: -1 if x == b'noise' else x)
                       .map(lambda x: 0 if x == b'L' else (1 if x == b'B' else 2)).astype(int).to_numpy())

    def get_data(self) -> Points:
        return self.pts

    def get_true_labels(self) -> PointLabels:
        return self.labels

    @classmethod
    def cluto_t4(cls):
        res = cls('../clustering-benchmark/src/main/resources/datasets/artificial/cluto-t4-8k.arff')
        res.params = ({"alpha": 9, "min_samples": 11, "grid_scale": 1})
        return res

    @classmethod
    def cluto_t5(cls):
        res = cls('../clustering-benchmark/src/main/resources/datasets/artificial/cluto-t5-8k.arff')
        res.params = ({"alpha": 9, "min_samples": 20, "grid_scale": 1})
        return res

    @classmethod
    def cluto_t7(cls):
        res = cls('../clustering-benchmark/src/main/resources/datasets/artificial/cluto-t7-10k.arff')
        res.params = ({"alpha": 12, "min_samples": 20, "grid_scale": 1})
        return res

    @classmethod
    def atom(cls):
        res = cls('../clustering-benchmark/src/main/resources/datasets/real-world/balance-scale.arff')
        res.params = ({"alpha": 1.1, "min_samples": 5, "grid_scale": 1})
        return res