import logging
import os
from datetime import datetime

import diffprivlib
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans

from data_provider import CsvDataProvider, SimpleDataProvider, ToyDataProvider, H5DataProvider, ArffDataProvider, \
    CsvWithHeaderDataProvider
from datatype_grid import GridSpace, GridCoord
from histogram import Histogram, SumHistogram, NoisyHistogram, max_diff, GridLabels, MockHistogram
from datatype_point import Points
from noise import *
from plot_data import Printer

import time
import argparse

parser = argparse.ArgumentParser(description='DPDBSCAN Experiments.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
parser.add_argument('--skip_dbscan', dest='skip_dbscan', action='store_const',
                    const=True, default=False,
                    help='skips running the original DBSCAN')
parser.add_argument('--skip_kmeans', dest='skip_kmeans', action='store_const',
                    const=True, default=False,
                    help='skips running the original KMEANS')
parser.add_argument('--skip_dp_dbscan', dest='skip_dp_dbscan', action='store_const',
                    const=True, default=False,
                    help='skips running DPDBSCAN')
parser.add_argument('--skip_dp_kmeans', dest='skip_dp_kmeans', action='store_const',
                    const=True, default=False,
                    help='skips running DPKmeans')

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
now = datetime.now()
now_str = now.strftime("%m-%d-%Y_%H-%M-%S-%f")
output_folder = "./output/" + now_str + "/"
os.makedirs(output_folder, exist_ok=True)
logging.info("Created directory: " + output_folder)

# generate data
# data_provider = ToyDataProvider()

# supported: toy circles moons varied aniso blobs no_structure
# data_provider = SimpleDataProvider("blobs")

data_provider = CsvWithHeaderDataProvider.collisions()

# supported: cluto_t4 cluto_t5
# data_provider = ArffDataProvider.atom()

pts = data_provider.get_data()
dim = pts.get_dim()
low, high = pts.get_ranges()
params = data_provider.get_params()
alpha = params["alpha"]
min_pts = params["min_samples"]
seed = 0
epsilon = 1.
beta = 1. / 2

printer = Printer(pts, output_folder, True, False, figsize=(6,6), draw_edge=False)
true_labels = data_provider.get_true_labels()
true_label_set = set(true_labels)
n_clusters = len(true_label_set) - (1 if -1 in true_label_set else 0)

printer.plot_labels(labels=true_labels, radius=0., title="true_cluster")

if not args.skip_dbscan:
    timer = time.time()
    dbs = DBSCAN(eps=alpha, min_samples=min_pts)
    dbs.fit(pts.get())
    dbscan_labels = Points.compute_dbscan_labels(dbs)
    print('ARI(Original, DBSCAN) = ', metrics.adjusted_rand_score(true_labels, dbscan_labels))
    print('AMI(Original, DBSCAN) = ', metrics.adjusted_mutual_info_score(true_labels, dbscan_labels))
    logging.info("DBSCAN time: %.2f seconds", time.time() - timer)
    printer.plot_labels(labels=dbscan_labels, radius=alpha, title="dbscan_" + str(alpha) + "_" + str(min_pts))

if not args.skip_kmeans:
    timer = time.time()
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pts.get())
    print('ARI(Original, KMeans) = ', metrics.adjusted_rand_score(true_labels, kmeans.labels_))
    print('AMI(Original, KMeans) = ', metrics.adjusted_mutual_info_score(true_labels, kmeans.labels_))
    logging.info("KMeans time: %.2f seconds", time.time() - timer)
    printer.plot_labels(labels=kmeans.labels_, radius=0., title="kmeans_" + str(n_clusters))

if not args.skip_dp_dbscan:
    # build grids
    timer = time.time()
    step_timer = time.time()
    grid_scale = params["grid_scale"]
    grid_space = GridSpace(dim, low, high, alpha, grid_scale)
    logging.info("GridSpace width: %.2g, num_grids: %d", grid_space.width, grid_space.num_grids)
    num_grids = grid_space.num_grids
    grid_counts = Histogram.build_from_pts(pts, grid_space)
    sum_hist = SumHistogram.build_from_counts(grid_counts, grid_space)
    logging.info("Histogram build time: %.2f seconds", time.time() - step_timer)

    # add noise
    step_timer = time.time()

    # directly adding noise to sum_hist, which has sensitivity neighbor
    noise_gen: NoiseGenerator = LaplaceNoise(seed, 1, epsilon)
    noises = noise_gen.generate(grid_space.num_grids)
    noise_bound = noise_gen.max_sum_noise(beta, noises.shape[0], len(grid_space.neighbor_offsets))

    noisy_counts = NoisyHistogram.build_with_noise(grid_counts, noises)
    noisy_sum = SumHistogram.build_from_counts(noisy_counts, grid_space)

    logging.info("Add noise time: %.2f seconds", time.time() - step_timer)
    logging.info("max noise seen: %.2f, max noise expected: %.2f, min_pts: %d",
                 max_diff(sum_hist, noisy_sum), noise_bound, min_pts)

    # find superset of core grids and give unique labels
    step_timer = time.time()
    grid_labels = GridLabels.label_high_freq(noisy_sum, min_pts + noise_bound)
    logging.info("Find core grids time: %.2f seconds", time.time() - step_timer)

    # merge neighboring core-grids by union-find
    step_timer = time.time()
    num_clusters = grid_labels.merge_neighbors(grid_space)
    logging.info("Merge grids time: %.2f seconds", time.time() - step_timer)
    logging.info("DPDBSCAN num clusters: %d", num_clusters)
    logging.info("DPDBSCAN time: %.2f seconds", time.time() - timer)

    printer.plot_grid(grid_space, labels=GridLabels.label_all(grid_space), hist=noisy_counts, title="noisy_counts")
    printer.plot_grid(grid_space, labels=GridLabels.label_all(grid_space), hist=sum_hist, title="neighbor_sum")
    printer.plot_grid(grid_space, labels=GridLabels.label_all(grid_space), hist=noisy_sum, title="noisy_sum")
    printer.plot_grid(grid_space, labels=grid_labels, hist=MockHistogram(), title="dp_dbscan_" + str(epsilon) + "_grids")
    point_labels = grid_labels.obtain_point_labels(pts, grid_space)
    printer.plot_labels(labels=point_labels, radius=0., title="dp_dbscan_" + str(epsilon) + "_pts")

    print('ARI(Original, DPDBSCAN) = ', metrics.adjusted_rand_score(true_labels, point_labels))
    print('AMI(Original, DPDBSCAN) = ', metrics.adjusted_mutual_info_score(true_labels, point_labels))


# k-means
if not args.skip_dp_kmeans:
    timer = time.time()

    kmeans = diffprivlib.models.k_means.KMeans(n_clusters, epsilon=epsilon, bounds=(low, high), random_state=seed)
    kmeans.fit(pts.get())

    logging.info("DPKmeans time: %.2f seconds", time.time() - timer)

    printer.plot_centers(centers=kmeans.cluster_centers_, title="dp_kmeans_" + str(epsilon) + "_" + str(n_clusters) + "_centers")
    printer.plot_labels(labels=kmeans.labels_, radius=0., title="dp_kmeans_" + str(epsilon) + "_" + str(n_clusters) + "_pts")

    print('ARI(Original, DPKmeans) = ', metrics.adjusted_rand_score(true_labels, kmeans.labels_))
    print('AMI(Original, DPKmeans) = ', metrics.adjusted_mutual_info_score(true_labels, kmeans.labels_))

