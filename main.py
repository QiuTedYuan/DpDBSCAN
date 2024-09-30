import argparse
import logging
import os
import time
from datetime import datetime

import diffprivlib
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans

from data_provider import *
from datatype_grid import GridSpace
from datatype_point import Points
from histogram import Histogram, SumHistogram, NoisyHistogram, max_diff, GridLabels, MockHistogram
from noise import *
from plot_data import Printer

# program args
parser = argparse.ArgumentParser(description='DPDBSCAN Experiments.')
parser.add_argument('-d', '--dataset', help="Dataset, default moons", required=False,
                    choices=['moons', 'blobs', 'circles', 'cluto_t4', 'cluto_t5', 'cluto_t7', 'cabspot', 'crash'],
                    default='moons')
parser.add_argument('-s', '--seed', help="Random Seed, default 0",
                    required=False, default=0, type=int)
parser.add_argument('--noise', help="Noise to generate DP histogram, default Laplace",
                    required=False, choices=['Laplace', 'Geometric', 'Gaussian'], default='Laplace')
parser.add_argument('--epsilon', help="Epsilon for DP mechanisms, default 1",
                    required=False, default=1.0, type=float)
parser.add_argument('--delta', help="Delta for DP mechanisms, default 0",
                    required=False, default=0.0, type=float)
parser.add_argument('--beta', help="Beta for Calculating Error Bounds, default 0.5",
                    required=False, default=0.5, type=float)
parser.add_argument('--alpha', help="override alpha for DBSCAN",
                    required=False, default=None, type=float)
parser.add_argument('--minpts', help="override minpts for DBSCAN",
                    required=False, default=None, type=int)
parser.add_argument('--grid_scale', help="override grid_scale for DP-DBSCAN",
                    required=False, default=None, type=float)

parser.add_argument('-p', '--plot', help="plot results",
                    action='store_true')
parser.add_argument('--debug', help="log level debug",
                    action='store_true')
parser.add_argument('--info', help="log level info",
                    action='store_true')
parser.add_argument('--skip_dbscan', action='store_true')
parser.add_argument('--skip_kmeans', action='store_true')
parser.add_argument('--skip_dp_dbscan', action='store_true')
parser.add_argument('--skip_dp_kmeans', action='store_true')

args = parser.parse_args()

match args.dataset:
    case 'toy':
        data_provider = SimpleDataProvider.toy()
        draw_label = True
        draw_edge = True
        fig_size = (10, 10)
        dpi = 100
    case 'moons':
        data_provider = SimpleDataProvider.moons(2000, 30)
        draw_label = True
        draw_edge = True
        fig_size = (10, 10)
        dpi = 100
    case 'blobs':
        data_provider = SimpleDataProvider.blobs(2000, 30)
        draw_label = True
        draw_edge = True
        fig_size = (10, 10)
        dpi = 100
    case 'circles':
        data_provider = SimpleDataProvider.circles(2000, 30)
        draw_label = True
        draw_edge = True
        fig_size = (10, 10)
        dpi = 100
    case 'cluto_t4':
        data_provider = ArffDataProvider.cluto_t4()
        draw_label = True
        draw_edge = True
        fig_size = (10, 6)
        dpi = 100
    case 'cluto_t5':
        data_provider = ArffDataProvider.cluto_t5()
        draw_label = True
        draw_edge = True
        fig_size = (10, 6)
        dpi = 100
    case 'cluto_t7':
        data_provider = ArffDataProvider.cluto_t7()
        draw_label = True
        draw_edge = True
        fig_size = (10, 6)
        dpi = 100
    case 'crash':
        data_provider = LongitudeLatitudeDataProvider.crash()
        fig_size = (10, 10)
        draw_label = False
        draw_edge = False
        dpi = 100
    case 'cabspot':
        data_provider = LongitudeLatitudeDataProvider.cabspot()
        fig_size = (10, 10)
        draw_label = False
        draw_edge = False
        dpi = 100
    case _:
        raise Exception("Unsupported dataset")

pts = data_provider.get_data()
dim = pts.get_dim()
low, high = pts.get_ranges()
params = data_provider.get_params()
alpha = params["alpha"] if args.alpha is None else args.alpha
min_pts = params["min_samples"] if args.minpts is None else args.minpts
grid_scale = params["grid_scale"] if args.grid_scale is None else args.grid_scale

if args.debug:
    log_level = logging.DEBUG
elif args.info:
    log_level = logging.INFO
else:
    log_level = logging.ERROR
logging.basicConfig(level=log_level)
logging.info('[Dataset] n=%d, d=%d, low=%s, high=%s, alpha=%.5g, min_pts=%d',
             pts.get_size(), dim, low, high, alpha, min_pts)

seed, epsilon, delta, beta = args.seed, args.epsilon, args.delta, args.beta
match args.noise:
    case 'Laplace':
        noise_gen: NoiseGenerator = LaplaceNoise(seed, 1, epsilon)
    case 'Geometric':
        noise_gen: NoiseGenerator = GeometricNoise(seed, 1, epsilon)
    case 'Gaussian':
        if delta <= 0:
            raise Exception("delta is required for Gaussian mechanism.")
        noise_gen: NoiseGenerator = GaussianNoise(seed, 1, epsilon, args.delta)
    case _:
        raise Exception("Unsupported noise type" + args.mechanism + ", try Laplace/Geometric/Gaussian")

output_folder = "./output/" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S-%f") + "/"
if args.plot:
    os.makedirs(output_folder, exist_ok=True)
    logging.info("Created directory: " + output_folder)
printer = Printer(data_provider, output_folder, args.plot, fig_size, draw_label, draw_edge, dpi)

if data_provider.has_true_labels():
    true_labels = data_provider.get_true_labels()
    true_label_set = set(true_labels)
    true_label_set.discard(-1)
    n_clusters = len(true_label_set)
    printer.plot_labels(labels=true_labels, radius=0., title="true_cluster")
else:
    true_labels = np.arange(0, pts.get_size())
    n_clusters = 3

if not args.skip_dbscan:

    timer = time.time()
    dbs = DBSCAN(eps=alpha, min_samples=min_pts)
    dbs.fit(pts.get())
    dbscan_labels = Points.compute_dbscan_labels(dbs)
    logging.info("[DBSCAN] time: %.5g seconds", time.time() - timer)
    dbscan_label_set = set(dbscan_labels)
    dbscan_label_set.discard(-1)
    logging.info("[DBSCAN] num clusters: %d", len(dbscan_label_set))
    printer.plot_labels(labels=dbscan_labels, radius=alpha, title="dbscan_" + str(alpha) + "_" + str(min_pts))
    if data_provider.has_true_labels():
        print('[DBSCAN] ARI = ', metrics.adjusted_rand_score(true_labels, dbscan_labels))
        print('[DBSCAN] AMI = ', metrics.adjusted_mutual_info_score(true_labels, dbscan_labels))

if not args.skip_kmeans:

    timer = time.time()
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pts.get())
    logging.info("[KMeans] time: %.5g seconds", time.time() - timer)

    printer.plot_labels(labels=kmeans.labels_, radius=0., title="kmeans_" + str(n_clusters))
    if data_provider.has_true_labels():
        print('[KMeans] ARI = ', metrics.adjusted_rand_score(true_labels, kmeans.labels_))
        print('[KMeans] AMI = ', metrics.adjusted_mutual_info_score(true_labels, kmeans.labels_))

if not args.skip_dp_dbscan:

    timer = time.time()

    # build grids
    step_timer = time.time()
    grid_space = GridSpace(dim, low, high, alpha, grid_scale)
    num_grids = grid_space.num_grids
    logging.info("[DP-DBSCAN] grid width: %.5g, num_grids: %d", grid_space.width, num_grids)
    grid_counts = Histogram.build_from_pts(pts, grid_space)
    logging.info("[DP-DBSCAN] histogram collect time: %.5g seconds", time.time() - step_timer)

    # add noise
    step_timer = time.time()

    # directly adding noise to sum_hist, which has sensitivity neighbor
    noises = noise_gen.generate(grid_space.num_grids)
    noise_bound = noise_gen.max_sum_noise(beta, noises.shape[0], len(grid_space.neighbor_offsets))

    noisy_counts = NoisyHistogram.build_with_noise(grid_counts, noises)
    noisy_sum = SumHistogram.build_from_counts(noisy_counts, grid_space)

    logging.info("[DP-DBSCAN] add noise time: %.5g seconds", time.time() - step_timer)

    # find superset of core grids and give unique labels
    step_timer = time.time()
    grid_labels = GridLabels.label_high_freq(noisy_sum, min_pts + noise_bound)
    logging.info("[DP-DBSCAN] find core grids time: %.5g seconds", time.time() - step_timer)

    # merge neighboring core-grids by union-find
    step_timer = time.time()
    num_clusters = grid_labels.merge_neighbors(grid_space)
    logging.info("[DP-DBSCAN] merge grids time: %.5g seconds", time.time() - step_timer)
    logging.info("[DP-DBSCAN] time: %.5g seconds", time.time() - timer)
    logging.info("[DP-DBSCAN] num clusters: %d", num_clusters)

    printer.plot_grid(grid_space, labels=GridLabels.label_all(grid_space), hist=noisy_counts, title="noisy_counts")
    printer.plot_grid(grid_space, labels=GridLabels.label_all(grid_space), hist=noisy_sum, title="noisy_sum")
    printer.plot_grid(grid_space, labels=grid_labels, hist=MockHistogram(),
                      title="dp_dbscan_" + str(epsilon) + "_grids")

    if data_provider.has_true_labels():
        point_labels = grid_labels.obtain_point_labels(pts, grid_space)
        printer.plot_labels(labels=point_labels, radius=0., title="dp_dbscan_" + str(epsilon) + "_pts")
        print('[DP-DBSCAN] ARI = ', metrics.adjusted_rand_score(true_labels, point_labels))
        print('[DP-DBSCAN] AMI = ', metrics.adjusted_mutual_info_score(true_labels, point_labels))
    elif not args.skip_dbscan:
        point_labels = grid_labels.obtain_point_labels(pts, grid_space)
        printer.plot_labels(labels=point_labels, radius=0., title="dp_dbscan_" + str(epsilon) + "_pts")
        print('[DP-DBSCAN] ARI(wrt DBSCAN) = ', metrics.adjusted_rand_score(dbscan_labels, point_labels))
        print('[DP-DBSCAN] AMI(wrt DBSCAN) = ', metrics.adjusted_mutual_info_score(dbscan_labels, point_labels))

    if log_level == logging.DEBUG:
        sum_hist = SumHistogram.build_from_counts(grid_counts, grid_space)
        logging.debug("[DP-DBSCAN] max noise seen: %.2f, max noise expected: %.2f, min_pts: %d",
                      max_diff(sum_hist, noisy_sum), noise_bound, min_pts)
        printer.plot_grid(grid_space, labels=GridLabels.label_all(grid_space), hist=sum_hist, title="neighbor_sum")

# k-means
if not args.skip_dp_kmeans:

    timer = time.time()

    kmeans = diffprivlib.models.k_means.KMeans(n_clusters, epsilon=epsilon, bounds=(low, high), random_state=seed)
    kmeans.fit(pts.get())

    logging.info("[DPKmeans] time: %.5g seconds", time.time() - timer)

    printer.plot_centers(scaled_centers=kmeans.cluster_centers_,
                         title="dp_kmeans_" + str(epsilon) + "_" + str(n_clusters) + "_centers")
    printer.plot_labels(labels=kmeans.labels_, radius=0.,
                        title="dp_kmeans_" + str(epsilon) + "_" + str(n_clusters) + "_pts")

    if data_provider.has_true_labels():
        print('[DPKmeans] ARI = ', metrics.adjusted_rand_score(true_labels, kmeans.labels_))
        print('[DPKmeans] AMI = ', metrics.adjusted_mutual_info_score(true_labels, kmeans.labels_))
