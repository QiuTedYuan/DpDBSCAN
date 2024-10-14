import argparse
import logging
import os
import time
from datetime import datetime

import diffprivlib
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances

from data_provider import get_data
from datatype_grid import GridSpace
from datatype_point import Points
from histogram import Histogram, SumHistogram, NoisyHistogram, max_diff, GridLabels, MockHistogram
from noise import *
from plot_data import Printer, PrinterParams

# program args
parser = argparse.ArgumentParser(description='DPDBSCAN Experiments.')
parser.add_argument('-d', '--dataset', help="Dataset, default moons", required=False,
                    choices=['moons', 'blobs', 'circles', 'cluto_t4', 'cluto_t5', 'cluto_t7', 'cluto_t8',
                             'cabs', 'cabs_tiny', 'crash', 'har'],
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
parser.add_argument('--dpi', help="figure dpi",
                    required=False, default=100, type=float)
parser.add_argument('--ext', help="figure extension",
                    required=False, choices=["png", "pdf", "svg"], default="png")
parser.add_argument('--debug', help="log level debug",
                    action='store_true')
parser.add_argument('--info', help="log level info",
                    action='store_true')
parser.add_argument('--linear', help="force using O(n) time histogram",
                    action='store_true')
parser.add_argument('--naive', help="force using O(|X|) time histogram",
                    action='store_true')

parser.add_argument('--skip_dbscan', action='store_true')
parser.add_argument('--skip_kmeans', action='store_true')
parser.add_argument('--skip_dp_dbscan', action='store_true')
parser.add_argument('--skip_dp_kmeans', action='store_true')
parser.add_argument('--run_trivial', action='store_true')

args = parser.parse_args()

# prepare data
data_provider = get_data(args.dataset)
printer_params = PrinterParams.from_data(data_provider, args.dpi, args.ext)

pts = data_provider.get_data()
n, dim = pts.get_size(), pts.get_dim()
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

seed, epsilon, delta, beta = args.seed, args.epsilon, args.delta, args.beta
noise_gen = get_noise_gen(args.noise, args.seed, 1,  args.epsilon, args.delta)

output_folder = "./output/" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S-%f") + "/"
if args.plot:
    os.makedirs(output_folder, exist_ok=True)
    logging.info("Created directory: " + output_folder)
printer = Printer(data_provider, output_folder, args.plot, printer_params)

if data_provider.has_true_labels():
    true_labels = data_provider.get_true_labels()
    true_label_set = set(true_labels)
    true_label_set.discard(-1)
    n_clusters = len(true_label_set)
    printer.plot_labels(labels=true_labels, radius=0., title="true_cluster")
else:
    true_labels = np.arange(0, n)
    n_clusters = 3

logging.info('[Dataset] n=%d, d=%d, low=%s, high=%s, alpha=%.5g, min_pts=%d, num_cluster:%d',
             n, dim, low, high, alpha, min_pts, n_clusters)

RUN_DBSCAN = not args.skip_dbscan
RUN_DP_DBSCAN = not args.skip_dp_dbscan
RUN_KMEANS = not args.skip_kmeans
RUN_DP_KMEANS = not args.skip_dp_kmeans
RUN_TRIVIAL = args.run_trivial

if RUN_DBSCAN:

    timer = time.time()
    dbs = DBSCAN(eps=alpha, min_samples=min_pts)
    dbs.fit(pts.get())
    dbscan_labels = Points.compute_dbscan_labels(dbs)
    logging.info("[DBSCAN][Time] %.5g seconds", time.time() - timer)

    dbscan_label_set = set(dbscan_labels)
    dbscan_label_set.discard(-1)
    logging.info("[DBSCAN] num clusters: %d", len(dbscan_label_set))
    printer.plot_labels(labels=dbscan_labels, radius=alpha, title="dbscan_" + str(alpha) + "_" + str(min_pts))
    if data_provider.has_true_labels():
        print('[DBSCAN] ARI = ', metrics.adjusted_rand_score(true_labels, dbscan_labels))
        print('[DBSCAN] AMI = ', metrics.adjusted_mutual_info_score(true_labels, dbscan_labels))

if RUN_KMEANS:

    timer = time.time()
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pts.get())
    logging.info("[KMeans][Time] %.5g seconds", time.time() - timer)

    printer.plot_centers(scaled_centers=kmeans.cluster_centers_,
                         title="kmeans_" + str(n_clusters) + "_centers")
    printer.plot_labels(labels=kmeans.labels_, radius=0., title="kmeans_" + str(n_clusters))
    if data_provider.has_true_labels():
        print('[KMeans] ARI = ', metrics.adjusted_rand_score(true_labels, kmeans.labels_))
        print('[KMeans] AMI = ', metrics.adjusted_mutual_info_score(true_labels, kmeans.labels_))

if RUN_DP_DBSCAN:

    timer = time.time()

    # build grids
    step_timer = time.time()
    grid_space = GridSpace(dim, low, high, alpha, grid_scale)
    num_grids = grid_space.num_grids
    logging.info("[DP-DBSCAN] cell width: %.5g, num cells: %d, kappa: %d", grid_space.width, num_grids, grid_space.kappa)

    grid_counts = Histogram.build_from_pts(pts, grid_space)
    logging.info("[DP-DBSCAN][Time] collect histogram: %.5g seconds", time.time() - step_timer)

    # add noise
    step_timer = time.time()
    linear: bool = n <= num_grids / 2.
    if args.linear:
        linear = True
    elif args.naive:
        linear = False

    if linear:
        noise_bound = noise_gen.max_sum_noise(beta, num_grids, grid_space.kappa)
        if num_grids > 2. * n:
            T = 1 / epsilon * np.log(num_grids / n)
        else:
            T = noise_bound / grid_space.kappa
        noise_bound += T * grid_space.kappa
        logging.info("[DP-DBSCAN][Linear] threshold T: %.3g, noise_bound: %.3g", T, noise_bound)
        noisy_counts = NoisyHistogram.linear_time_build(grid_counts, noise_gen, num_grids, T)
        logging.info("[DP-DBSCAN][Time] noisy histogram O(n): %.5g seconds", time.time() - step_timer)
    else:
        noisy_counts = NoisyHistogram.naive_build(grid_counts, noise_gen, num_grids)
        noise_bound = noise_gen.max_sum_noise(beta, num_grids, grid_space.kappa)
        logging.info("[DP-DBSCAN][Naive] noise_bound: %.3g",  noise_bound)
        logging.info("[DP-DBSCAN][Time] noisy histogram O(|X|): %.5g seconds", time.time() - step_timer)

    # add noise
    step_timer = time.time()
    noisy_sum = SumHistogram.build_from_counts(noisy_counts, grid_space)
    logging.info("[DP-DBSCAN][Time] neighbor sum: %.5g seconds", time.time() - step_timer)

    # find superset of core grids and give unique labels
    step_timer = time.time()
    grid_labels = GridLabels.label_high_freq(noisy_sum, min_pts + noise_bound)
    logging.info("[DP-DBSCAN][Time] core cells: %.5g seconds", time.time() - step_timer)

    # merge neighboring core-grids by union-find
    step_timer = time.time()
    num_clusters = grid_labels.merge_neighbors(grid_space)
    logging.info("[DP-DBSCAN][Time] merge cells: %.5g seconds", time.time() - step_timer)

    logging.info("[DP-DBSCAN][Time] %.5g seconds", time.time() - timer)
    logging.info("[DP-DBSCAN] num clusters: %d", num_clusters)

    #printer.plot_hist(grid_space, hist=noisy_counts, title="noisy_counts")
    #printer.plot_hist(grid_space, hist=noisy_sum, title="noisy_sum")
    printer.plot_grid(grid_space, labels=grid_labels, title="dp_dbscan_" + str(epsilon) + "_grids")

    if data_provider.has_true_labels():
        point_labels = grid_labels.obtain_point_labels(pts, grid_space)
        printer.plot_labels(labels=point_labels, radius=0., title="dp_dbscan_" + str(epsilon) + "_pts")
        print('[DP-DBSCAN] ARI = ', metrics.adjusted_rand_score(true_labels, point_labels))
        print('[DP-DBSCAN] AMI = ', metrics.adjusted_mutual_info_score(true_labels, point_labels))
    elif RUN_DBSCAN:
        point_labels = grid_labels.obtain_point_labels(pts, grid_space)
        printer.plot_labels(labels=point_labels, radius=0., title="dp_dbscan_" + str(epsilon) + "_pts")
        print('[DP-DBSCAN] ARI(wrt DBSCAN) = ', metrics.adjusted_rand_score(dbscan_labels, point_labels))
        print('[DP-DBSCAN] AMI(wrt DBSCAN) = ', metrics.adjusted_mutual_info_score(dbscan_labels, point_labels))

    if log_level == logging.DEBUG:
        sum_hist = SumHistogram.build_from_counts(grid_counts, grid_space)
        logging.debug("[DP-DBSCAN] max noise seen: %.2f, max noise expected: %.2f, min_pts: %d",
                      max_diff(sum_hist, noisy_sum), noise_bound, min_pts)
        printer.plot_hist(grid_space, hist=sum_hist, title="neighbor_sum")

# k-means
if RUN_DP_KMEANS:

    timer = time.time()

    kmeans = diffprivlib.models.k_means.KMeans(n_clusters, epsilon=epsilon, bounds=(low, high), random_state=seed)
    kmeans.fit(pts.get())

    logging.info("[DPKmeans][Time] %.5g seconds", time.time() - timer)

    printer.plot_centers(scaled_centers=kmeans.cluster_centers_,
                         title="dp_kmeans_" + str(epsilon) + "_" + str(n_clusters) + "_centers")
    printer.plot_labels(labels=kmeans.labels_, radius=0.,
                        title="dp_kmeans_" + str(epsilon) + "_" + str(n_clusters) + "_pts")

    if data_provider.has_true_labels():
        print('[DPKmeans] ARI = ', metrics.adjusted_rand_score(true_labels, kmeans.labels_))
        print('[DPKmeans] AMI = ', metrics.adjusted_mutual_info_score(true_labels, kmeans.labels_))

# trivial
if RUN_TRIVIAL:
    timer = time.time()

    pw_dist = pairwise_distances(pts.get())
    sensitivity = Points.dist(low, high)
    # changing a point affects n distances
    noise_gen = get_noise_gen(args.noise, args.seed, n, args.epsilon, args.delta)
    noises = noise_gen.generate((n + 1) * n // 2)
    idx = 0
    for i in range(0, n):
        for j in range(i+1, n):
            noisy_dist = min(max(pw_dist[i, j] + noises[idx], 0), sensitivity)
            idx += 1
            pw_dist[i, j] = noisy_dist
            pw_dist[j, i] = noisy_dist

    dbs_trivial = DBSCAN(eps=alpha, min_samples=min_pts, metric='precomputed')
    dbs_trivial.fit(pw_dist)
    trivial_labels = Points.compute_dbscan_labels(dbs_trivial)
    logging.info("[Trivial][Time] %.5g seconds", time.time() - timer)

    trivial_label_set = set(trivial_labels)
    trivial_label_set.discard(-1)
    logging.info("[Trivial] num clusters: %d", len(trivial_label_set))
    printer.plot_labels(labels=trivial_labels, radius=alpha, title="trivial_" + str(alpha) + "_" + str(min_pts))
    if data_provider.has_true_labels():
        print('[Trivial] ARI = ', metrics.adjusted_rand_score(true_labels, trivial_labels))
        print('[Trivial] AMI = ', metrics.adjusted_mutual_info_score(true_labels, trivial_labels))
