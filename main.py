from sklearn.cluster import DBSCAN
from unionfind import unionfind

from gen_data import *
from plot_data import *


# generate data
n_samples = 1000
random_state = 0
dim = 2
univ = 1
gen = Generator(n_samples, random_state)
pts, labels_true = gen.generate_blobs(univ)
printer = Printer(dim)

# exact DBSCAN
alpha = 0.06
min_pts = 15
dbs = DBSCAN(eps=alpha, min_samples=min_pts)
dbs.fit(pts)
printer.plot_points(pts=pts,
                    labels=dbs.labels_,
                    radius=0.,
                    title="DBSCAN (" + str(alpha) + ", " + str(min_pts) + ")-clusters")
printer.plot_points(pts=pts,
                    labels=dbs.labels_,
                    radius=alpha,
                    title="Real cluster spans")

# build grids
s = 0.8
grid_helper = GridHelper(dim, univ, alpha, s)
num_grids = grid_helper.num_grids
grid_counts = grid_helper.count_points(pts)
sum_hist = grid_helper.compute_upper_bound_hist(grid_counts)
printer.plot_grid(grid_helper, labels=np.array([int(x > 0) - 1 for x in sum_hist]), hist=sum_hist,
                  title="Neighbor Sums")

# noisy grids
eps = 1
np.random.seed(0)
noises = np.random.laplace(0, 1 / eps, num_grids)
print("noise range: [", min(noises), ",", max(noises), "]")
noisy_counts = np.add(grid_counts, noises)
print("noisy count range: [", min(noisy_counts), ",", max(noisy_counts), "]")
noisy_sum_hist = grid_helper.compute_upper_bound_hist(noisy_counts)
print("noisy sum range: [", min(noisy_sum_hist), ",", max(noisy_sum_hist), "]")
printer.plot_grid(grid_helper, labels=np.ones(num_grids, dtype=int), hist=noisy_sum_hist, title="Noisy Sums")

# find superset of core grids and give unique labels
core_grid_count = 0
grid_labels = np.zeros(num_grids, dtype=int)
tau = 2 / eps * math.sqrt(len(grid_helper.neighbor_offsets))
for i in range(num_grids):
    if noisy_sum_hist[i] >= min_pts + tau:
        grid_labels[i] = core_grid_count
        core_grid_count += 1
    else:
        grid_labels[i] = -1
printer.plot_grid(grid_helper,
                  labels=grid_labels,
                  hist=noisy_sum_hist,
                  title="Core Grids")

# merge neighboring core-grids by union-find
uf = unionfind(num_grids)
for current_idx in range(num_grids):
    if grid_labels[current_idx] == -1:
        continue
    current_coord: GridCoord = grid_helper.get_coord(current_idx)
    for offset in grid_helper.neighbor_offsets:
        neighbor_coord: GridCoord = np.add(current_coord, offset)
        neighbor_index = grid_helper.get_index(neighbor_coord)
        if grid_helper.is_valid(neighbor_coord) and grid_labels[neighbor_index] != -1:
            uf.unite(current_idx, neighbor_index)

# normalize
unique_labels = {}
current_label = 0
for idx in range(num_grids):
    if grid_labels[idx] != -1:
        uf_label = uf.find(idx)
        if uf_label not in unique_labels:
            unique_labels[uf_label] = current_label
            current_label += 1
        grid_labels[idx] = unique_labels[uf_label]
printer.plot_grid(grid_helper, labels=grid_labels, hist=np.ones(num_grids, dtype=int), title="(1,0)-DP Clusters")

# finally
printer.display(univ, file="algoBlob.pdf")
