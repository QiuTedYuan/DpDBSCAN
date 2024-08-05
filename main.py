import math

from sklearn.cluster import DBSCAN

from gen_data import *
from grid_helper import GridHelper
from plot_data import *
from unionfind import unionfind

printer = Printer()

# generate data
n_samples = 1000
random_state = 0
u = 1
gen = Generator(n_samples, random_state)
pts, labels = gen.generate_moons(u)

# exact DBSCAN
alpha = 0.05
min_pts = 12
dbs = DBSCAN(eps=alpha, min_samples=min_pts)
dbs.fit(pts)

# plot spans
printer.plot_points(pts, dbs.labels_, title="DBSCAN (" + str(alpha) + ", " + str(min_pts) + ")-clusters")

# build grids
eta = 0.8
grids = GridHelper(u, alpha, eta)
grid_counts = np.zeros(grids.num_grids)
for x in pts:
    grid_x, grid_y = grids.loc2grid(x[0], x[1])
    grid_idx = grids.grid2idx(grid_x, grid_y)
    grid_counts[grid_idx] += 1
fake_labels = np.zeros(grids.num_grids)
for idx in range(grids.num_grids):
    if grid_counts[idx] == 0:
        fake_labels[idx] = -1
sum_hist = grids.get_neighbor_sum_hist(grid_counts)
printer.plotGrid(grids, fake_labels, sum_hist, title="Neighbor Sums")

# noisy grids
eps = 1
np.random.seed(0)
noises = np.random.laplace(0, 1 / eps, grids.num_grids)
max_noise = max(abs(x) for x in noises)
print("max_noise:", max_noise)
noisy_counts = grid_counts
for x in range(grids.num_grids):
    noisy_counts[x] += noises[x]
noisy_sum_hist = grids.get_neighbor_sum_hist(noisy_counts)
printer.plotGrid(grids, np.ones(grids.num_grids), noisy_sum_hist, title="Noisy Sums")

# find superset of core grids
current_cluster = 0
ids = np.zeros(grids.num_grids)
tau = 2 / eps * math.sqrt(len(grids.neighbor_offsets))
for i in range(grids.num_grids):
    if noisy_sum_hist[i] >= min_pts + tau:
        ids[i] = current_cluster
        current_cluster += 1
    else:
        ids[i] = -1
printer.plotGrid(grids, ids, noisy_sum_hist, title="Core Grids")

uf = unionfind(grids.num_grids)
for idx in range(grids.num_grids):
    if ids[idx] == -1:
        continue
    x, y = grids.idx2grid(idx)
    for offset in grids.neighbor_offsets:
        x_ = x + offset[0]
        y_ = y + offset[1]
        idx_ = grids.grid2idx(x_, y_)
        if not grids.is_valid_grid(x_, y_) or ids[idx_] == -1:
            continue
        uf.unite(idx, idx_)

for idx in range(grids.num_grids):
    if ids[idx] != -1:
        ids[idx] = uf.find(idx)
printer.plotGrid(grids, ids, np.ones(grids.num_grids), title="DP Clusters")

# finally
printer.plot_points(pts, dbs.labels_, radius=alpha, title="Real Cluster spans")
printer.display(u, file="algoMoons.pdf")
