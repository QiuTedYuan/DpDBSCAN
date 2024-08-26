import matplotlib.pyplot as plt

from data_types import *


class Printer:
    def __init__(self, dim: int):
        self.skip = (dim != 2)
        self.fig, self.axs = plt.subplots(2, 3, figsize=(15, 10))
        self.fig_count = 0

    def next_ax(self):
        ax = self.axs[self.fig_count // 3, self.fig_count % 3]
        self.fig_count += 1
        return ax

    @staticmethod
    def reverse_color(c: np.array):
        return np.array([1. - c[0], 1. - c[1], 1. - c[2], c[3]])

    @staticmethod
    def get_colors(cnt: int):
        return plt.cm.get_cmap("Spectral")(np.linspace(0, 1, cnt))

    # plot the points where colors represent the clusters
    # labels = clusters where -1 means no cluster
    # radius = alpha in DBSCAN, for drawing the span
    # colors = colors to choose from
    def plot_points(self, pts: Points, labels: Labels = None, radius=0., colors=None, title=None):
        if self.skip:
            return
        ax = self.next_ax()
        labels = labels if labels is not None else np.ones(len(pts), dtype=int)
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = colors if colors is not None else self.get_colors(len(unique_labels))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Black used for noise.
                color = [0, 0, 0, 1]

            label_points = np.where(labels == label)[0]
            for idx in label_points:
                ax.plot(pts[idx][0], pts[idx][1],
                        "x" if label == -1 else "o",
                        markerfacecolor=color,
                        markeredgecolor="black",
                        markersize=8, )

            # plot spans for core points
            if radius > 0 and -1 != label:
                for idx in label_points:
                    cir = plt.Circle(pts[idx],
                                     radius,
                                     color=color,
                                     fill=True,
                                     alpha=0.3)
                    ax.add_patch(cir)
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()

    def plot_grid(self, grid_helper: GridHelper, labels: Labels, hist: Histogram = None, title=None):
        if self.skip:
            return
        ax = self.next_ax()
        hist = hist if hist is not None else np.ones(grid_helper.num_grids, dtype=float)
        unique_labels = set(labels)
        colors = self.get_colors(len(unique_labels))

        max_hist = max(abs(x) for x in hist)

        for k, col in zip(unique_labels, colors):
            if k == -1:
                continue

            class_index = np.where(labels == k)[0]
            for idx in class_index:
                low_point = [x * grid_helper.width for x in grid_helper.get_coord(idx)]
                x = low_point[0]
                y = low_point[1]
                x_w = min(grid_helper.width, grid_helper.univ - x)
                y_w = min(grid_helper.width, grid_helper.univ - y)
                freq = hist[idx]
                facecol = col if freq >= 0 else self.reverse_color(col)
                rectangle = plt.Rectangle((x, y), x_w, y_w,
                                          edgecolor="black",
                                          facecolor=facecol,
                                          fill=True,
                                          alpha=abs(freq) / max_hist)
                ax.add_patch(rectangle)
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()

    def display(self, u, file=None):
        if self.skip:
            return
        plt.setp(self.axs, xlim=[-0.1 * u, 1.1 * u], ylim=[-0.1 * u, 1.1 * u])
        plt.savefig("./fig/" + file)
        plt.show()
