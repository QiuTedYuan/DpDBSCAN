import matplotlib.pyplot as plt
import numpy as np

from grid_helper import GridHelper


class Printer:
    def __init__(self):
        self.fig, self.axs = plt.subplots(2, 3, figsize=(15, 10))
        self.fig_count = 0

    def next_ax(self):
        ax = self.axs[self.fig_count // 3, self.fig_count % 3]
        self.fig_count += 1
        return ax

    # plot the points where colors represent the clusters
    # labels = clusters where -1 means no cluster
    # radius = alpha in DBSCAN, for drawing the span
    # colors = colors to choose from
    def plot_points(self, pts: np.ndarray, labels=None, radius=0., colors=None, title=None):
        ax = self.next_ax()
        labels = labels if labels is not None else np.ones(len(pts))
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(x) for x in np.linspace(0, 1, len(unique_labels))]

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

    def plotGrid(self, grids: GridHelper, labels, hist=None, title=None):
        ax = self.next_ax()
        hist = hist if hist is not None else np.ones(grids.num_grids)

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        max_hist = max(abs(x) for x in hist)

        for k, col in zip(unique_labels, colors):
            if k == -1:
                continue

            class_index = np.where(labels == k)[0]
            for ci in class_index:
                pt_x, pt_y = grids.idx2loc_bot_left(ci)
                # ax.text(pt_x, pt_y, round(hist[ci]), fontsize="xx-small")
                facecol = col if (hist[ci] > 0) else "green"
                rectangle = plt.Rectangle((pt_x, pt_y), grids.w, grids.w,
                                          edgecolor="black", facecolor=facecol, fill=True,
                                          alpha=abs(hist[ci]) / max_hist)
                ax.add_patch(rectangle)
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()

    def display(self, u, file=None):
        plt.setp(self.axs, xlim=[-0.1 * u, 1.1 * u], ylim=[-0.1 * u, 1.1 * u])
        plt.savefig("./fig/"+file)
        plt.show()
