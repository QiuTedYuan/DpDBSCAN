import matplotlib.pyplot as plt
import numpy as np
from shapely import MultiPoint
from shapely.geometry import box
from shapely.ops import voronoi_diagram
from shapely.plotting import plot_polygon

from datatype_grid import GridSpace
from datatype_point import Points, PointLabels
from histogram import Histogram, GridLabels


class Printer:
    def __init__(self, pts: Points, output_folder: str, enabled=True, enable_label=True, figsize=(10,6), draw_edge=True):
        self.output_folder = output_folder
        self.dim = pts.get_dim()
        self.skip = (self.dim > 3) or not enabled
        self.skip_label = not enable_label
        self.figsize = figsize
        self.draw_edge = draw_edge
        if not self.skip:
            low, high = pts.get_ranges()
            x_offset = 0.05 * (high[0] - low[0])
            y_offset = 0.05 * (high[1] - low[1])
            self.xlim = [low[0] - x_offset, high[0] + x_offset]
            self.ylim = [low[1] - y_offset, high[1] + y_offset]
            self.pts = pts

    def save_fig(self, ax, file):
        if self.skip:
            return
        plt.setp(ax, xlim=self.xlim, ylim=self.ylim)
        plt.tight_layout()
        plt.savefig(self.output_folder + file + ".png", dpi=1200)

    @staticmethod
    def reverse_color(c: np.array):
        return np.array([1. - c[0], 1. - c[1], 1. - c[2], c[3]])

    @staticmethod
    def get_color_map(unique_labels: set[int]):
        unique_labels.discard(-1)
        colors = plt.cm.get_cmap("Spectral")(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, colors))
        color_map[-1] = [0, 0, 0, 1]
        return color_map

    def plot_3d_labels(self, labels: PointLabels, title):
        _, ax = plt.subplot(projection='3d')
        color_map = Printer.get_color_map(set(labels))
        for pt, label in zip(self.pts.get(), labels):
            color = color_map[label]
            ax.scatter(pt[0], pt[1], zs=pt[2],
                       marker="x" if label == -1 else "o",
                       edgecolor="black" if self.draw_edge else color,
                       facecolor=color, )
        # if title is not None:
        #     ax.set_title(title)
        self.save_fig(ax, title)

    # plot the points where colors represent the clusters
    # labels = clusters where -1 means no cluster
    # radius = alpha in DBSCAN, for drawing the span
    def plot_labels(self, labels: PointLabels, radius, title: str):
        if self.skip or self.skip_label:
            return
        if self.dim == 3:
            return self.plot_3d_labels(labels, title)

        _, ax = plt.subplots(figsize=self.figsize)
        color_map = Printer.get_color_map(set(labels))
        for pt, label in zip(self.pts.get(), labels):
            color = color_map[label]
            ax.plot(pt[0], pt[1],
                    "x" if label == -1 else "o",
                    markerfacecolor=color,
                    markeredgecolor="black" if self.draw_edge else color,
                    markersize=4, )

            # plot spans for core points
            if radius > 0 and -1 != label:
                cir = plt.Circle(pt, radius, color=color, fill=True, alpha=0.3)
                ax.add_patch(cir)
        # if title is not None:
        #     ax.set_title(title)
        self.save_fig(ax, title)

    def plot_3d_grid(self, grid_space: GridSpace, labels: GridLabels, hist: Histogram, title):
        _, ax = plt.subplot(projection='3d')
        color_map = Printer.get_color_map(set(labels.values()))
        max_freq = hist.max_freq()

        for key, label in labels.items():
            if label == -1:
                continue
            freq = hist.get_by_key(key)
            x, y, z = grid_space.get_low_point_of_grid(grid_space.decode_from_key(key))
            color = color_map[label]
            ax.scatter(x, y, zs=z,
                       edgecolor="black" if self.draw_edge else color,
                       facecolor=color,
                       alpha=abs(freq) / max_freq)
        # if title is not None:
        #     ax.set_title(title)
        self.save_fig(ax, title)

    def plot_grid(self, grid_space: GridSpace, labels: GridLabels, hist: Histogram, title: str):
        if self.skip:
            return
        if self.dim == 3:
            return self.plot_3d_grid(grid_space, labels, hist, title)

        _, ax = plt.subplots(figsize=self.figsize)
        color_map = Printer.get_color_map(set(labels.values()))
        max_freq = hist.max_freq()
        for key, label in labels.items():
            if label == -1:
                continue

            freq = hist.get_by_key(key)
            if freq == 0:
                continue

            x, y = grid_space.get_low_point_of_grid(grid_space.decode_from_key(key))
            x_high, y_high = grid_space.high
            color = color_map[label] if freq >= 0 else self.reverse_color(color_map[label])
            x_width = min(grid_space.width, x_high - x)
            y_width = min(grid_space.width, y_high - y)

            rectangle = plt.Rectangle((x, y), x_width, y_width,
                                      edgecolor="black" if self.draw_edge else color,
                                      facecolor=color,
                                      fill=True,
                                      alpha=abs(freq) / max_freq)
            ax.add_patch(rectangle)
        self.save_fig(ax, title)

    def plot_centers(self, centers, title: str):
        if self.skip or self.dim == 3:
            return

        _, ax = plt.subplots(figsize=self.figsize)
        plt.setp(ax, xlim=self.xlim, ylim=self.ylim)
        color_map = Printer.get_color_map(set(range(len(centers))))
        regions = voronoi_diagram(geom=MultiPoint(centers), envelope=box(self.xlim[0], self.ylim[0], self.xlim[1], self.ylim[1]))
        idx = 0
        for region in regions.geoms:
            color = None
            for idx, center in enumerate(centers):
                if region.intersects(MultiPoint([center])):
                    color = color_map[idx]
                    break
            plot_polygon(region, ax=ax, add_points=False, color=color)
            idx += 1

        for label, center in enumerate(centers):
            color = color_map[label]
            ax.plot(center[0], center[1],
                    "o",
                    markerfacecolor=color,
                    markeredgecolor="black" if self.draw_edge else color,
                    markersize=4, )
        self.save_fig(ax, title)
