import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from shapely import MultiPoint
from shapely.geometry import box
from shapely.ops import voronoi_diagram
from shapely.plotting import plot_polygon

from data_provider import DataProvider, ArffDataProvider, LongitudeLatitudeDataProvider, HighDimDataProvider
from datatype_grid import GridSpace
from datatype_point import Points, PointLabels, Point
from histogram import Histogram, GridLabels


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

    @classmethod
    def from_data(cls, data_provider: DataProvider, dpi, ext):
        if isinstance(data_provider, ArffDataProvider):
            return cls(dpi, ext, fig_size=(10, 6))
        elif isinstance(data_provider, LongitudeLatitudeDataProvider):
            return cls(dpi, ext, draw_label=False, draw_edge=False)
        elif isinstance(data_provider, HighDimDataProvider):
            return cls(dpi, ext, draw_label=True, draw_edge=False)
        else:
            return cls(dpi, ext)


class Printer:
    def __init__(self, data_provider: DataProvider, output_folder: str, enabled: bool, params: PrinterParams):
        self.output_folder = output_folder
        self.pts = data_provider.get_data()
        self.scales = data_provider.get_scales()
        self.dim = self.pts.get_dim()
        self.skip = (self.dim > 3) or not enabled
        self.params = params
        matplotlib.rcParams.update({'font.size': params.font_size})
        if not self.skip:
            scaled_low, scaled_high = self.pts.get_ranges()
            low = self.get_original_pt(scaled_low)
            high = self.get_original_pt(scaled_high)
            x_offset = 0.05 * (high[0] - low[0])
            y_offset = 0.05 * (high[1] - low[1])
            self.xlim = [low[0] - x_offset, high[0] + x_offset]
            self.ylim = [low[1] - y_offset, high[1] + y_offset]

    def get_original_pt(self, pt: Point):
        return np.true_divide(pt, self.scales)

    def save_fig(self, ax, file):
        if self.skip:
            return
        plt.setp(ax, xlim=self.xlim, ylim=self.ylim)
        plt.tight_layout()
        plt.savefig(self.output_folder + file + "." + self.params.ext, dpi=self.params.dpi)

    @staticmethod
    def reverse_color(c: np.array):
        return np.array([1. - c[0], 1. - c[1], 1. - c[2], c[3]])

    def edge_color(self, color):
        return "black" if self.params.draw_edge else color

    @staticmethod
    def get_marker(label):
        return "x" if label == -1 else "o"

    @staticmethod
    def get_color_map(unique_labels: set[int]):
        unique_labels.discard(-1)
        colors = plt.cm.get_cmap("Spectral")(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, colors))
        color_map[-1] = [0, 0, 0, 1]
        return color_map

    def plot_3d_labels(self, labels: PointLabels, title):
        _ = plt.figure(figsize=self.params.fig_size)
        ax = plt.axes(projection='3d')
        ax.grid()
        color_map = Printer.get_color_map(set(labels))
        for scaled_pt, label in zip(self.pts.get(), labels):
            color = color_map[label]
            pt = self.get_original_pt(scaled_pt)
            ax.scatter(pt[0], pt[1], zs=pt[2],
                       marker=self.get_marker(label),
                       facecolor=color, )
        self.save_fig(ax, title)

    # plot the points where colors represent the clusters
    # labels = clusters where -1 means no cluster
    # radius = alpha in DBSCAN, for drawing the span
    def plot_labels(self, labels: PointLabels, radius, title: str):
        if self.skip or not self.params.draw_label:
            return
        if self.dim == 3:
            return self.plot_3d_labels(labels, title)

        _, ax = plt.subplots(figsize=self.params.fig_size)
        color_map = Printer.get_color_map(set(labels))
        for scaled_pt, label in zip(self.pts.get(), labels):
            color = color_map[label]
            pt = self.get_original_pt(scaled_pt)
            ax.plot(pt[0], pt[1],
                    self.get_marker(label),
                    markerfacecolor=color,
                    markeredgecolor=self.edge_color(color),
                    markersize=self.params.marker_size, )

            # plot spans for core points
            if radius > 0 and -1 != label:
                cir = plt.Circle(pt, radius, color=color, fill=True, alpha=0.3)
                ax.add_patch(cir)

        self.save_fig(ax, title)

    def plot_3d_grid(self, grid_space: GridSpace, labels: GridLabels, title):
        _ = plt.figure(figsize=self.params.fig_size)
        ax = plt.axes(projection='3d')
        ax.grid()
        color_map = Printer.get_color_map(set(labels.values()))

        for key, label in labels.items():
            if label == -1:
                continue
            x, y, z = self.get_original_pt(grid_space.get_low_point_of_grid(grid_space.decode_from_key(key)))
            color = color_map[label]
            ax.scatter(x, y, zs=z,
                       facecolor=color)
        self.save_fig(ax, title)

    def plot_grid(self, grid_space: GridSpace, labels: GridLabels, title: str):
        if self.skip:
            return
        if self.dim == 3:
            return self.plot_3d_grid(grid_space, labels, title)

        _, ax = plt.subplots(figsize=self.params.fig_size)
        color_map = Printer.get_color_map(set(labels.values()))
        for key, label in labels.items():
            if label == -1:
                continue

            x, y = self.get_original_pt(grid_space.get_low_point_of_grid(grid_space.decode_from_key(key)))
            x_high, y_high = self.get_original_pt(grid_space.high)
            color = color_map[label]
            x_width = min(grid_space.width / self.scales[0], x_high - x)
            y_width = min(grid_space.width / self.scales[1], y_high - y)

            rectangle = plt.Rectangle((x, y), x_width, y_width,
                                      edgecolor=self.edge_color(color),
                                      facecolor=color,
                                      fill=True,)
            ax.add_patch(rectangle)
        self.save_fig(ax, title)

    def plot_hist(self, grid_space: GridSpace, hist: Histogram, title: str):
        if self.skip or self.dim >= 3:
            return

        _, ax = plt.subplots(figsize=self.params.fig_size)
        color = Printer.get_color_map({0})[0]
        max_freq = hist.max_freq()
        for key, freq in hist.items():
            freq = hist.get_by_key(key)
            if freq == 0:
                continue

            x, y = self.get_original_pt(grid_space.get_low_point_of_grid(grid_space.decode_from_key(key)))
            x_high, y_high = self.get_original_pt(grid_space.high)
            x_width = min(grid_space.width / self.scales[0], x_high - x)
            y_width = min(grid_space.width / self.scales[1], y_high - y)

            rectangle = plt.Rectangle((x, y), x_width, y_width,
                                      edgecolor=self.edge_color(color),
                                      facecolor=color if freq >= 0 else self.reverse_color(color),
                                      fill=True,
                                      alpha=abs(freq) / max_freq)
            ax.add_patch(rectangle)
        self.save_fig(ax, title)

    def plot_centers(self, scaled_centers, title: str):
        if self.skip or self.dim == 3:
            return

        centers = [self.get_original_pt(center) for center in scaled_centers]

        _, ax = plt.subplots(figsize=self.params.fig_size)
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
                    markeredgecolor=self.edge_color(color),
                    markersize=self.params.marker_size, )
        self.save_fig(ax, title)
