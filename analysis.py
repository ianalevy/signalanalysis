from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyqtgraph as pg


@dataclass
class HistogramResults:
    bins: np.ndarray
    counts: np.ndarray
    centers: np.ndarray | None = None

    def __post_init__(self):
        if self.centers is None:
            self.centers = (self.bins[1:] + self.bins[:-1]) / 2

    def interp_distr(self) -> Callable:
        return np.interp(self.centers, self.counts)


def compute_histogram(
    data: np.ndarray,
    bins: int = 10,
) -> HistogramResults:
    counts, edges = np.histogram(data, bins=bins)

    return HistogramResults(edges, counts)


def plot_hist(win, hist: HistogramResults):
    win.resize(800, 480)
    win.setWindowTitle("Histogram")
    plt = win.addPlot()

    bgi = pg.BarGraphItem(
        x0=hist.bins[:-1],
        x1=hist.bins[1:],
        height=hist.counts,
        pen="w",
        brush=(0, 0, 255, 150),
    )
    plt.addItem(bgi)


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    data = rng.normal(loc=3, size=1000)
    hist = compute_histogram(data, bins=50)

    win = pg.GraphicsLayoutWidget(show=True)
    plot_hist(win, hist)

    pg.exec()
