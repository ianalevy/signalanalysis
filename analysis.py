from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyqtgraph as pg
from scipy import stats
from scipy.stats import ks_1samp


@dataclass
class HistogramResults:
    bins: np.ndarray
    counts: np.ndarray
    centers: np.ndarray | None = None

    def __post_init__(self):
        if self.centers is None:
            self.centers = (self.bins[1:] + self.bins[:-1]) / 2
        raw_cdf = np.cumsum(self.counts)
        self.cdf = raw_cdf / raw_cdf[-1]

    def interp_cdf(self, x) -> Callable:
        return np.interp(x, self.centers, self.cdf)

    def sample_with_search(self, size: int = 1, seed: int = 42) -> np.ndarray:
        """Sample from the distribution using a search algorithm.

        Parameters
        ----------
        size : int, optional
            _description_, by default 1
        seed : int, optional
            _description_, by default 42

        Returns
        -------
        np.ndarray

        """
        rng = np.random.default_rng(seed=seed)
        values = rng.random(size=size)
        value_bins = np.searchsorted(self.cdf, values)
        return self.centers[value_bins]

    def sample(self, size: int = 1, seed: int = 42) -> np.ndarray:
        """Sample using numpy choice function.

        Parameters
        ----------
        size : int, optional
            _description_, by default 1
        seed : int, optional
            _description_, by default 42

        Returns
        -------
        np.ndarray
            _description_

        """
        rng = np.random.default_rng(seed=seed)
        return rng.choice(
            self.centers,
            size=size,
            p=self.counts / np.sum(self.counts, dtype=float),
        )


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
