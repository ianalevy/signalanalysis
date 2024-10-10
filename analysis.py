from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyqtgraph as pg
from scipy import stats
from scipy.stats import ks_1samp


@dataclass
class KSTestResult:
    statistic: float
    pvalue: float
    statistic_location: float
    statistic_sign: int


@dataclass
class HistogramResults:
    bins: np.ndarray
    counts: np.ndarray

    @property
    def centers(self) -> np.ndarray:
        return (self.bins[1:] + self.bins[:-1]) / 2

    @property
    def cdf(self) -> np.ndarray:
        raw_cdf = np.cumsum(self.counts)
        return raw_cdf / raw_cdf[-1]

    def interp_cdf(self, x) -> Callable:
        return np.interp(x, self.bins[1:], self.cdf)

    def sample_with_search(
        self,
        size: int = 1,
        rng=np.random.default_rng(seed=42),
    ) -> np.ndarray:
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
        values = rng.random(size=size)
        value_bins = np.searchsorted(self.cdf, values)
        return self.centers[value_bins]

    def sample(self, size: int = 1, rng=np.random.default_rng(seed=42)) -> np.ndarray:
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
        return rng.choice(
            self.centers,
            size=size,
            p=self.counts / np.sum(self.counts, dtype=float),
        )

    def ks_test_new_data(
        self,
        new_data: np.ndarray,
        confidence: float | None = None,
    ) -> KSTestResult:
        """Kolmogorov-Smirnov test for similarity.

        Parameters
        ----------
        new_data : np.ndarray
        confidence : float | None, optional
            if provided will update counts in histogram if pvalue
            greater than confidence level, by default None

        Returns
        -------
        KSTestResult

        """
        scipy_res = ks_1samp(new_data, self.interp_cdf)._asdict()

        test_result = KSTestResult(
            **scipy_res,
        )
        if (confidence is not None) and (test_result.pvalue > confidence):
            self.update(new_data)

        return test_result

    def update(self, new_data: np.ndarray):
        """Update bin counts with additional data.

        TODO: update edges as well.

        Parameters
        ----------
        new_data : np.ndarray

        """
        new_counts, _ = np.histogram(new_data, bins=self.bins)
        self.counts += new_counts


def compute_histogram(
    data: np.ndarray,
    bins: int = 10,
) -> HistogramResults:
    counts, edges = np.histogram(data, bins=bins)

    return HistogramResults(edges, counts)


def plot_hist(win, hist: HistogramResults):
    """Histogram plot.

    Parameters
    ----------
    win : _type_
        _description_
    hist : HistogramResults
        _description_

    """
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


def plot_line(win, x, y):
    """Line plot.

    Parameters
    ----------
    win : _type_
        _description_
    x : _type_
        _description_
    y : _type_
        _description_

    """
    win.resize(800, 480)
    win.setWindowTitle("Histogram")
    plt = win.addPlot()
    plt.plot(x, y)


def do_ks_test(
    hist: HistogramResults,
    new_data: np.ndarray,
) -> Any:
    """Do Kolmogorov Smirnov test.

    Parameters
    ----------
    hist : HistogramResults
        _description_
    new_data : np.ndarray
        _description_

    Returns
    -------
    Any
        _description_

    """
    return ks_1samp(new_data, hist.interp_cdf)


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    data = rng.normal(loc=3, size=1000)
    hist = compute_histogram(data, bins=50)

    win = pg.GraphicsLayoutWidget(show=True)
    plot_hist(win, hist)

    pg.exec()
