from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

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
        confidence: float = 0.05,
        update_counts: bool | None = None,
    ) -> KSTestResult:
        """Kolmogorov-Smirnov test for similarity.

        Parameters
        ----------
        new_data : np.ndarray
            _description_
        confidence : float, optional
            _description_, by default 0.05
        update_counts : bool | None, optional
            If True will update counts in histogram, by default None

        Returns
        -------
        KSTestResult

        """
        scipy_res = ks_1samp(new_data, self.interp_cdf)._asdict()

        test_result = KSTestResult(
            **scipy_res,
        )
        if update_counts and (test_result.pvalue > confidence):
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


def ks_test_data(
    ref_data: np.ndarray,
    new_data: np.ndarray,
    bins: int = 100,
    confidence: float = 0.05,
) -> tuple[bool, KSTestResult]:
    """Check if two data sets from same distribution.

    Parameters
    ----------
    ref_data : np.ndarray
        reference data set
    new_data : np.ndarray
        new data set
    bins : int, optional
        number of bins to use in histogram, by default 100
    confidence : float, optional
        confidence level, by default 0.05

    Returns
    -------
    tuple[bool, KSTestResult]

    """
    hist = compute_histogram(ref_data, bins=bins)
    ks_res = hist.ks_test_new_data(new_data, confidence=confidence)

    data_match_q = ks_res.pvalue > confidence
    return (data_match_q, ks_res)


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
    mean = 4
    data = rng.normal(loc=mean, size=10000000)
    # data = rng.exponential(scale=1, size=10000)
    hist = compute_histogram(data, bins=1000)

    # win = pg.GraphicsLayoutWidget(show=True)
    # plot_hist(win, hist)
    # plot_line(win, hist.centers, hist.interp_cdf(hist.centers))
    # win.nextRow()

    stat1s = []
    stat2s = []
    for _ in list(range(1000)):
        new_data = hist.sample(100, rng=rng)
        res = stats.ks_1samp(new_data, lambda x: stats.norm.cdf(x, loc=mean))
        res2 = do_ks_test(hist, new_data)
        stat1s.append(res.statistic)
        stat2s.append(res2.statistic)

    print(np.mean(stat1s))
    print(np.mean(stat2s))

    exit()
    new_data = hist.sample(100)

    res = stats.ks_1samp(new_data, lambda x: stats.norm.cdf(x, loc=mean))
    print(res)

    res = do_ks_test(hist, new_data)
    print(res)

    # new_hist = compute_histogram(new_data, bins=50)
    # plot_hist(win, new_hist)
    # plot_line(win, new_hist.centers, new_hist.cdf)

    # pg.exec()
