import unittest

import numpy as np
from numpy.testing import assert_array_equal

from analysis import HistogramResults, compute_histogram


class TestAnalysis(unittest.TestCase):
    def test_centers(self):
        bins = np.arange(1, 12, 3)
        counts = [4, 3, 6, 2]
        res = HistogramResults(bins, counts)
        assert_array_equal(res.bins, bins)
        assert_array_equal(res.centers, np.array([2.5, 5.5, 8.5]))

    def test_compute_histogram(self):
        rng = np.random.default_rng(seed=42)
        data = rng.normal(loc=3, size=12)
        data_len = 6
        res = compute_histogram(data, data_len)
        assert len(res.bins) == data_len + 1
        assert len(res.counts) == data_len
        assert len(res.centers) == data_len

    def test_histogram_results(self):
        rng = np.random.default_rng(seed=42)
        mean = 4
        data = rng.normal(loc=mean, size=1000)
        hist = compute_histogram(data, 100)

        my_sample = hist.sample(100)
        assert len(my_sample) == 100
        assert np.abs(np.mean(my_sample) - mean) < 0.1

        assert hist.interp_cdf(mean + 3) > 0.99
        assert hist.interp_cdf(mean - 3) < 0.01


if __name__ == "__main__":
    unittest.main()
