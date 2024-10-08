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


if __name__ == "__main__":
    unittest.main()
