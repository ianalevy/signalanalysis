import unittest

import numpy as np
from numpy.testing import assert_array_equal

from analysis import HistogramResults, compute_histogram, ks_test_data


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

    def test_ks_test(self):
        rng = np.random.default_rng(seed=42)
        data_len = 1000
        data = rng.uniform(10, 20, size=data_len)
        hist = compute_histogram(data, 100)

        new_data = rng.uniform(10, 40, size=10)
        res = hist.ks_test_new_data(new_data)
        assert res.pvalue < 0.01
        assert np.abs(res.statistic - 0.659) < 0.001
        assert not (ks_test_data(data, new_data)[0])
        assert np.sum(hist.counts) == data_len

        new_data = rng.uniform(10, 20, size=10)
        assert ks_test_data(data, new_data)[0]
        res = hist.ks_test_new_data(
            new_data,
            confidence=0.05,
            update_counts=True,
        )
        # data is from same distribution so will have large p value
        assert res.pvalue > 0.05
        # all data should be added to histogram
        assert np.sum(hist.counts) == data_len + 10

    def test_update_hist(self):
        rng = np.random.default_rng(seed=42)
        data = rng.uniform(10, 20, size=20)
        hist = compute_histogram(data, 10)
        assert_array_equal(hist.counts, np.array([3, 1, 0, 1, 3, 1, 3, 3, 3, 2]))

        new_data = rng.uniform(15, 20, size=5)
        hist.update(new_data)
        assert_array_equal(hist.counts, np.array([3, 1, 0, 1, 3, 1, 4, 3, 4, 4]))


if __name__ == "__main__":
    unittest.main()
