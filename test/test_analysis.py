import unittest

import numpy as np
from numpy.testing import assert_array_equal

from analysis import HistogramResults


class TestAnalysis(unittest.TestCase):
    def test_centers(self):
        bins = np.arange(1, 12, 3)
        counts = [4, 3, 6, 2]
        res = HistogramResults(bins, counts)
        assert_array_equal(res.bins, bins)
        assert_array_equal(res.centers, np.array([2.5, 5.5, 8.5]))


if __name__ == "__main__":
    unittest.main()
