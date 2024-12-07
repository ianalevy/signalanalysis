import unittest

import numpy as np
from numpy.testing import assert_allclose

from deinterleaver import indices_matching, indices_to_pulse_pairs


class TestDeinterleaver(unittest.TestCase):
    def test_1(self):
        toa1 = np.array([3.4, 4.5, 5.6, 12.3, 13.4, 14.5, 21.2])
        toa2 = np.array([4.51, 5.2, 5.61, 12.31, 13.41, 14.51])

        res = indices_matching(toa1, toa2, tol=0.1)

        assert_allclose(
            res,
            np.array(
                [[4.5, 4.51], [5.6, 5.61], [12.3, 12.31], [13.4, 13.41], [14.5, 14.51]],
            ),
        )


    def test_indices_to_pulses(self):
        in1 = np.array([11, 7, 3, 5, 9, 21])
        in2 = np.array([2, 6, 8, 4, 16, 12])
        matches = np.array([[2, 1], [3, 5], [5, 4]])
        res = indices_to_pulse_pairs(in1, in2, matches)
        assert_allclose(res, np.array([[3, 6], [5, 12], [21, 16]]))


if __name__ == "__main__":
    unittest.main()
