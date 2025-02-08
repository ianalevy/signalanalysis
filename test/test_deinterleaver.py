import unittest

import numpy as np
import polars as pl
from numpy.testing import assert_allclose
from polars.testing import assert_frame_equal

from deinterleaver import (
    deinterleave_pairs,
    filter_by_pri,
    indices_to_pulse_pairs,
)

# def test_remove_dupes(self):
#     data = pl.DataFrame(
#         {
#             "toa": [2, 10, 11, 25, 50, 51, 52],
#             "rf": [1000.1000, 1000, 1000, 1000, 1000, 1000, 1000],
#             "id": [3, 0, 1, 1, 1, 0, 2],
#         },
#     )
#     res = remove_dupes(data)

#     assert_frame_equal(
#         res,
#         pl.DataFrame({"toa": [2, 10, 25, 50], "id": [3, 0, 1, 1]}),
#     )

#     data = pl.DataFrame(
#         {"toa": [10, 11, 25, 50, 51, 52], "id": [0, 1, 1, 1, 0, 2]},
#     )
#     res = remove_dupes(data)

#     assert_frame_equal(
#         res,
#         pl.DataFrame({"toa": [10, 25, 50], "id": [0, 1, 1]}),
#     )

#     data = pl.DataFrame(
#         {"toa": [10, 11, 25, 50, 51, 54], "id": [0, 1, 1, 1, 0, 2]},
#     )
#     res = remove_dupes(data, tol=2)

#     assert_frame_equal(
#         res,
#         pl.DataFrame({"toa": [10, 25, 50, 54], "id": [0, 1, 1, 2]}),
#     )


class TestDeinterleaver(unittest.TestCase):
    def test_indices_to_pulses(self):
        in1 = np.array([11, 7, 3, 5, 9, 21])
        in2 = np.array([2, 6, 8, 4, 16, 12])
        matches = np.array([[2, 1], [3, 5], [5, 4]])
        res = indices_to_pulse_pairs(in1, in2, matches)
        assert_allclose(res, np.array([[3, 6], [5, 12], [21, 16]]))

    def test_deinterleave(self):
        toa1 = np.array([3.4, 4.5, 5.6, 12.3, 13.4, 14.5, 21.2])
        toa2 = np.array([4.51, 5.2, 5.61, 12.31, 13.41, 14.51])

        res = deinterleave_pairs(toa1, toa2, tol=0.1)

        assert_allclose(
            res,
            np.array(
                [[4.5, 4.51], [5.6, 5.61], [12.3, 12.31], [13.4, 13.41], [14.5, 14.51]],
            ),
        )

        # smaller tolerance
        toa1 = np.array([3.4, 4.5, 5.4, 12.25, 13.4, 14.5, 21.2])
        toa2 = np.array([4.52, 5.2, 5.66, 12.31, 13.41, 14.53])
        res = deinterleave_pairs(toa1, toa2, tol=0.04)

        assert_allclose(
            res,
            np.array(
                [[4.5, 4.52], [13.4, 13.41], [14.5, 14.53]],
            ),
        )

    def test_2(self):
        # now case where multiple within tolerance
        toa1 = np.array([3.4, 4.5, 5.6, 12.3, 13.4, 14.5, 15.62, 15.61])
        toa2 = np.array([4.53, 4.51, 12.31, 13.41, 14.51, 15.6])

        res = deinterleave_pairs(toa1, toa2, tol=0.04)
        print(res)

        assert_allclose(
            res,
            np.array(
                [
                    [4.5, 4.51],
                    [12.3, 12.31],
                    [13.4, 13.41],
                    [14.5, 14.51],
                    [15.62, 15.6],
                    [15.61, 15.6],
                ],
            ),
        )


class TestPrecisePri(unittest.TestCase):
    def test_filter_by_pri(self):
        data = pl.DataFrame(
            {
                "toa": [10.0, 12.52, 18.99],
                "rf": [1, 2, 1],
            },
        )

        res = filter_by_pri(data, 2.5)

        assert_frame_equal(
            res,
            pl.DataFrame(
                {
                    "toa": [10.0, 12.52],
                    "rf": [1, 2],
                },
            ),
        )
        data = pl.DataFrame(
            {
                "toa": [10.0, 18.8, 19.3],
                "rf": [1, 2, 1],
            },
        )

        res = filter_by_pri(data, 0.4, tol=0.2)

        assert_frame_equal(
            res,
            pl.DataFrame(
                {
                    "toa": [18.8, 19.3],
                    "rf": [2, 1],
                },
            ),
        )

        data = pl.DataFrame(
            {
                "toa": [10.0, 12.52, 14.99, 15.2, 17.71, 23, 24, 26.5, 29],
                "rf": [1, 2, 1, 1, 1, 2, 5, 9, 2],
            },
        )

        res = filter_by_pri(data, 2.5)

        assert_frame_equal(
            res,
            pl.DataFrame(
                {
                    "toa": [10.0, 12.52, 14.99, 15.2, 17.71, 24, 26.5, 29],
                    "rf": [1, 2, 1, 1, 1, 5, 9, 2],
                },
            ),
        )

    def test_group_by_burst(self):
        data = pl.DataFrame(
            {
                "toa": [10.5, 11.6, 15, 16.1, 17.2],
                "rf": [1, 2, 3, 4, 5],
            },
        )
        res = group_by_burst(data)

        assert_frame_equal(
            res,
            pl.DataFrame(
                {
                    "toa": [10.5, 11.6, 15, 16.1, 17.2],
                    "rf": [1, 2, 3, 4, 5],
                    "burst_group": [0, 0, 1, 1, 1],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
