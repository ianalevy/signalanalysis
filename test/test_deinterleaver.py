import unittest

import polars as pl
from polars.testing import assert_frame_equal

from deinterleaver import (
    burst_stats,
    filter_by_pri,
    group_by_burst,
    remove_duplicates,
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

    def test_find_burst_starts(self):
        data = pl.DataFrame(
            {
                "toa": [5, 7, 10, 12, 15, 32, 37],
                "rf": [1, 2, 3, 4, 5, 6, 7],
            },
        )
        res = find_burst_starts(data, 5, 0.5, 3)
        assert_frame_equal(
            res,
            pl.DataFrame(
                {
                    "toa": [5, 10, 15],
                    "rf": [1, 3, 5],
                    "burst_group": [0, 0, 0],
                },
            ),
        )

        data = pl.DataFrame(
            {
                "toa": [5, 7, 10, 12, 15, 32, 37, 25, 30, 35],
                "rf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
        )
        res = find_burst_starts(data, 5, 0.5, 3)
        assert_frame_equal(
            res,
            pl.DataFrame(
                {
                    "toa": [5, 10, 15, 25, 30, 35],
                    "rf": [1, 3, 5, 8, 9, 10],
                    "burst_group": [0, 0, 0, 1, 1, 1],
                },
            ),
        )
        data = pl.DataFrame(
            {
                "toa": [5, 7, 10, 12, 15, 32, 37],
                "rf": [1, 2, 3, 4, 5, 6, 7],
            },
        )
        res = find_burst_starts(data, 5, 0.5, 2)
        assert_frame_equal(
            res,
            pl.DataFrame(
                {
                    "toa": [5, 10, 15, 7, 12, 32, 37],
                    "rf": [1, 3, 5, 2, 4, 6, 7],
                    "burst_group": [0, 0, 0, 1, 1, 2, 2],
                },
            ),
        )

    def test_burst_stats(self):
        data = pl.DataFrame(
            {
                "toa": [10.5, 11.62, 15, 16.11, 17.2, 18.32],
                "rf": [1, 2, 3, 4, 5, 6],
                "burst_group": [0, 0, 1, 1, 1, 1],
            },
        )

        res = burst_stats(data)
        assert_frame_equal(
            res,
            pl.DataFrame(
                {
                    "burst_group": [0, 1],
                    "mean": [1.12, 1.1066666],
                    "rf": [[1, 2], [3, 4, 5, 6]],
                },
            ),
        )

    def test_remove_duplicates(self):
        data = [
            pl.DataFrame(
                {
                    "toa": [5, 10, 15, 22, 24, 26],
                    "rf": [1, 2, 3, 4, 5, 6],
                    "burst_group": [0, 0, 0, 1, 1, 1],
                },
            ),
            pl.DataFrame(
                {
                    "toa": [2, 4, 5, 10, 15],
                    "rf": [1, 2, 1, 2, 3],
                    "burst_group": [0, 0, 1, 1, 1],
                },
            ),
        ]

        res = remove_duplicates(data)
        assert_frame_equal(
            res,
            pl.DataFrame(
                {
                    "toa": [2, 4, 5, 10, 15, 22, 24, 26],
                    "rf": [1, 2, 1, 2, 3, 4, 5, 6],
                    "burst_group": [0, 0, 1, 1, 1, 2, 2, 2],
                },
            ),
        )


if __name__ == "__main__":
    pl.Config(tbl_rows=-1)
    unittest.main()
