import unittest

import numpy as np
from numpy.testing import assert_allclose

from deinterleaver import (
    deinterleave_pairs,
    find_next_match,
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
    def test_find_next_match(self):
        data = np.array([1.5, 3.01, 3.3, 3.4, 6.52, 7.2, 8.7, 9.1, 9.2, 10.21, 10.7])
        res = find_next_match(data, 1.5, 1, 0.001)
        assert len(res) == 0

        res = find_next_match(data, 1.5, 1, 0.01)
        # print(res)
        # assert res == [4]

    # def test_pairs_to_list(self):
    #     pairs = [[0, 1], [1, 4], [5, 6], [6, 9], [7, 10], [8, 10]]
    #     cor = [[0, 1, 4, 5, 6, 9], [7, 10], [8, 10]]
    #     res = pairs_to_list(pairs)
    #     assert len(res) == 3
    #     assert_array_equal(res[0], cor[0])
    #     assert_array_equal(res[1], cor[1])
    #     assert_array_equal(res[2], cor[2])

    # def test_find_precise_pri(self):
    #     data = np.array([1.5, 3.01, 3.3, 3.4, 6.51, 7.2, 8.7, 9.1, 9.2, 10.21, 10.7])
    #     pulses = pairs_to_list(data)
    #     pulses = [[0, 1, 4, 5, 6, 9], [7, 10], [8, 10]]

    #     pulse_groups = find_pulse_groups(data, pulses, 0.2)
    #     assert 4 == 5


if __name__ == "__main__":
    unittest.main()
