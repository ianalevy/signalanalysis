import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from pulse_simulator import (
    Pdw,
    Pulse,
    frame_array,
    make_signal,
    moving_average,
    noise_filter,
    sampled_dw,
    try_pris,
)


class TestPdw(unittest.TestCase):
    def test_sample(self):
        pdw = Pdw(
            np.array([0.05, 0.10, 0.15, 0.20]),
            np.array([0.01, 0.02, 0.03, 0.001]),
            np.array([2, 2, 2, 2]),
            np.array([2.1, 4.2, 6.5, 1.3]),
        )
        res = sampled_dw(pdw, 80)
        assert_array_almost_equal(
            res.toa_s,
            np.array(
                [
                    0.05,
                    0.0625,
                    0.075,
                    0.0875,
                    0.1,
                    0.1125,
                    0.125,
                    0.1375,
                    0.15,
                    0.1625,
                    0.175,
                    0.1875,
                    0.2,
                ],
            ),
        )
    def test_moving_average(self):
        data = np.array([1.0, 1, 1, 2.2, 5.5, 1, 1, 1, 1])
        # order 1 does nothing
        res = moving_average(data, 1)
        truth = np.array([1, 1, 1, 2.2, 5.5, 1, 1, 1, 1])
        assert_array_almost_equal(res, truth)

        data = np.array([1.0, 1, 1, 2.2, 5.5, 1, 1, 1, 1])
        res = moving_average(data, 2)
        truth = np.array([1 / 2, 1, 1, 3.2 / 2, 7.7 / 2, 6.5 / 2, 1, 1, 1])
        assert_array_almost_equal(res, truth)

        data = np.array([1.0, 1, 1, 2.2, 5.5, 1, 1, 1, 1])
        res = moving_average(data, 3)
        truth = np.array([2 / 3, 1, 4.2 / 3, 8.7 / 3, 8.7 / 3, 7.5 / 3, 1, 1, 2 / 3])
        assert_array_almost_equal(res, truth)

        # window of size 8
        data = np.array([1.0, 1, 1, 2.2, 5.5, 1, 1, 1])
        # order 1 does nothing
        res = moving_average(data, 1)
        truth = np.array([1, 1, 1, 2.2, 5.5, 1, 1, 1])
        assert_array_almost_equal(res, truth)

        data = np.array([1.0, 1.0])
        res = moving_average(data, 3)
        truth = np.array([2.0 / 3.0, 2.0 / 3.0])
        assert_array_almost_equal(res, truth)

        data = np.array([1.0, 1, 1, 5])
        res = moving_average(data, 3)
        truth = np.array([2.0 / 3.0, 1.0, 7.0 / 3, 6.0 / 3.0])
        assert_array_almost_equal(res, truth)
        res = moving_average(data, 3)
        truth = np.array([0, 1, 7 / 3, 7 / 3, 7 / 3, 1, 1, 2 / 3])
        assert_array_almost_equal(res, truth)


class TestAnalysis(unittest.TestCase):
    def test_frame_array(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        res = frame_array(data, 2)
        assert_array_equal(data, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))

        assert_array_equal(
            res,
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 0]]),
        )

        res = frame_array(data, 3)
        assert_array_equal(
            res,
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 0]]),
        )

    def test_pulse(self):
        pulse = Pulse()
        res = pulse.sample_pulse(1e-2)

        assert len(res) == 20 / (1.0e-2)

    def test_make_signal(self):
        (times, signal) = make_signal(0.1, 0.01, 3, 0.02)
        assert len(times) == 33
        stop = 3 * 0.1 + 0.02
        assert_array_almost_equal(
            times,
            np.arange(0, stop, step=0.01),
        )

    def test_calc_norm(self):
        data = np.array([[0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1]])
        res = calc_norm(data)

        assert np.abs(res - 5.0 / 3) < 0.0001

    def test_detector(self):
        pw_s = 0.03
        sample_rate_s = 0.01
        (times, data) = make_signal(0.1, sample_rate_s, 3, pw_s)

        # print(data)
        res = detector(data, sample_rate_s, pw_s)
        # print(res)
        # assert 4 == 5
        # fmt: off
        true_detects = np.array(
            [
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        )

        assert_array_equal(
            res,
            true_detects,
        )

if __name__ == "__main__":
    unittest.main()
