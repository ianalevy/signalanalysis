import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from pulse_simulator import (
    Pulse,
    calc_norm,
    detector,
    frame_array,
    make_signal,
    try_pris,
)


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
