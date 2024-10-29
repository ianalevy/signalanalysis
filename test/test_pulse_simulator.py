import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from pulse_simulator import Pulse, frame_array, make_signal


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
        assert len(times) == 32
        stop = 3 * 0.1 + 0.02
        assert_array_almost_equal(
            times,
            np.linspace(0, stop, num=int(stop / 0.01)),
        )


if __name__ == "__main__":
    unittest.main()
