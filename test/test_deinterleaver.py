import unittest

import numpy as np
from numpy.testing import assert_allclose

from deinterleaver import deinterleave_bursts


class TestDeinterleaver(unittest.TestCase):
    def test_1(self):
        toa1 = np.array([4.5, 5.6, 12.3, 13.4, 14.5])
        toa2 = np.array([4.51, 5.61, 12.31, 13.41, 14.51])

        res = deinterleave_bursts(toa1, toa2)
        print(res)

        assert_allclose(
            res,
            np.array(
                [[4.5, 4.51], [5.6, 5.61], [12.3, 12.31], [13.4, 13.41], [14.5, 14.51]],
            ),
        )


if __name__ == "__main__":
    unittest.main()
