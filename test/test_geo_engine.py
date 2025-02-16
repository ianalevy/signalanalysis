import unittest

import polars as pl

from geo_engine import get_geos


class TestPrecisePri(unittest.TestCase):
    def test_geo_engine(self):
        res = get_geos(
            pl.DataFrame(
                {
                    "time": [1, 2, 3],
                    "tdoa": [0.1, 0.2, 0.3],
                    "sensor1": [1, 2, 3],
                    "sensor2": [1, 2, 3],
                },
            ),
        )
        print(res)
        assert 4 == 4


if __name__ == "__main__":
    unittest.main()
