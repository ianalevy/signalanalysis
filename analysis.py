from dataclasses import dataclass

import numpy as np


@dataclass
    class HistogramResults:
        bins: np.ndarray
        counts: np.ndarray
        centers = np.ndarray = np.array([])

        def __post__init(self):
            self.centers = (self.bins[1:] - self.bins[:-1]) / 2
