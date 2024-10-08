from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HistogramResults:
    bins: np.ndarray
    counts: np.ndarray
    centers: np.ndarray | None = None

    def __post_init__(self):
        if self.centers is None:
            self.centers = (self.bins[1:] + self.bins[:-1]) / 2


def compute_histogram(
    data: np.ndarray,
    bins: int | None = None,
) -> HistogramResults:
    counts, edges = np.histogram(data, bins=bins)

    return HistogramResults(edges, counts)
