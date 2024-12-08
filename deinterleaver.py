import numpy as np


def indices_to_pulse_pairs(
    in1: np.ndarray,
    in2: np.ndarray,
    matches: list | np.ndarray,
) -> np.ndarray:
    """Get columns of pairs from indices.

    Parameters
    ----------
    in1 : np.ndarray
    in2 : np.ndarray
        _description_
    matches : list | np.ndarray
        indices of matching pairs

    Returns
    -------
    np.ndarray

    """
    if isinstance(matches, list):
        matches = np.array(matches)
    return np.column_stack((in1[matches[:, 0]], in2[matches[:, 1]]))


if __name__ == "__main__":
    print("hello")
