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


def deinterleave_pairs(toa1: np.ndarray, toa2: np.ndarray, tol=0.01) -> np.ndarray:
    """Deinterleave pairs within tolerance.

    Parameters
    ----------
    toa1 : np.ndarray
    toa2 : np.ndarray
    tol : float, optional
        by default 0.01

    Returns
    -------
    np.ndarray

    """
    match_indices = []
    for in1, time in np.ndenumerate(toa1):
        indices = np.argwhere(np.abs(toa2 - time) < tol)
        if len(indices > 0):
            match_indices.append([in1[0], indices[0][0].tolist()])

    return indices_to_pulse_pairs(toa1, toa2, match_indices)


if __name__ == "__main__":
    print("hello")
