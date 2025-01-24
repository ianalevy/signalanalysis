import numpy as np
import polars as pl


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

    For each time in toa1 searches toa2 for nearerst time within tolerance.

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
        new_match = np.argmin(np.abs(toa2 - time))
        if np.abs(toa2[new_match] - time) < tol:
            match_indices.append([in1[0], new_match])

    return indices_to_pulse_pairs(toa1, toa2, match_indices)


def find_next_match(
    pulses: np.ndarray,
    rough_pri: float,
    start_index: int,
    tol: float = 0.1,
) -> list:
    """Find next matching pulse in group to tol.

    Parameters
    ----------
    pulses : np.ndarray
        _description_
    rough_pri : float
        _description_
    start_index : int
        _description_
    tol : float, optional
        _description_, by default 0.1

    Returns
    -------
    list

    """
    group_start = pulses[start_index]
    later_pulses = pulses[start_index + 1 :]
    diffs = np.abs((later_pulses - group_start) - rough_pri)

    return np.where(diffs < tol)[0] + start_index + 1


def find_pri_pairs(toas: np.ndarray, rough_pri: float, tol: float = 0.1) -> list:
    pulse_groups = []
    num_toas = len(toas)
    for start_index in range(num_toas - 1):
        matches = find_next_match(toas, rough_pri, start_index, tol=tol)
        if len(matches) > 0:
            pulse_groups.append([start_index, matches[0]])
    return pulse_groups


def pairs_to_list(pairs: list) -> np.ndarray:
    """Convert pairs to list of groups.

    Parameters
    ----------
    pairs : list
        _description_

    Returns
    -------
    np.ndarray
        _description_

    """
    last_end = -1
    good_indices = [[]]
    for start, end in pairs:
        if start == last_end:
            good_indices[-1].append(end)
        elif start > last_end:
            good_indices[-1] += [start, end]

        else:
            good_indices.append([start, end])
        last_end = end

    return good_indices


def remove_dupes(df: pl.DataFrame, tol: int = 5) -> pl.DataFrame:
    """Remove duplicates near enough in time.

    Parameters
    ----------
    df : pl.DataFrame
        _description_
    tol : int, optional
        _description_, by default 5

    Returns
    -------
    pl.DataFrame

    """
    df = df.with_columns(pl.col("toa").diff().fill_null(2 * tol).alias("deltas"))
    df = df.with_columns(
        pl.when(pl.col("deltas") > tol).then(1).otherwise(0).alias("jumps"),
    )

    df = df.filter(pl.col("jumps") == 1).drop(["deltas", "jumps"])

    return df


if __name__ == "__main__":
    print("hello")
