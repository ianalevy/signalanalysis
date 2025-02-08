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


def find_pulse_groups(
    toas: np.ndarray,
    pulse_groups: list,
    pulse_gap: float = 0.1,
) -> list:
    bursts = []
    for group in pulse_groups:
        group_toas = toas[group]
        print(group_toas)
        deltas = np.diff(group_toas)
        print(deltas)
    # pulse_groups = []

    # group_start = pulse_groups[0][0]
    # pulse_groups.append([group_start])
    # for pulse_pair in pulse_groups:
    #     duration = toas[pulse_pair[1]] - toas[group_start]
    #     if duration < pulse_gap:
    #         pulse_groups.append(pulse_pair[1])
    #     else:
    #         pulse_groups.append([pulse_pair])

    return bursts


def find_precise_pri(toas: np.ndarray, rough_pri: float, tol: float = 0.1) -> list:
    pulse_groups = []
    num_toas = len(toas)
    for start_index in range(num_toas - 1):
        print(start_index)
        matches = find_next_match(toas, rough_pri, start_index, tol=tol)
        print(matches)
        if len(matches) > 0:
            pulse_groups.append([start_index, matches[0]])
        # if len(matches) > 0:
        #     last_match = matches[0]
        #     pulse_groups.append([start_index, last_match])
        #     current_index = last_match
        #     elapsed_time = 0
        #     current_index = last_match
        #     while (current_index < len(toas)) and (elapsed_time) < rough_pri + tol:
        #         next_match = find_next_match(toas, rough_pri, current_index, tol=tol)
        #         if len(next_match) > 0:
        #             last_match = next_match[0]
        #             pulse_groups[-1].append(last_match)
        #             current_index = last_match
        #         else:
        #             current_index += 1
        #             if current_index < len(toas):
        #                 elapsed_time = toas[current_index] - toas[last_match]
        #                 print(elapsed_time)

    return pulse_groups


def remove_dupes(df: pl.DataFrame, tol: int = 5, rf_tol: float = 10) -> pl.DataFrame:
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


def filter_by_pri(df: pl.DataFrame, pri: float, tol: float = 0.1) -> pl.DataFrame:
    """Fitler for pulses that match the PRI.

    Parameters
    ----------
    df : pl.DataFrame
        _description_
    pri : float
        _description_
    tol : float, optional
        _description_, by default 0.1

    Returns
    -------
    pl.DataFrame

    """
    match_next = (
        df.with_columns((pl.col("toa") + pri).alias("next"))
        .join_asof(
            df.select("toa"),
            left_on="next",
            right_on="toa",
            strategy="nearest",
            tolerance=tol,
        )
        .filter(pl.col("toa_right").is_not_null())
    ).drop("toa_right", "next")

    match_pre = (
        df.with_columns((pl.col("toa") - pri).alias("pre"))
        .join_asof(
            df.select("toa"),
            left_on="pre",
            right_on="toa",
            strategy="nearest",
            tolerance=tol,
        )
        .filter(pl.col("toa_right").is_not_null())
    ).drop("toa_right", "pre")

    return (
        pl.concat([match_next, match_pre])
        .sort("toa")
        .filter(pl.col("toa").diff().fill_null(tol).abs() > 0)
    )


def group_by_burst(df: pl.DataFrame, pri: float, tol: float = 0.01) -> pl.DataFrame:
    """Group bursts.

    Parameters
    ----------
    df : pl.DataFrame
        _description_
    pri : float
        _description_
    tol : float, optional
        _description_, by default 0.01

    Returns
    -------
    pl.DataFrame
        _description_

    """
    return df.with_columns(
        ((pl.col("toa").diff().fill_null(pri) - pri).abs() > tol)
        .cum_sum()
        .cast(pl.Int64)
        .alias("burst_group"),
    )


def burst_stats(df: pl.DataFrame) -> pl.DataFrame:
    df = df.group_by("burst_group").agg(
        pl.col("toa").diff().mean().alias("mean"),
        pl.col("toa").diff().std().alias("std"),
        pl.col("rf"),
    )

    return df


if __name__ == "__main__":
    print("hello")
