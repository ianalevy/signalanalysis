import polars as pl


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


def find_burst_starts(df: pl.DataFrame, pri: float, tol: float = 0.1) -> pl.DataFrame:
    toas = df.select("toa").to_numpy()
    last_toa = toas[0]
    burst_starts = [[last_toa]]
    for toa in toas[1:]:
        for burst in burst_starts:
            if np.min(np.abs((toa - pri) - burst)) < tol:
                burst.append(toa)
                break
        else:
            burst_starts.append([toa])

    burst_starts = [np.concat(_) for _ in burst_starts]
    bgs = []
    for idx, bg in enumerate(burst_starts):
        bgs.append(
            df.filter(pl.col("toa").is_in(pl.Series(values=bg))).with_columns(
                pl.Series("burst_group", [idx for _ in range(len(bg))]),
            ),
        )

    return pl.concat(bgs)


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
    return (
        df.with_columns(pl.col("toa").diff().over("burst_group").alias("group_deltas"))
        .group_by("burst_group")
        .agg(pl.col("group_deltas").mean().alias("mean"), pl.col("rf"))
    ).sort("burst_group")


def remove_duplicates(
    groups: list,
    burst_col: str = "burst_group",
    time_col: str = "toa",
) -> pl.DataFrame:
    """Remove duplicate entries from the DataFrame based on 'toa' and 'rf' columns.

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame with possible duplicates.

    Returns
    -------
    pl.DataFrame
        DataFrame without duplicates.

    """
    combined: pl.DataFrame = pl.concat(
        [
            group.with_columns(
                (f"{idx}_" + pl.col(burst_col).cast(pl.String)).alias(
                    burst_col,
                ),
            )
            for idx, group in enumerate(groups)
        ],
    ).sort(burst_col, descending=False)

    toas_found = []
    unique_groups = []

    for name, group in combined.group_by(burst_col):
        new_toas = group[time_col].to_list()
        if new_toas not in toas_found:
            toas_found.append(new_toas)
            unique_groups.append(group)

    return (
        pl.concat(unique_groups)
        .sort(time_col)
        .with_columns(
            pl.col("burst_group").rle_id().alias("burst_group").cast(pl.Int64),
        )
    )


if __name__ == "__main__":
    print("hello")
