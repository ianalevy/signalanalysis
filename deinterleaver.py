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


def group_by_burst(
    df: pl.DataFrame,
    pri: float,
    tol: float = 0.1,
    min_num_pulses: int = 5,
    time_col: str = "toa",
    burst_col: str = "burst_group",
) -> pl.DataFrame:
    df = df.with_columns(
        pl.lit(-1).alias(burst_col),
    ).sort(
        time_col,
    )
    for idx, toa in enumerate(df.select(time_col).to_numpy()):
        dist_col = "dist"
        bursts_found = (
            df.filter(pl.col(burst_col) > -1)
            .with_columns(
                (toa - pri - pl.col(time_col))
                .abs()
                .min()
                .over(burst_col)
                .alias(dist_col),
            )
            .select(burst_col, dist_col)
        )
        if (len(bursts_found) == 0) or (bursts_found[dist_col].min() > tol):
            df[idx, burst_col] = max(df[burst_col]) + 1
        else:
            df[idx, burst_col] = bursts_found.item(
                bursts_found[dist_col].arg_min(),
                burst_col,
            )

    return (
        df.filter(
            pl.col(burst_col) > -1,
            pl.len().over(burst_col) >= min_num_pulses,
        )
        .sort(burst_col)
        .with_columns(pl.col(burst_col).rle_id().cast(pl.Int64))
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
