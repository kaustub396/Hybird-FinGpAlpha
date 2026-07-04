import numpy as np
import pandas as pd


def _to_datetime_index(values, name="date"):
    idx = pd.to_datetime(values, errors="coerce")
    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx.values, name=name)
    elif not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx, name=name)
    return idx


def align_gp_signal_to_afm(
    afm_df,
    gp_signal,
    date_col="date",
    gp_col="gp_signal",
    fill_method="ffill",
    max_fill_days=None,
    drop_na_gp=True,
):
    """
    Align GP signal to AFM dataframe by date and append as a feature column.

    Parameters
    ----------
    afm_df : DataFrame
        AFM input data containing a date column.
    gp_signal : Series
        Time-series signal indexed by date.
    date_col : str
        AFM date column name.
    gp_col : str
        Output GP signal column name.
    fill_method : str or None
        Fill policy after join. Supported: 'ffill', 'bfill', None.
    max_fill_days : int or None
        Optional max distance (in days) allowed for propagated fills.
    drop_na_gp : bool
        Whether to drop rows where gp signal remains missing.

    Returns
    -------
    DataFrame
        AFM dataframe with `gp_col` appended and date-sorted.
    """
    if date_col not in afm_df.columns:
        raise KeyError(f"Missing required date column: {date_col}")
    if not isinstance(gp_signal, pd.Series):
        raise TypeError("gp_signal must be a pandas Series indexed by date")

    out = afm_df.copy()
    out[date_col] = _to_datetime_index(out[date_col], name=date_col)
    out = out.sort_values(date_col)

    gp = gp_signal.copy()
    gp.index = _to_datetime_index(gp.index, name=date_col)
    gp = gp.sort_index()
    gp.name = gp_col

    out = out.merge(gp.to_frame(), how="left", left_on=date_col, right_index=True)

    if fill_method in ("ffill", "bfill"):
        out[gp_col] = out[gp_col].fillna(method=fill_method)
    elif fill_method is not None:
        raise ValueError("fill_method must be one of: 'ffill', 'bfill', None")

    if max_fill_days is not None:
        if max_fill_days < 0:
            raise ValueError("max_fill_days must be >= 0")
        left = out[[date_col]].copy()
        right = gp.reset_index()
        right.columns = [date_col, gp_col]
        nearest = pd.merge_asof(
            left.sort_values(date_col),
            right.sort_values(date_col),
            on=date_col,
            direction="nearest",
            tolerance=pd.Timedelta(days=int(max_fill_days)),
        )
        nearest.index = left.sort_values(date_col).index
        out.loc[nearest[gp_col].isna(), gp_col] = np.nan

    if drop_na_gp:
        out = out.dropna(subset=[gp_col])

    return out.reset_index(drop=True)

