import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from indicators import indicators

real_price_cols = ["real_high", "real_low", "real_close"]
funding_cols = ["funding_rate"]
btcd_cols = ["time", "btcd_open", "btcd_high", "btcd_low", "btcd_close", "btcd_volume"]
btcdom_cols = ["time", "btcdom_open", "btcdom_high", "btcdom_low", "btcdom_close", "btcdom_volume"]
fng_cols = ["time", "fng_value"]


def add_indicators(
    df: pd.DataFrame,
    ttype: str,
    configs: dict,
    df_higher: pd.DataFrame = None,
) -> pd.DataFrame:
    """Add technical indicators to df.

    Computes RSI, STOCH, ATR, CCI, and SAR on `df`. If `df_higher` is provided,
    also computes MACD and Trend on the higher timeframe and merges the result
    into `df` (time-shifted by +3h).

    Parameters
    ----------
    df : pd.DataFrame
        1h OHLCV DataFrame for one ticker.
    ttype : str
        Trade type, ``"buy"`` or ``"sell"``.
    configs : dict
        Full config dict; indicator parameters are read from ``configs["Model"]``.
    df_higher : pd.DataFrame, optional
        4h OHLCV DataFrame for the same ticker. When provided, MACD and Trend
        features from this frame are merged into `df`.

    Returns
    -------
    pd.DataFrame
        `df` with indicator columns appended in-place.
    """
    rsi = indicators.RSI(ttype, configs)
    df = rsi.get_indicator(df, "", "", 0)
    stoch = indicators.STOCH(ttype, configs)
    df = stoch.get_indicator(df, "", "", 0)
    atr = indicators.ATR(ttype, configs)
    df = atr.get_indicator(df, "", "", 0)
    cci = indicators.CCI(ttype, configs)
    df = cci.get_indicator(df, "", "", 0)
    sar = indicators.SAR(ttype, configs)
    df = sar.get_indicator(df, "", "", 0)
    if df_higher is not None:
        df_higher = df_higher.copy()
        macd = indicators.MACD(ttype, configs)
        df_higher = macd.get_indicator(df_higher, "", "", 0)
        trend = indicators.Trend(ttype, configs)
        df_higher = trend.get_indicator(df_higher, "", "", 0)
        df_higher["time"] = df_higher["time"] + pd.to_timedelta(3, unit="h")
        ohlcv_cols = {"open", "high", "low", "close", "volume"}
        cols_to_merge = ["time"] + [
            c for c in df_higher.columns if c not in ohlcv_cols and c != "time"
        ]
        merged = pd.merge(df[["time"]], df_higher[cols_to_merge], how="left", on="time")
        for c in cols_to_merge[1:]:
            df[c] = merged[c].values
        df = df.drop(columns=["close_smooth"], errors="ignore")
        df = df.drop(columns=[c for c in df.columns if c.endswith("_dir")])
    return df


def merge_btc_dominance(df: pd.DataFrame, btcd: pd.DataFrame, btcdom: pd.DataFrame) -> pd.DataFrame:
    """Merge BTC dominance dataframes into df and forward-fill the merged columns.

    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame with a ``time`` column.
    btcd : pd.DataFrame
        Daily BTC dominance OHLCV (``btcd_open/high/low/close/volume``).
    btcdom : pd.DataFrame
        4h BTC.D index OHLCV (``btcdom_open/high/low/close/volume``).

    Returns
    -------
    pd.DataFrame
        `df` with BTC dominance columns appended and forward-filled.
    """
    btcd_cols = list(btcd.columns)
    btcdom_cols = list(btcdom.columns)
    btcd_data_cols = [c for c in btcd_cols if c != "time"]
    df[btcd_data_cols] = pd.merge(
        df[["time"]], btcd[btcd_cols].drop_duplicates("time"), how="left", on="time"
    )[btcd_data_cols].values
    btcdom_data_cols = [c for c in btcdom_cols if c != "time"]
    df[btcdom_data_cols] = pd.merge(
        df[["time"]], btcdom[btcdom_cols].drop_duplicates("time"), how="left", on="time"
    )[btcdom_data_cols].values
    df[btcd_data_cols + btcdom_data_cols] = df[btcd_data_cols + btcdom_data_cols].ffill()
    return df


def merge_fng(df: pd.DataFrame, fng: pd.DataFrame) -> pd.DataFrame:
    """Merge the Crypto Fear & Greed index into df and forward-fill it.

    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame with a ``time`` column.
    fng : pd.DataFrame
        Daily Fear & Greed index (``time``, ``fng_value``).

    Returns
    -------
    pd.DataFrame
        `df` with the ``fng_value`` column appended and forward-filled.
    """
    fng_data_cols = [c for c in fng.columns if c != "time"]
    df[fng_data_cols] = pd.merge(df[["time"]], fng.drop_duplicates("time"), how="left", on="time")[
        fng_data_cols
    ].values
    df[fng_data_cols] = df[fng_data_cols].ffill()
    return df


def scale_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Convert columns to percent change (* 100) to normalise scale across tickers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns will be transformed.
    cols : list
        Column names to convert.

    Returns
    -------
    pd.DataFrame
        `df` with the specified columns replaced by their percent-change values.
    """
    for c in cols:
        df[c] = df[c].pct_change() * 100
    return df


def get_file(ticker):
    """Load 1h and 4h OHLCV pickle files for a ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. ``"BTCUSDT"``.

    Returns
    -------
    tmp_df_1h : pd.DataFrame or None
        1h candle DataFrame, or ``None`` if the file is missing.
    tmp_df_4h : pd.DataFrame or None
        4h candle DataFrame, or ``None`` if the file is missing.
    """
    try:
        tmp_df_1h = pd.read_pickle(f"data/tickers/{ticker}_1h.pkl")
    except FileNotFoundError:
        print(f"File not found: {ticker}_1h.pkl")
        tmp_df_1h = None
    try:
        tmp_df_4h = pd.read_pickle(f"data/tickers/{ticker}_4h.pkl")
    except FileNotFoundError:
        print(f"File not found: {ticker}_4h.pkl")
        tmp_df_4h = None
    return tmp_df_1h, tmp_df_4h


def create_train_df(
    df,
    btcd,
    btcdom,
    fng,
    ttype,
    configs,
    target_offset,
    first,
    last,
    step,
    target_high_ratio,
    target_low_ratio,
    train_df_prev=None,
):
    """Create train dataset from signal statistics and ticker candle data.

    For every signal row in `df`, looks up the corresponding OHLCV history,
    attaches a lookback window of lagged features, and computes a binary
    ``target`` label based on whether the trade would have hit TP or SL within
    `target_offset` hours.

    Parameters
    ----------
    df : pd.DataFrame
        Raw signal stat DataFrame with columns ``ticker``, ``time``, ``pattern``.
    btcd : pd.DataFrame
        Daily BTC dominance context (passed to :func:`merge_btc_dominance`).
    btcdom : pd.DataFrame
        4h BTC.D index context (passed to :func:`merge_btc_dominance`).
    fng : pd.DataFrame
        Daily Fear & Greed index context (passed to :func:`merge_fng`).
    ttype : str
        Trade type, ``"buy"`` or ``"sell"``.
    configs : dict
        Full config dict.
    target_offset : int
        Number of hours ahead to evaluate the trade outcome.
    first : int
        First lookback step (hours).
    last : int
        Last lookback step (hours, inclusive).
    step : int
        Step size between lookback windows (hours).
    target_high_ratio : float
        Higher price level (e.g. ``1.02`` for +2 %).
    target_low_ratio : float
        Lower price level (e.g. ``0.98`` for -2 %).
    train_df_prev : pd.DataFrame, optional
        Previously built training DataFrame. When provided, only rows with
        ``time > max(train_df_prev.time)`` per ticker are processed, allowing
        incremental dataset updates.

    Returns
    -------
    pd.DataFrame
        Training dataset with indicator features, lookback columns, and:

        * ``target`` — 1 if the trade was profitable, 0 otherwise.
        * ``first_price`` — entry price.
        * ``last_price`` — closing price at the end of the trade window.
        * ``close_time`` — timestamp when TP or SL was hit.
        * ``max_price_deviation`` — max price move in the favorable direction.
        * ``min_price_deviation`` — max price move in the adverse direction.
    """
    train_df = list()
    tickers = df["ticker"].unique()
    cols_to_scale = configs["Model"]["params"]["cols_to_scale"]

    for ticker in tqdm(tickers):
        # get signals with current ticker
        signal_df = df[df["ticker"] == ticker]
        times = signal_df["time"]

        # load max time for that ticker from the previously created dataset
        if train_df_prev is not None:
            max_time = train_df_prev.loc[train_df_prev["ticker"] == ticker, "time"].max()
        else:
            max_time = None

        # load candle history of this ticker
        tmp_df_1h, tmp_df_4h = get_file(ticker)

        # add indicators
        try:
            tmp_df_1h = add_indicators(tmp_df_1h, ttype, configs, tmp_df_4h)
            tmp_df_1h = merge_btc_dominance(tmp_df_1h, btcd, btcdom)
            tmp_df_1h = merge_fng(tmp_df_1h, fng)
            tmp_df_1h = tmp_df_1h.ffill()  # for higher_features NaNs
            tmp_df_1h[real_price_cols] = tmp_df_1h[["high", "low", "close"]]
            tmp_df_1h = scale_cols(tmp_df_1h, cols_to_scale)
        except TypeError as te:
            print(f"TypeError {te}, ticker - {ticker}")
            continue

        # add historical data for current ticker
        for i, t in enumerate(times.to_list()):
            if max_time and t <= max_time:
                continue

            pass_cycle = False
            pattern = signal_df.iloc[i, signal_df.columns.get_loc("pattern")]
            row = tmp_df_1h.loc[tmp_df_1h["time"] == t, :].reset_index(drop=True)

            parts = []
            for j in range(first, last + 1, step):
                # collect features every 4 hours, save difference between the current feature and the lagged features
                time_prev = t + timedelta(hours=-j)
                try:
                    row_tmp = tmp_df_1h.loc[
                        tmp_df_1h["time"] == time_prev,
                        [c for c in tmp_df_1h.columns if c not in real_price_cols],
                    ].reset_index(drop=True)
                    if j % 8 != 0:
                        row_tmp = row_tmp.drop(columns=funding_cols)
                    if j % 24 != 0:
                        row_tmp = row_tmp.drop(columns=btcd_cols + ["fng_value"])
                    row_tmp.columns = [c + f"_prev_{j}" for c in row_tmp.columns]
                except IndexError:
                    pass_cycle = True
                    break
                parts.append(row_tmp.iloc[:, 1:])

            if pass_cycle:
                continue

            row = pd.concat([row] + parts, axis=1)
            row["ticker"] = ticker
            row["pattern"] = pattern
            row["target"] = 0
            row["max_price_deviation"] = 0
            row["min_price_deviation"] = 0
            row["first_price"] = 0
            row["last_price"] = 0
            row["ttype"] = ttype

            # If ttype = buy and during the selected period high price was higher than close_price * target_ratio
            # and earlier low price wasn't lower than close_price / target_ratio, than target is True, else target is False.
            # Similarly for ttype = sell
            if pattern.startswith("MACD"):
                close_price = tmp_df_1h.loc[
                    tmp_df_1h["time"] == t + timedelta(hours=3), "real_close"
                ]
            else:
                close_price = tmp_df_1h.loc[tmp_df_1h["time"] == t, "real_close"]

            # move to the next ticker if can't find any data corresponding to time t
            if close_price.shape[0] == 0:
                continue

            # only admit signals whose full outcome window is available, so the label is
            # final. A signal fired less than target_offset hours ago still has an open
            # (provisional) outcome that can flip on later runs as new/revised candles
            # arrive; admitting it would make incremental updates diverge from a fresh build.
            if pattern.startswith("MACD"):
                final_time = t + timedelta(hours=3 + target_offset)
            else:
                final_time = t + timedelta(hours=target_offset)
            if (tmp_df_1h["time"] == final_time).sum() == 0:
                continue

            row["first_price"] = close_price.values[0]
            row["close_time"] = row["time"].values[0] + pd.to_timedelta(target_offset, unit="h")

            close_price = close_price.values[0]
            target_high_price = close_price * target_high_ratio
            target_lower_price = close_price * target_low_ratio

            real_high_prices, real_low_prices = [], []
            for offset in range(1, target_offset + 1):
                if pattern.startswith("MACD"):
                    time_next = t + timedelta(hours=3 + offset)
                else:
                    time_next = t + timedelta(hours=offset)

                real_high_price = tmp_df_1h.loc[tmp_df_1h["time"] == time_next, "real_high"]
                real_low_price = tmp_df_1h.loc[tmp_df_1h["time"] == time_next, "real_low"]

                real_high_prices.append(real_high_price)
                real_low_prices.append(real_low_price)

                if real_high_price.shape[0] == 0 or real_low_price.shape[0] == 0:
                    pass_cycle = True
                    break

                real_high_price = real_high_price.values[0]
                real_low_price = real_low_price.values[0]

                # set TPs and SLs
                # IMPORTANT !!!
                # because my experience says that buy STOCH_RSI signal sends sell signal
                # and vise versa - sell STOCH_RSI signal sends sell signal
                # both TP and SL are inverted
                if ttype == "buy":
                    if pattern.startswith("STOCH"):
                        TP = real_low_price < target_lower_price
                        SL = real_high_price > target_high_price
                    else:
                        TP = real_high_price > target_high_price
                        SL = real_low_price < target_lower_price
                else:
                    if pattern.startswith("STOCH"):
                        TP = real_high_price > target_high_price
                        SL = real_low_price < target_lower_price
                    else:
                        TP = real_low_price < target_lower_price
                        SL = real_high_price > target_high_price

                # if both TP and SL flag is on - don't consider that trade
                if TP and SL:
                    if row["target"].values[0] == 0:
                        pass_cycle = True
                    break
                elif SL:
                    # if reach SL - write the time when the trade was closed (but only one time)
                    if row["close_time"].values[0] == row["time"].values[0] + pd.to_timedelta(
                        target_offset, unit="h"
                    ):
                        row["close_time"] = row["time"].values[0] + pd.to_timedelta(
                            offset, unit="h"
                        )
                    break
                elif TP:
                    # if reach TP - write the time when the trade was closed (but only one time)
                    if row["close_time"].values[0] == row["time"].values[0] + pd.to_timedelta(
                        target_offset, unit="h"
                    ):
                        row["close_time"] = row["time"].values[0] + pd.to_timedelta(
                            offset, unit="h"
                        )
                    row["target"] = 1

                # if price doesn't reaches both TP and SL thresholds but price above / below enter price for buy / sell trade - set TP flag
                # (depends on ttype and pattern)
                # as mentioned above - for STOCH_RSI signal TP and SL signals are inverted
                if offset == target_offset:
                    last_price = tmp_df_1h.loc[tmp_df_1h["time"] == time_next, "real_close"].values[
                        0
                    ]
                    if pattern.startswith("STOCH"):
                        l1 = ttype == "buy" and last_price < close_price
                        l2 = ttype == "sell" and last_price > close_price
                    else:
                        l1 = ttype == "buy" and last_price > close_price
                        l2 = ttype == "sell" and last_price < close_price
                    # if price doesn't reach both TP and SL - write its last price
                    if row["target"].values[0] == 0:
                        row["last_price"] = last_price

                    if l1 or l2:
                        row["target"] = 1

                # set the maximum price deviation to the correct side for the current trade period
                if ttype == "sell":
                    curr_price_pos = (real_high_price - close_price) / close_price
                    curr_price_neg = (close_price - real_low_price) / close_price
                else:
                    curr_price_pos = (close_price - real_low_price) / close_price
                    curr_price_neg = (real_high_price - close_price) / close_price

                row["max_price_deviation"] = max(
                    row["max_price_deviation"].values[0], curr_price_pos
                )
                row["min_price_deviation"] = max(
                    row["min_price_deviation"].values[0], curr_price_neg
                )

            if pass_cycle:
                continue

            # add data to the dataset
            train_df.append(row)

    train_df = pd.concat(train_df).reset_index(drop=True)
    train_df = train_df.drop(columns=real_price_cols)
    return train_df


def q10(x):
    return x.quantile(0.1)


def q20(x):
    return x.quantile(0.2)


def q30(x):
    return x.quantile(0.3)


def q90(x):
    return x.quantile(0.9)


def _trust_interval(row, z=1.95):
    """Wilson-style trust interval for a Bernoulli proportion (lower, upper)."""
    sum_, val1 = row["total"], row["count"]
    val2 = sum_ - val1
    n = val1 + val2
    p = val1 / n
    low_bound = p - z * np.sqrt(p * (1 - p) / n)
    high_bound = p + z * np.sqrt(p * (1 - p) / n)
    return round(low_bound, 4), round(high_bound, 4)


def _profitable_hours_ti(df: pd.DataFrame, TI_low_bound: float) -> list:
    """Profitable hours by Trust Interval (TI).

    Pivots `df` by hour-of-day and target, computes the trust interval of the
    profitable ratio per hour, and keeps hours whose lower bound is at least
    `TI_low_bound`.
    """
    pvt = df[["target", "pattern", "time", "max_price_deviation"]].copy()
    pvt["hour"] = pvt["time"].dt.hour
    pvt = pvt.pivot_table(
        index=["hour", "target"],
        values=["pattern", "max_price_deviation"],
        aggfunc={
            "pattern": "count",
            "max_price_deviation": ["median", q10, q20, q30, q90],
        },
    ).reset_index()
    pvt.columns = [
        "hour",
        "target",
        "max_price_dev_q50",
        "max_price_dev_q10",
        "max_price_dev_q20",
        "max_price_dev_q30",
        "max_price_dev_q90",
        "pattern",
    ]
    pvt["total"] = pvt.groupby("hour")["pattern"].transform("sum")
    pvt = pvt.rename(columns={"pattern": "count"})
    pvt = pvt[pvt["target"] == 1]
    pvt["trust_interval"] = pvt.apply(_trust_interval, axis=1)
    mask = pvt["trust_interval"].apply(lambda x: x[0]) >= TI_low_bound
    return pvt.loc[mask, "hour"].tolist()


def _profitable_hours_rm(df: pd.DataFrame, RM_percent_above_0_5: float) -> list:
    """Profitable hours by Rolling Mean (RM).

    For each hour-of-day, takes the 168-period rolling mean of the target and
    keeps hours whose rolling mean stays above 0.5 for at least
    `RM_percent_above_0_5` percent of the time.
    """
    pvt = df[["time", "target"]].copy()
    pvt["hour"] = pvt["time"].dt.hour
    pvt = pvt.pivot_table(index="time", columns="hour", values="target", aggfunc="mean")

    results = []
    for hour in range(24):
        if hour in pvt.columns:
            valid_rolling = pvt[hour].dropna().rolling(window=168).mean().dropna()
            pct_above = (valid_rolling > 0.5).mean() * 100 if len(valid_rolling) > 0 else np.nan
        else:
            pct_above = np.nan
        results.append({"hour": hour, "percent_above_0_5": pct_above})

    summary = pd.DataFrame(results)
    return summary.query("percent_above_0_5 >= @RM_percent_above_0_5")["hour"].tolist()


def find_profitable_hours(
    df_buy: pd.DataFrame,
    df_sell: pd.DataFrame = None,
    test_date=None,
    TI_low_bound: float = 0.495,
    RM_percent_above_0_5: float = 60,
    path: str = "data/context/profitable_hours.csv",
) -> pd.DataFrame:
    """Compute profitable buy/sell hours and append them to a CSV.

    Profitable hours are found two ways and intersected: Trust Interval (TI,
    see :func:`_profitable_hours_ti`) and Rolling Mean (RM, see
    :func:`_profitable_hours_rm`). The result is appended as one new row to
    `path` (created if absent).

    Parameters
    ----------
    df_buy : pd.DataFrame
        Either the buy training set, or — when `df_sell` is None — a combined
        training set holding both buy and sell rows (split via the ``ttype``
        column).
    df_sell : pd.DataFrame, optional
        The sell training set. If None, `df_buy` is treated as combined.
    test_date : optional
        Only rows with ``time < test_date`` are used. If None, the whole
        dataset is used.
    TI_low_bound : float
        Minimum trust-interval lower bound for an hour to count as profitable.
    RM_percent_above_0_5 : float
        Minimum percent of time the rolling mean must stay above 0.5.
    path : str
        Destination CSV path.

    Returns
    -------
    pd.DataFrame
        The single new row that was appended.
    """
    if df_sell is None:
        df_sell = df_buy[df_buy["ttype"] == "sell"]
        df_buy = df_buy[df_buy["ttype"] == "buy"]

    if test_date is not None:
        df_buy = df_buy[df_buy["time"] < test_date]
        df_sell = df_sell[df_sell["time"] < test_date]

    profitable_buy_hours_TI = _profitable_hours_ti(df_buy, TI_low_bound)
    profitable_sell_hours_TI = _profitable_hours_ti(df_sell, TI_low_bound)
    profitable_buy_hours_RM = _profitable_hours_rm(df_buy, RM_percent_above_0_5)
    profitable_sell_hours_RM = _profitable_hours_rm(df_sell, RM_percent_above_0_5)

    profitable_buy_hours = sorted(set(profitable_buy_hours_RM) & set(profitable_buy_hours_TI))
    profitable_sell_hours = sorted(set(profitable_sell_hours_RM) & set(profitable_sell_hours_TI))

    new_row = pd.DataFrame(
        [
            {
                "time": datetime.now(),
                "test_date": test_date,
                "TI_low_bound": TI_low_bound,
                "RM_percent_above_0_5": RM_percent_above_0_5,
                "profitable_buy_hours_RM": profitable_buy_hours_RM,
                "profitable_buy_hours_TI": profitable_buy_hours_TI,
                "profitable_buy_hours": profitable_buy_hours,
                "profitable_sell_hours_RM": profitable_sell_hours_RM,
                "profitable_sell_hours_TI": profitable_sell_hours_TI,
                "profitable_sell_hours": profitable_sell_hours,
            }
        ]
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        profitable_hours_df = pd.concat([pd.read_csv(path), new_row], ignore_index=True)
    else:
        profitable_hours_df = new_row
    profitable_hours_df.to_csv(path, index=False)

    return new_row
