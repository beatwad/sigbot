import pandas as pd

from indicators import indicators


def add_indicators(
    df: pd.DataFrame,
    ttype: str,
    configs: dict,
    df_higher: pd.DataFrame = None,
) -> pd.DataFrame:
    """Add technical indicators to df. If df_higher is provided, also computes MACD and
    Trend on the higher timeframe and merges the result into df (time-shifted by +3h)."""
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
    """Merge BTC dominance dataframes into df and forward-fill the merged columns."""
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


def scale_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Convert columns to percent change (* 100) to normalise scale across tickers."""
    for c in cols:
        df[c] = df[c].pct_change() * 100
    return df
