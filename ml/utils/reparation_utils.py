import os
import shutil
import time
from datetime import datetime
from typing import Any

import ccxt
import pandas as pd
import requests
from tqdm import tqdm

FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
FNG_CSV = "data/context/fng.csv"
TRAIN_FILES = {
    "buy": "data/signal_stat/train_buy_272.pkl",
    "sell": "data/signal_stat/train_sell_272.pkl",
}

# Lookback steps that create_train_df keeps for daily context (j in [4, 272], j % 24 == 0).
FNG_LAGS = [j for j in range(4, 273, 4) if j % 24 == 0]


def fetch_all_ohlcv(
    symbol: str, timeframe: str, btcdom_cols: list, since_str: str = "2021-01-01"
) -> None:
    """
    Fetch all OHLCV data for a BTC domination ticker.
    Use it when btcdom.csv is missing.

    Parameters
    ----------
    symbol : str
        The symbol of the cryptocurrency (e.g., 'BTCDOM/USDT:USDT').
    timeframe : str
        The timeframe for the K-lines (e.g., '4h').
    btcdom_cols : list
        The column names for the K-lines.
    since_str : str, optional
        The start date for the K-lines (e.g., '2021-01-01').
    """
    exchange = ccxt.binanceusdm()
    since = exchange.parse8601(f"{since_str}T00:00:00Z")
    all_ohlcv = []

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        print(
            f"Получено: {len(all_ohlcv)} свечей, последняя: {pd.to_datetime(ohlcv[-1][0], unit='ms')}"
        )

        if len(ohlcv) < 1000:
            break

        time.sleep(0.2)  # rate limit

    df = pd.DataFrame(all_ohlcv, columns=btcdom_cols)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df["time"] = df["time"] + pd.to_timedelta(3, unit="h")
    df = df.iloc[:-1]
    df.to_csv("data/btcdom.csv", index=False)


def load_historical_data_from_exchange(exchange: Any, tickers: list):
    """
    Load historical data from any exchange.
    Use it when some ticker data was missed during initial loading.

    Parameters
    ----------
    exchange : Any
        The exchange class to load data from.
    tickers : list
        The list of tickers to load data for.
    timeframe : str
        The timeframe for the K-lines (e.g., '1h', '4h').
    """

    min_time = datetime.now().replace(microsecond=0, second=0, minute=0) - pd.to_timedelta(
        365 * 10, unit="D"
    )

    for ticker in tqdm(tickers):
        for timeframe in ["1h", "4h"]:
            try:
                klines = exchange.get_historical_klines(ticker, "1h", 1000, min_time)
            except ValueError:
                print(ticker)
                break
            except KeyboardInterrupt:
                return
            except Exception as e:
                print(ticker)
                print(e)
                break

            klines["funding_rate"] = 0.0
            klines["time"] = pd.to_datetime(klines["time"], unit="ms")
            ohlcv = ["open", "high", "low", "close", "volume", "funding_rate"]
            klines[ohlcv] = klines[ohlcv].astype(float)
            klines.to_pickle(f"data/tickers/{ticker}_{timeframe}.pkl")


def load_fng() -> pd.DataFrame:
    """Read FNG from the context CSV, falling back to the live API."""
    if os.path.exists(FNG_CSV):
        fng = pd.read_csv(FNG_CSV, parse_dates=["time"])
    else:
        data = requests.get(FNG_URL, timeout=30).json()["data"]
        fng = pd.DataFrame(data)
        # +3h to match the UTC+3 basis the rest of the pipeline uses (see add_utc_3)
        fng["time"] = pd.to_datetime(fng["timestamp"].astype(int), unit="s") + pd.to_timedelta(
            3, unit="h"
        )
        fng["fng_value"] = fng["value"].astype(int)
        fng = fng[["time", "fng_value"]]
    return fng.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def fng_at(times: pd.Series, fng_sorted: pd.DataFrame) -> pd.Series:
    """Return the FNG value in effect at each timestamp.

    Mirrors merge-on-time + forward-fill from create_train_df: the value of the
    most recent daily FNG observation at or before each time. Original row order
    is preserved.
    """
    left = pd.DataFrame({"time": times}).reset_index()
    merged = pd.merge_asof(left.sort_values("time"), fng_sorted, on="time", direction="backward")
    return merged.set_index("index").sort_index()["fng_value"]


def add_fng_columns(df: pd.DataFrame, fng_sorted: pd.DataFrame) -> pd.DataFrame:
    """Add fng_value and its daily lookback lags to df, matching a rebuild."""
    # idempotency: strip any previously-added FNG columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("fng_value")], errors="ignore")

    df["fng_value"] = fng_at(df["time"], fng_sorted).values
    for j in FNG_LAGS:
        df[f"fng_value_prev_{j}"] = fng_at(
            df["time"] - pd.to_timedelta(j, unit="h"), fng_sorted
        ).values
    return df


def backfill_fng() -> None:
    """One-time backfill of the Fear & Greed index into the built training datasets.

    train_buy_272.pkl / train_sell_272.pkl were created before FNG existed, so they
    lack the ``fng_value`` column and its daily lags. The regular prepare_data.ipynb
    update path only appends *new* signals, leaving historical rows with NaN FNG. This
    adds the FNG columns to every existing row so the datasets match a from-scratch
    rebuild. Idempotent: re-running drops any existing FNG columns and recomputes them.
    """
    fng_sorted = load_fng()
    print(
        f"FNG: {len(fng_sorted)} rows, {fng_sorted['time'].min().date()} -> "
        f"{fng_sorted['time'].max().date()}"
    )
    print(
        f"Adding columns: fng_value + fng_value_prev_{{{FNG_LAGS[0]}..{FNG_LAGS[-1]}}} "
        f"({1 + len(FNG_LAGS)} total)\n"
    )

    for ttype, path in TRAIN_FILES.items():
        df = pd.read_pickle(path)
        before_rows, before_cols = df.shape

        df = add_fng_columns(df, fng_sorted)

        fng_cols = [c for c in df.columns if c.startswith("fng_value")]
        nan_rows = int(df[fng_cols].isnull().any(axis=1).sum())
        if nan_rows:
            # signals older than FNG coverage — a rebuild would dropna() these
            df = df.dropna(subset=fng_cols).reset_index(drop=True)
            print(f"{ttype}: dropped {nan_rows} rows with no FNG coverage")

        # safety backup (only the first time, so a good copy is never clobbered)
        bak = path + ".pre_fng.bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        df.to_pickle(path)

        print(
            f"{ttype}: {before_rows}x{before_cols} -> {df.shape[0]}x{df.shape[1]} "
            f"(fng range {int(df['fng_value'].min())}-{int(df['fng_value'].max())}, "
            f"backup: {bak})"
        )
