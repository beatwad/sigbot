import time
from datetime import datetime
from typing import Any

import ccxt
import pandas as pd
from tqdm import tqdm


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
