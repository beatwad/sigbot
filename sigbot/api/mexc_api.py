"""
This module provides functionality for interacting with the Binance cryptocurrency
exchange API. It defines the `MEXC` class, which extends the `ApiBase` class to
retrieve and manipulate market data such as ticker symbols, K-line (candlestick)
data, and historical data for specified intervals.
"""

from datetime import datetime
from typing import List, Tuple

import pandas as pd
import requests
from api.api_base import ApiBase


class MEXC(ApiBase):
    """Class for accessing MEXC cryptocurrency exchange API"""

    URL = "https://api.mexc.com/api/v3"

    def get_ticker_names(
        self, min_volume: float
    ) -> Tuple[List[str], List[float], List[str]]:
        """
        Get ticker symbols and their corresponding volumes,
        filtering by a minimum volume.

        Parameters
        ----------
        min_volume : float
            The minimum volume to filter tickers.

        Returns
        -------
        tuple of lists
            A tuple containing:
            - A list of filtered symbols.
            - A list of their respective volumes.
            - A list of all symbols before filtering.
        """
        tickers = pd.DataFrame(
            requests.get(self.URL + "/ticker/24hr", timeout=3).json()
        )
        tickers = tickers[
            (tickers["symbol"].str.endswith("USDT"))
            | (tickers["symbol"].str.endswith("USDC"))
        ]

        all_tickers = tickers["symbol"].to_list()

        tickers["quoteVolume"] = tickers["quoteVolume"].astype(float)
        tickers = tickers[tickers["quoteVolume"] >= min_volume // 2]

        filtered_symbols = self.check_symbols(tickers["symbol"])
        tickers = tickers[tickers["symbol"].isin(filtered_symbols)]
        filtered_symbols = self.delete_duplicate_symbols(tickers["symbol"])
        tickers = tickers[tickers["symbol"].isin(filtered_symbols)].reset_index(
            drop=True
        )

        return tickers["symbol"].to_list(), tickers["volume"].to_list(), all_tickers

    def get_klines(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        """
        Retrieve K-line (candlestick) data for a given symbol and interval.

        Parameters
        ----------
        symbol : str
            The symbol of the cryptocurrency (e.g., 'BTCUSDT').
        interval : str
            The interval for the K-lines (e.g., '1h', '1d').
        limit : int
            The maximum number of data points to retrieve.

        Returns
        -------
        pd.DataFrame
            DataFrame containing time, open, high, low, close, and
            volume for the specified symbol.
        """
        if interval == "1h":
            interval = "60m"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        tickers = pd.DataFrame(
            requests.get(self.URL + "/klines", params=params, timeout=3).json()
        )
        tickers = tickers.rename(
            {0: "time", 1: "open", 2: "high", 3: "low", 4: "close", 7: "volume"}, axis=1
        )
        return tickers[["time", "open", "high", "low", "close", "volume"]]

    def get_historical_klines(
        self, symbol: str, interval: str, limit: int, min_time: datetime
    ) -> pd.DataFrame:
        """
        Retrieve historical K-line data for a given symbol and
        interval before a specified minimum time.

        Parameters
        ----------
        symbol : str
            The symbol of the cryptocurrency (e.g., 'BTCUSDT').
        interval : str
            The interval for the K-lines (e.g., '1h', '1d').
        limit : int
            The maximum number of data points to retrieve in each request.
        min_time : datetime
            The earliest time for which data should be retrieved.

        Returns
        -------
        pd.DataFrame
            DataFrame containing historical time, open, high, low, close, and
            volume for the specified symbol.
        """
        interval_secs = self.convert_interval_to_secs(interval)
        if interval == "1h":
            interval = "60m"
        params = {"symbol": symbol, "interval": interval, "limit": limit, "timeout": 3}
        tmp_limit = limit
        prev_time, earliest_time = None, datetime.now()
        timestamp_ = int(self.get_timestamp() / 3600) * 3600
        tickers = pd.DataFrame()

        while earliest_time > min_time:
            start_time = (timestamp_ - (tmp_limit * interval_secs)) * 1000
            end_time = (timestamp_ - ((tmp_limit - limit) * interval_secs)) * 1000
            params["startTime"] = start_time
            params["endTime"] = end_time
            tmp = pd.DataFrame(
                requests.get(self.URL + "/klines", params=params, timeout=3).json()
            )
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp[0].min()
            earliest_time = self.convert_timestamp_to_time(earliest_time, unit="ms")
            # prevent endless cycle if there are no candles that earlier than min_time
            if prev_time == earliest_time:
                break

            # drop duplicated rows
            if tickers.shape[0] > 0:
                tickers = tickers[tickers[0] > tmp[0].max()]
            tickers = pd.concat([tmp, tickers])
            tmp_limit += limit

        tickers = tickers.rename(
            {0: "time", 1: "open", 2: "high", 3: "low", 4: "close", 7: "volume"}, axis=1
        )
        tickers = tickers.sort_values("time", ignore_index=True)
        return tickers[["time", "open", "high", "low", "close", "volume"]].reset_index(
            drop=True
        )


# if __name__ == "__main__":
#     mexc = MEXC()
#     min_time = datetime.now().replace(microsecond=0, second=0, minute=0) -\
#           pd.to_timedelta(365 * 10, unit="D")
#     data = mexc.get_historical_klines("FURUCOMBOUSDT", "4h", 200, min_time)
#     data.to_pickle("FURUCOMBOUSDT_1h.pkl")
