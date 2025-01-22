"""
This module provides functionality for interacting with the Binance cryptocurrency
exchange API. It defines the `MEXCFutures` class, which extends the `ApiBase` class to
retrieve and manipulate market data such as ticker symbols, K-line (candlestick)
data, and historical data for specified intervals.
"""

from datetime import datetime
from typing import Dict, List, Tuple, Union

import pandas as pd
import requests

from api.api_base import ApiBase


class MEXCFutures(ApiBase):
    """Class for accessing MEXC Futures cryptocurrency exchange API"""

    URL = "https://contract.mexc.com/api/v1/contract/"
    interval_dict = {
        "1m": "Min1",
        "5m": "Min5",
        "15m": "Min15",
        "30m": "Min30",
        "1h": "Min60",
        "4h": "Hour4",
        "8h": "Hour8",
        "1d": "Day1",
        "1w": "Week1",
    }

    def get_ticker_names(self, min_volume: float) -> Tuple[List[str], List[float], List[str]]:
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
        tickers = pd.DataFrame(requests.get(self.URL + "/ticker", timeout=3).json()["data"])
        tickers = tickers[tickers["symbol"].str.endswith("USDT")]

        all_tickers = tickers["symbol"].str.replace("_", "").to_list()

        tickers["amount24"] = tickers["amount24"].astype(float)
        tickers = tickers[tickers["amount24"] >= min_volume // 2]

        filtered_symbols = self.check_symbols(tickers["symbol"])
        tickers = tickers[tickers["symbol"].isin(filtered_symbols)]
        tickers = tickers[tickers["symbol"].isin(filtered_symbols)].reset_index(drop=True)

        return tickers["symbol"].to_list(), tickers["volume24"].to_list(), all_tickers

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
            DataFrame containing time, open, high, low, close,
            and volume for the specified symbol.
        """
        interval_secs = self.convert_interval_to_secs(interval)
        start = self.get_timestamp() - (limit * interval_secs)
        interval = self.interval_dict[interval]

        params = {"interval": interval, "start": start}
        tickers = pd.DataFrame(
            requests.get(self.URL + f"/kline/{symbol}", params=params, timeout=3).json()["data"]
        )
        tickers = tickers.rename({"vol": "volume"}, axis=1)
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
            DataFrame containing historical time, open, high, low, close,
            and volume for the specified symbol.
        """
        interval_secs = self.convert_interval_to_secs(interval)
        interval = self.interval_dict[interval]
        params = {"interval": interval, "limit": limit}
        tmp_limit = limit
        prev_time, earliest_time = None, datetime.now()
        timestamp_ = int(self.get_timestamp() / 3600) * 3600
        tickers = pd.DataFrame()

        while earliest_time > min_time:
            start_time = timestamp_ - (tmp_limit * interval_secs)
            end_time = timestamp_ - ((tmp_limit - limit) * interval_secs)
            params["start"] = start_time
            params["end"] = end_time
            tmp = pd.DataFrame(
                requests.get(self.URL + f"/kline/{symbol}", params=params, timeout=3).json()["data"]
            )
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp["time"].min()
            earliest_time = self.convert_timestamp_to_time(earliest_time, unit="s")
            # prevent endless cycle if there are no candles that earlier than min_time
            if prev_time == earliest_time:
                break

            # drop duplicated rows
            if tickers.shape[0] > 0:
                tickers = tickers[tickers["time"] > tmp["time"].max()]
            tickers = pd.concat([tmp, tickers])
            tmp_limit += limit

        tickers = tickers.rename(
            {
                0: "time",
                1: "open",
                2: "high",
                3: "low",
                4: "close",
                5: "asset_volume",
                6: "close_time",
                "vol": "volume",
            },
            axis=1,
        )
        tickers = tickers.sort_values("time", ignore_index=True)
        return tickers[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def get_historical_funding_rate(
        self, symbol: str, limit: int, min_time: datetime
    ) -> pd.DataFrame:
        """
        Retrieve historical funding rate information for a cryptocurrency pair
        before a specified minimum time.

        Parameters
        ----------
        symbol : str
            The symbol of the cryptocurrency pair (e.g., 'BTCUSDT').
        limit : int
            The maximum number of data points to retrieve in each request.
        min_time : datetime
            The earliest time for which data should be retrieved.

        Returns
        -------
        pd.DataFrame
            DataFrame containing time and funding rate for the specified symbol.
        """
        params: Dict[str, Union[str, int]] = {
            "symbol": symbol,
            "page_num": 1,
            "page_size": 100,
        }
        prev_time, earliest_time = None, datetime.now()
        funding_rates = pd.DataFrame()

        while earliest_time > min_time:
            tmp = pd.DataFrame(
                requests.get(self.URL + "/funding_rate/history", params=params, timeout=3).json()[
                    "data"
                ]["resultList"]
            )
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp["settleTime"].min()
            earliest_time = self.convert_timestamp_to_time(earliest_time, unit="ms")
            # prevent endless cycle if there are no candles that earlier than min_time
            if prev_time == earliest_time:
                break

            # drop duplicated rows
            if funding_rates.shape[0] > 0:
                funding_rates = funding_rates[funding_rates["settleTime"] > tmp["settleTime"].max()]

            funding_rates = pd.concat([funding_rates, tmp])
            params["page_num"] = int(params["page_num"]) + 1

        funding_rates = funding_rates.rename(
            {"settleTime": "time", "fundingRate": "funding_rate"}, axis=1
        )
        return funding_rates[["time", "funding_rate"]][::-1].reset_index(drop=True)
