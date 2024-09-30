"""
This module provides functionality for interacting with the Binance cryptocurrency
exchange API. It defines the `OKEX Swap` class, which extends the `ApiBase` class to
retrieve and manipulate market data such as ticker symbols, K-line (candlestick)
data, and historical data for specified intervals.
"""

from datetime import datetime
from typing import List, Tuple

import pandas as pd
import requests
from api.api_base import ApiBase


class OKEXSwap(ApiBase):
    """Class for accessing OKEXSwap cryptocurrency exchange API"""

    URL = "https://www.okex.com"

    def get_ticker_names(self, min_volume) -> Tuple[List[str], List[float], List[str]]:
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
            requests.get(
                self.URL + "/api/v5/market/tickers?instType=SWAP", timeout=3
            ).json()["data"]
        )
        tickers["symbol"] = (
            tickers["instId"].str.replace("-", "").str.replace("SWAP", "")
        )
        all_tickers = tickers["symbol"].to_list()

        tickers = tickers[
            (tickers["instId"].str.endswith("USDT-SWAP"))
            | (tickers["instId"].str.endswith("USDC-SWAP"))
        ]
        # meaning of vol24h is different between SPOT and SWAP
        tickers["volCcy24h"] = tickers["volCcy24h"].astype(float)
        tickers["last"] = tickers["last"].astype(float)
        ticker_vol = tickers["volCcy24h"] * tickers["last"]
        tickers = tickers[ticker_vol >= min_volume // 2]

        filtered_symbols = self.check_symbols(tickers["instId"])
        tickers = tickers[tickers["instId"].isin(filtered_symbols)]
        filtered_symbols = self.delete_duplicate_symbols(tickers["instId"])
        tickers = tickers[tickers["instId"].isin(filtered_symbols)].reset_index(
            drop=True
        )
        tickers = tickers.drop_duplicates(subset=["instId"])

        return tickers["instId"].to_list(), tickers["volCcy24h"].to_list(), all_tickers

    def get_klines(self, symbol: str, interval: str, limit=200) -> pd.DataFrame:
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

        if not interval.endswith("m"):
            interval = interval.upper()
        params = {"instId": symbol, "bar": interval, "limit": limit}
        tickers = pd.DataFrame(
            requests.get(
                self.URL + "/api/v5/market/candles", params=params, timeout=3
            ).json()["data"]
        )
        # at first time get candles from previous interval
        # to overcome API limit restrictions
        if limit > 100:
            after = tickers.iloc[0, 0]
            after = int(after) - (limit - 1) * interval_secs * 1000
            params["after"] = str(after)
            tmp = pd.DataFrame(
                requests.get(
                    self.URL + "/api/v5/market/candles", params=params, timeout=3
                ).json()["data"]
            )
            if tmp.shape[0] > 0:
                tickers = pd.concat([tickers, tmp])

        tickers = tickers.rename(
            {0: "time", 1: "open", 2: "high", 3: "low", 4: "close", 6: "volume"}, axis=1
        )
        return tickers[["time", "open", "high", "low", "close", "volume"]][
            ::-1
        ].reset_index(drop=True)

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
        # maximum limit for this endpoint is 100
        limit = 100
        interval_secs = self.convert_interval_to_secs(interval)
        if not interval.endswith("m"):
            interval = interval.upper()
        params = {"instId": symbol, "bar": interval, "limit": limit}
        tmp_limit = 0
        prev_time, earliest_time = None, datetime.now()
        timestamp_ = int(self.get_timestamp() / 3600) * 3600
        tickers = pd.DataFrame()

        while earliest_time > min_time:
            after = (timestamp_ - tmp_limit * interval_secs) * 1000
            params["after"] = str(after)
            tmp = pd.DataFrame(
                requests.get(
                    self.URL + "/api/v5/market/history-candles",
                    params=params,
                    timeout=3,
                ).json()["data"]
            )
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_candlestick_time = earliest_time, tmp[0].min()
            earliest_time = self.convert_timestamp_to_time(
                int(earliest_candlestick_time), unit="ms"
            )
            # prevent endless cycle if there are no candles
            # that are earlier than min_time
            if prev_time == earliest_time:
                break

            # drop duplicated rows
            if tickers.shape[0] > 0:
                tickers = tickers[tickers[0] > tmp[0].max()]
            tickers = pd.concat([tickers, tmp])
            tmp_limit += limit

        tickers = tickers.sort_values(0, ignore_index=True)
        tickers = tickers.rename(
            {0: "time", 1: "open", 2: "high", 3: "low", 4: "close", 6: "volume"}, axis=1
        )
        return tickers[["time", "open", "high", "low", "close", "volume"]].reset_index(
            drop=True
        )

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
        interval_secs = 8 * 3600 * 1000
        before = int(self.get_timestamp() / 3600) * 3600 * 1000
        limit = 100
        params = {"instId": symbol, "limit": limit}
        prev_time, earliest_time = None, datetime.now()
        funding_rates = pd.DataFrame()

        while earliest_time > min_time:
            before = before - (limit * interval_secs)
            params["before"] = str(before)
            tmp = pd.DataFrame(
                requests.get(
                    self.URL + "/api/v5/public/funding-rate-history",
                    params=params,
                    timeout=3,
                ).json()["data"]
            )
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp["fundingTime"].min()
            earliest_time = self.convert_timestamp_to_time(earliest_time, unit="ms")
            # prevent endless cycle if there are no candles that earlier than min_time
            if prev_time == earliest_time:
                break

            # drop duplicated rows
            if funding_rates.shape[0] > 0:
                funding_rates = funding_rates[
                    funding_rates["fundingTime"] > tmp["fundingTime"].max()
                ]

            funding_rates = pd.concat([funding_rates, tmp])

        funding_rates = funding_rates.rename(
            {"fundingTime": "time", "fundingRate": "funding_rate"}, axis=1
        )
        return funding_rates[["time", "funding_rate"]][::-1].reset_index(drop=True)
