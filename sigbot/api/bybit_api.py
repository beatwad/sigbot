from typing import Tuple, List

from datetime import datetime
import pandas as pd
from api.api_base import ApiBase
from pybit import unified_trading


class ByBit(ApiBase):
    client = ""

    def __init__(self, api_key: str = "Key", api_secret: str = "Secret"):
        """
        Initialize the ByBit API connection.

        Parameters
        ----------
        api_key : str, optional
            The API key for the ByBit account (default is "Key").
        api_secret : str, optional
            The API secret for the ByBit account (default is "Secret").
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.connect_to_api('', '')

    def connect_to_api(self, api_key: str, api_secret: str):
        """
        Connect to the Binance Futures API using the provided credentials.

        Parameters
        ----------
        api_key : str
            The API key for the Binance account.
        api_secret : str
            The API secret for the Binance account.
        """
        self.client = unified_trading.HTTP(api_key=api_key, api_secret=api_secret)

    def get_ticker_names(self, min_volume: float) -> Tuple[List[str], List[float], List[str]]:
        """
        Get ticker symbols and their corresponding volumes, filtering by a minimum volume.

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
        tickers = pd.DataFrame(self.client.get_tickers(category="spot")['result']['list'])
        all_tickers = tickers['symbol'].to_list()

        tickers = tickers[(tickers['symbol'].str.endswith('USDT')) | (tickers['symbol'].str.endswith('USDC'))]
        tickers['volume24h'] = tickers['volume24h'].astype(float)
        tickers['lastPrice'] = tickers['lastPrice'].astype(float)
        ticker_vol = tickers['volume24h'] * tickers['lastPrice']
        tickers = tickers[ticker_vol >= min_volume // 2]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        filtered_symbols = self.delete_duplicate_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list(), tickers['volume24h'].to_list(), all_tickers

    def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
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
            DataFrame containing time, open, high, low, close, and volume for the specified symbol.
        """
        interval = self.convert_interval(interval)
        tickers = pd.DataFrame(self.client.get_kline(category='spot', symbol=symbol,
                                                     interval=interval, limit=limit)['result']['list'])
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 6: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']][::-1].reset_index(drop=True)

    def get_historical_klines(self, symbol: str, interval: str, limit: int, min_time: datetime) -> pd.DataFrame:
        """
        Retrieve historical K-line data for a given symbol and interval before a specified minimum time.

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
            DataFrame containing historical time, open, high, low, close, and volume for the specified symbol.
        """
        interval_secs = self.convert_interval_to_secs(interval)
        interval = self.convert_interval(interval)
        tmp_limit = limit
        prev_time, earliest_time = None, datetime.now()
        ts = int(self.get_timestamp() / 3600) * 3600
        tickers = pd.DataFrame()

        # get historical data in cycle until we reach the min_time
        while earliest_time > min_time:
            start = (ts - (tmp_limit * interval_secs)) * 1000
            tmp = pd.DataFrame(self.client.get_kline(category='spot', symbol=symbol,
                                                     interval=interval, start=start, limit=limit)['result']['list'])
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp[0].min()
            earliest_time = self.convert_timstamp_to_time(earliest_time, unit='ms')
            # prevent endless cycle if there are no candles that earlier than min_time
            if prev_time == earliest_time:
                break
            
            # drop duplicated rows
            if tickers.shape[0] > 0:
                tickers = tickers[tickers[0] > tmp[0].max()]
            tickers = pd.concat([tickers, tmp])
            tmp_limit += limit
            
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 6: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']][::-1].reset_index(drop=True)


if __name__ == '__main__':
    key = ""
    secret = ""
    bybit_api = ByBit()
    tickers_ = bybit_api.get_ticker_names(500000)
    kline_ = bybit_api.get_klines('BTCUSDT', '1h', 300)
    pass
