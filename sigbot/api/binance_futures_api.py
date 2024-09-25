from typing import Tuple, List

from datetime import datetime
import pandas as pd
from api.api_base import ApiBase
from binance.client import Client


class BinanceFutures(ApiBase):
    client: Client = ""

    def __init__(self, api_key: str = "Key", api_secret: str = "Secret"):
        """
        Initialize the Binance Futures API connection.

        Parameters
        ----------
        api_key : str, optional
            The API key for the Binance account (default is "Key").
        api_secret : str, optional
            The API secret for the Binance account (default is "Secret").
        """
        if api_key != "Key" and api_secret != "Secret":
            self.connect_to_api(api_key, api_secret)
        else:
            self.api_key = api_key
            self.api_secret = api_secret

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
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = Client(self.api_key, api_secret)

    @staticmethod
    def delete_duplicate_symbols(symbols: pd.Series) -> List[str]:
        """
        Remove duplicate symbols where pairs with USDT exist and delete the corresponding BUSD pairs.

        Parameters
        ----------
        symbols : pd.Series
            A series of cryptocurrency symbols.

        Returns
        -------
        list
            A list of filtered symbols without duplicates (BUSD pairs removed if USDT exists).
        """
        filtered_symbols = list()
        symbols = symbols.to_list()

        for symbol in symbols:
            if symbol.endswith('BUSD'):
                prefix = symbol[:-4]
                if prefix + 'USDT' not in symbols:
                    filtered_symbols.append(symbol)
            else:
                filtered_symbols.append(symbol)
        return filtered_symbols

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
        tickers = pd.DataFrame(self.client.futures_ticker())
        all_tickers = tickers['symbol'].to_list()

        tickers = tickers[(tickers['symbol'].str.endswith('USDT')) | (tickers['symbol'].str.endswith('BUSD'))]
        tickers['quoteVolume'] = tickers.loc[:, 'quoteVolume'].astype(float)
        tickers = tickers[tickers['quoteVolume'] >= min_volume // 2]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        filtered_symbols = self.delete_duplicate_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list(), tickers['volume'].to_list(), all_tickers

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
        tickers = pd.DataFrame(self.client.futures_klines(symbol=symbol, interval=interval, limit=limit))
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 7: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']]

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
        tmp_limit = limit
        prev_time, earliest_time = None, datetime.now()
        ts = int(self.get_timestamp() / 3600) * 3600
        tickers = pd.DataFrame()

        # Get historical data in a loop until the min_time is reached
        while earliest_time > min_time:
            start_time = (ts - (tmp_limit * interval_secs)) * 1000
            tmp = pd.DataFrame(self.client.futures_klines(symbol=symbol, interval=interval, startTime=start_time,
                                                          limit=limit))
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp[0].min()
            earliest_time = self.convert_timstamp_to_time(earliest_time, unit='ms')

            # Prevent an endless loop if no earlier candles exist
            if prev_time == earliest_time:
                break

            # Drop duplicated rows
            if tickers.shape[0] > 0:
                tickers = tickers[tickers[0] > tmp[0].max()]
            tickers = pd.concat([tmp, tickers])
            tmp_limit += limit

        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 7: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)

    def get_historical_funding_rate(self, symbol: str, limit: int, min_time: datetime) -> pd.DataFrame:
        """
        Retrieve historical funding rate information for a cryptocurrency pair before a specified minimum time.

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
        prev_time, earliest_time = None, datetime.now()
        start_time = int(self.get_timestamp() / 3600) * 3600 * 1000
        funding_rates = pd.DataFrame()

        while earliest_time > min_time:
            start_time, end_time = (start_time - (limit * interval_secs)), start_time
            tmp = pd.DataFrame(self.client.futures_funding_rate(symbol=symbol, limit=limit, startTime=start_time,
                                                                endTime=end_time))
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp['fundingTime'].min()
            earliest_time = self.convert_timstamp_to_time(earliest_time, unit='ms')

            # Prevent an endless loop if no earlier funding rates exist
            if prev_time == earliest_time:
                break

            # Drop duplicated rows
            if funding_rates.shape[0] > 0:
                funding_rates = funding_rates[funding_rates['fundingTime'] > tmp['fundingTime'].max()]

            funding_rates = pd.concat([tmp, funding_rates])
        funding_rates = funding_rates.rename({'fundingTime': 'time', 'fundingRate': 'funding_rate'}, axis=1)
        funding_rates['time'] = (funding_rates['time'] // 1000) * 1000
        return funding_rates[['time', 'funding_rate']].reset_index(drop=True)


if __name__ == '__main__':
    key = "7arxKITvadhYavxsQr5dZelYK4kzyBGM4rsjDCyJiPzItNlAEdlqOzibV7yVdnNy"
    secret = "3NvopCGubDjCkF4SzqP9vj9kU2UIhE4Qag9ICUdESOBqY16JGAmfoaUIKJLGDTr4"
    binance_api = BinanceFutures(key, secret)
    klines = binance_api.get_klines('BTCUSDT', '1h', 300)
