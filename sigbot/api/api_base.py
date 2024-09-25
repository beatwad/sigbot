import re
import pandas as pd
from abc import ABCMeta
from datetime import datetime
from typing import List


class ApiBase(metaclass=ABCMeta):
    @staticmethod
    def delete_duplicate_symbols(symbols: List[str]) -> List[str]:
        """
        Remove duplicate symbols where pairs with USDC are replaced by pairs with USDT.

        Parameters
        ----------
        symbols : pd.Series
            A list of cryptocurrency symbols.

        Returns
        -------
        list
            Filtered list of symbols without duplicates 
            (USDC pairs removed if corresponding pair with USDT exists).
        """
        filtered_symbols = list()
        symbols = symbols.to_list()

        for symbol in symbols:
            if symbol.endswith('USDC'):
                prefix = symbol[:-4]
                if prefix + 'USDT' not in symbols:
                    filtered_symbols.append(symbol)
            else:
                filtered_symbols.append(symbol)

        return filtered_symbols

    @staticmethod
    def check_symbols(symbols: List[str]) -> List[str]:
        """
        Filter out symbols that are pairs with fiat, stablecoins, or leverage types.

        Parameters
        ----------
        symbols : list
            A list of cryptocurrency symbols.

        Returns
        -------
        list
            Filtered list of symbols excluding fiat currency, stablecoins, or leverage pairs.
        """
        filtered_symbols = list()
        for symbol in symbols:
            if (symbol.startswith('USD') or symbol.startswith('BUSD') or symbol.startswith('TUSDUS')
                    or symbol.startswith('BTCDOM') or symbol.startswith('BSCYFI')):
                continue
            if (symbol.endswith('USD') and symbol[-4] != 'B') or symbol.endswith('UST'):
                continue
            if re.match(r'.+[23][LS]', symbol) or re.match(r'.+UP-?(BUSD|USD[TC])', symbol) or \
                    re.match(r'.+DOWN-?(BUSD|USD[TC])', symbol):
                continue
            fiat = ['EUR', 'CHF', 'GBP', 'JPY', 'CNY', 'RUB', 'AUD', 'DAI']
            for f in fiat:
                if symbol.startswith(f) and (len(symbol) == 7 or symbol[3] == '-'):
                    break
            else:
                filtered_symbols.append(symbol)
        return filtered_symbols

    @staticmethod
    def get_timestamp() -> datetime.timestamp:
        """
        Get the current timestamp in seconds.

        Returns
        -------
        int
            Timestamp of the current time.
        """
        today_now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        dt = datetime.strptime(today_now, '%Y-%m-%d %H:%M:%S')
        ts = int(dt.timestamp())
        return ts

    @staticmethod
    def convert_interval_to_secs(interval: str) -> int:
        """
        Convert an interval string to seconds.

        Parameters
        ----------
        interval : str
            A string representing time interval (e.g., '1h', '1d', '1w').

        Returns
        -------
        int
            The interval in seconds.
        """
        if interval[-1] == 'h':
            interval = int(interval[:-1]) * 60 * 60
        elif interval[-1] == 'd':
            interval = int(interval[:-1]) * 60 * 60 * 24
        elif interval[-1] == 'w':
            interval = int(interval[:-1]) * 60 * 60 * 24 * 7
        else:
            interval = int(interval[:-1]) * 60
        return interval

    @staticmethod
    def convert_interval(interval: str) -> str:
        """
        Convert an interval string to a different format.

        Parameters
        ----------
        interval : str
            A string representing a time interval (e.g., '1h', '1d', '1w').

        Returns
        -------
        str
            The converted interval in a different format (e.g., minutes, days, weeks, etc).
        """
        if interval[-1] == 'h':
            interval = str(int(interval[:-1]) * 60)
        elif interval[-1] == 'd':
            interval = 'D'
        elif interval[-1] == 'w':
            interval = 'W'
        else:
            interval = interval[:-1]
        return interval
    
    @staticmethod
    def convert_timstamp_to_time(timestamp: int, unit: str) -> datetime:
        """
        Convert a timestamp to a human-readable time with timezone adjustment.

        Parameters
        ----------
        timestamp : int
            A Unix timestamp.
        unit : str
            The unit of the timestamp (e.g., 's' for seconds).

        Returns
        -------
        pd.Timestamp
            Converted time with a 3-hour adjustment.
        """
        time = pd.to_datetime(timestamp, unit=unit)
        time += pd.to_timedelta(3, unit='h')
        return time

    def get_historical_funding_rate(self, symbol: str, limit: int, min_time: datetime) -> pd.DataFrame:
        """
        Retrieve historical funding rate information for a cryptocurrency pair.

        Parameters
        ----------
        symbol : str
            The symbol of the cryptocurrency pair.
        limit : int
            The number of data points to retrieve.
        min_time : datetime
            The minimum date for which data should be retrieved.

        Returns
        -------
        pd.DataFrame
            DataFrame containing historical funding rate information (time and funding_rate).
        """
        return pd.DataFrame(columns=['time', 'funding_rate'])


if __name__ == '__main__':
    symbol = 'TRXUPUSDT'
    print(re.match(r'.+UP-?(BUSD|USD[TC])', symbol))
