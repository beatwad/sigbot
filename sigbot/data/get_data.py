import requests
import pandas as pd
from api.binance_api import Binance
from api.binance_futures_api import BinanceFutures
from api.okex_api import OKEX
from api.okex_swap_api import OKEXSwap
from api.bybit_api import ByBit
from api.bybit_perpetual_api import ByBitPerpetual
from datetime import datetime
from json.decoder import JSONDecodeError
from log.log import logger


class DataFactory(object):
    @staticmethod
    def factory(exchange, **configs):
        if exchange == 'Binance':
            return GetBinanceData(**configs)
        elif exchange == 'BinanceFutures':
            return GetBinanceFuturesData(**configs)
        elif exchange == 'OKEX':
            return GetOKEXData(**configs)
        elif exchange == 'OKEXSwap':
            return GetOKEXSwapData(**configs)
        elif exchange == 'ByBit':
            return GetByBitData(**configs)
        elif exchange == 'ByBitPerpetual':
            return GetByBitPerpetualData(**configs)


class GetData:
    type = 'Data'
    name = 'Basic'

    def __init__(self, **configs):
        self.configs = configs[self.type][self.name]['params']
        # basic interval (number of candles) to upload at startup
        self.limit = self.configs.get('limit', 0)
        # minimum trading volume (USD) for exchange ticker to be added to watch list
        self.min_volume = self.configs.get('min_volume', 0)
        # parameter to convert seconds to intervals
        self.timeframe_div = self.configs.get('timeframe_div', dict())
        # dict to store timestamp for every timeframe
        self.ticker_dict = dict()
        self.api = None

    def get_data(self, df: pd.DataFrame, ticker: str, timeframe: str, dt_now: datetime) -> (pd.DataFrame, int):
        """ Get data from exchange """
        limit = self.get_limit(df, ticker, timeframe, dt_now)
        # get data from exchange only when there is at least one interval to get
        if limit > 1:
            try:
                klines = self.api.get_klines(ticker, timeframe, limit + 2)
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, JSONDecodeError):
                logger.exception(f'Catch an exception while trying to get data. API is {self.api}')
                return df, 0
            df = self.process_data(klines, df)
        return df, limit

    def get_tickers(self) -> list:
        """ Get list of available ticker names """
        tickers = self.api.get_ticker_names(self.min_volume)
        return tickers

    def fill_ticker_dict(self, tickers: str) -> None:
        """ For every ticker set timestamp of the current time """
        # dt = datetime.now()
        for ticker in tickers:
            self.ticker_dict[ticker] = dict()
            for tf in self.timeframe_div.keys():
                self.ticker_dict[ticker][tf] = -1

    @staticmethod
    def add_utc_3(df):
        df['time'] = df['time'] + pd.to_timedelta(3, unit='h')
        return df

    def process_data(self, klines: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """ Update dataframe for current ticker or create new dataframe if it's first run """
        # convert numeric data to float type
        klines[['open', 'high', 'low', 'close', 'volume']] = klines[['open', 'high', 'low',
                                                                     'close', 'volume']].astype(float).copy()
        # convert time to UTC+3
        if self.name == 'ByBitPerpetual':
            klines['time'] = pd.to_datetime(klines['time'], unit='s')
        else:
            klines['time'] = pd.to_datetime(klines['time'], unit='ms')
        klines = self.add_utc_3(klines)
        # If dataframe is empty - fill it with the new data
        if df.shape[0] == 0:
            df = klines
        else:
            # Update dataframe with new candles if it's not empty
            earliest_time = klines['time'].min()
            df = df[df['time'] < earliest_time]
            df = pd.concat([df, klines])
            # if size of dataframe more than limit - short it
            df = df.iloc[max(df.shape[0]-self.limit, 0):].reset_index(drop=True)
        return df

    @staticmethod
    def add_indicator_data(dfs: dict, df: pd.DataFrame, ttype: str, indicators: list, ticker: str, timeframe: str,
                           data_qty: int) -> dict:
        """ Add indicator data to cryptocurrency dataframe """
        levels = list()
        indicators = [i for i in indicators if i.ttype == ttype]

        for indicator in indicators:
            df = indicator.get_indicator(df, ticker, timeframe, data_qty)
        # Update dataframe dict
        if ticker not in dfs:
            dfs[ticker] = dict()
        if timeframe not in dfs[ticker]:
            dfs[ticker][timeframe] = dict()
        if 'data' not in dfs[ticker][timeframe]:
            dfs[ticker][timeframe]['data'] = dict()
        dfs[ticker][timeframe]['data'][ttype] = df.copy()
        dfs[ticker][timeframe]['levels'] = levels.copy()
        return dfs

    @staticmethod
    def get_time_label(dt_now: datetime, timeframe: str) -> int:
        """ Define time label according to the timeframe """
        if timeframe == '5m':
            return int(dt_now.minute / 5)
        elif timeframe == '15m':
            return int(dt_now.minute / 15)
        elif timeframe == '30m':
            return int(dt_now.minute / 30)
        elif timeframe == '1h':
            return dt_now.hour
        elif timeframe == '4h':
            return int(dt_now.hour / 4)
        elif timeframe == '12h':
            return int(dt_now.hour / 12)
        else:
            return dt_now.day

    def get_limit(self, df: pd.DataFrame, ticker: str, timeframe: str, dt_now: datetime) -> int:
        """ Get interval needed to download from exchange according to time label """
        dt_measure = self.get_time_label(dt_now, timeframe)
        if df.shape[0] == 0:
            self.ticker_dict[ticker][timeframe] = dt_measure
            return self.limit
        else:
            # if enough time has passed and time label has changed - increase the limit to update candle data
            if dt_measure != self.ticker_dict[ticker][timeframe]:
                limit = 2
                self.ticker_dict[ticker][timeframe] = dt_measure
            else:
                limit = 1
            return min(self.limit, limit)


class GetBinanceData(GetData):
    name = 'Binance'

    def __init__(self, **configs):
        super(GetBinanceData, self).__init__(**configs)
        self.key = "7arxKITvadhYavxsQr5dZelYK4kzyBGM4rsjDCyJiPzItNlAEdlqOzibV7yVdnNy"
        self.secret = "3NvopCGubDjCkF4SzqP9vj9kU2UIhE4Qag9ICUdESOBqY16JGAmfoaUIKJLGDTr4"
        self.api = Binance(self.key, self.secret)


class GetBinanceFuturesData(GetData):
    name = 'BinanceFutures'

    def __init__(self, **configs):
        super(GetBinanceFuturesData, self).__init__(**configs)
        self.key = "QD5nRIFvOXYBdVsnkfWf5G8D91CKTVgZXqReyO6PqL70r9PjP8SbbVh3bYlJc9cy"
        self.secret = "ht5hw25DzKOvfaU2rTqpSsy0CDTsKfYsb2JSQLSCbrz7zoLrnnKWi9SBh7NYFSZD"
        self.api = BinanceFutures(self.key, self.secret)


class GetOKEXData(GetData):
    name = 'OKEX'

    def __init__(self, **configs):
        super(GetOKEXData, self).__init__(**configs)
        self.api = OKEX()


class GetOKEXSwapData(GetData):
    name = 'OKEX'

    def __init__(self, **configs):
        super(GetOKEXSwapData, self).__init__(**configs)
        self.api = OKEXSwap()


class GetByBitData(GetData):
    name = 'ByBit'

    def __init__(self, **configs):
        super(GetByBitData, self).__init__(**configs)
        self.api = ByBit()


class GetByBitPerpetualData(GetData):
    name = 'ByBitPerpetual'

    def __init__(self, **configs):
        super(GetByBitPerpetualData, self).__init__(**configs)
        self.api = ByBitPerpetual()
