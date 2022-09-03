import pandas as pd
from abc import abstractmethod

from api.binance_api import Binance
from datetime import datetime


class DataFactory(object):
    @staticmethod
    def factory(exchange, **params):
        if exchange == 'Binance':
            return GetBinanceData(**params)
        elif exchange == 'OKEX':
            return GetOKEXData(**params)


class GetData:
    type = 'Data'
    name = 'Basic'

    def __init__(self, **params):
        self.params = params[self.type][self.name]['params']
        # basic interval (number of candles) to upload at startup
        self.limit = self.params.get('limit', 0)
        # minimum trading volume (USD) for exchange ticker to be added to watch list
        self.min_volume = self.params.get('min_volume', 0)
        # parameter to convert seconds to intervals
        self.timeframe_div = self.params.get('timeframe_div', dict())
        # dict to store timestamp for every timeframe
        self.ticker_dict = dict()

    @abstractmethod
    def get_data(self, df, ticker, timeframe):
        """ Get candle and volume data from exchange """
        pass

    @abstractmethod
    def get_tickers(self):
        """ Get list of available ticker names """
        pass

    def fill_ticker_dict(self, tickers):
        """ For every ticker set timestamp of the current time """
        dt = datetime.now()
        for ticker in tickers:
            self.ticker_dict[ticker] = dict()
            for tf in self.timeframe_div.keys():
                self.ticker_dict[ticker][tf] = dt

    @staticmethod
    def process_data(klines: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """ Update dataframe for current ticker or create new dataframe if it's first run """
        # convert numeric data to float type
        klines[['open', 'high', 'low', 'close', 'volume']] = klines[['open', 'high', 'low',
                                                                     'close', 'volume']].astype(float)
        # convert time to UTC+3
        klines['time'] = pd.to_datetime(klines['time'], unit='ms')
        klines['time'] = klines['time'] + pd.to_timedelta(3, unit='h')
        # If dataframe is empty - fill it with the new data
        if df.shape[0] == 0:
            df = klines
        else:
            # Update dataframe with new candles if it's not empty
            latest_time = df['time'].iloc[-1]
            klines = klines[klines['time'] > latest_time]
            df = pd.concat([df, klines])
            df = df.iloc[klines.shape[0]:].reset_index(drop=True)
        return df

    @staticmethod
    def add_indicator_data(dfs: dict, df: pd.DataFrame, indicators: list, ticker: str, timeframe: str,
                           configs: dict) -> (dict, pd.DataFrame):
        """ Add indicator data to cryptocurrency dataframe """
        levels = list()
        for indicator in indicators:
            # If indicator is support-resistance levels - get levels and add them to 'levels' category of dfs dict
            if indicator.name == 'SUP_RES':
                merge = timeframe == configs['Timeframes']['work_timeframe']
                higher_timeframe = configs['Timeframes']['higher_timeframe']
                higher_levels = dfs.get(ticker, dict()).get(higher_timeframe, dict()).get('levels', list())
                levels = indicator.get_indicator(df, ticker, timeframe, higher_levels, merge)
            else:
                df = indicator.get_indicator(df, ticker, timeframe)
        # Update dataframe dict
        if ticker not in dfs:
            dfs[ticker] = dict()
        if timeframe not in dfs[ticker]:
            dfs[ticker][timeframe] = dict()
        dfs[ticker][timeframe]['data'] = df
        dfs[ticker][timeframe]['levels'] = levels
        return dfs, df


class GetBinanceData(GetData):
    name = 'Binance'
    api = Binance()

    def __init__(self, **params):
        super(GetBinanceData, self).__init__(**params)
        # self.api.connect_to_api(self.key, self.secret)

    def get_limit(self, df: pd.DataFrame, ticker: str, timeframe: str) -> int:
        """ Get interval needed to download from exchange according to difference between current time and
            time of previous download"""
        if df.shape[0] == 0:
            return self.limit
        else:
            # get time passed from previous download and select appropriate interval
            time_diff_sec = (datetime.now() - self.ticker_dict[ticker][timeframe]).total_seconds()
            limit = int(time_diff_sec/self.timeframe_div[timeframe]) + 1
            # if time passed more than one interval - get it
            return min(self.limit, limit)

    def get_data(self, df: pd.DataFrame, ticker: str, timeframe: str) -> (pd.DataFrame, int):
        """ Get data from Binance exchange """
        limit = self.get_limit(df, ticker, timeframe)
        # get data from exchange only when there is at least one interval to get
        if limit > 1:
            klines = self.api.get_klines(ticker, timeframe, limit)
            df = self.process_data(klines, df)
            # update timestamp for current timeframe
            self.ticker_dict[ticker][timeframe] = datetime.now()
        return df, limit

    def get_tickers(self):
        """ Get list of available ticker names """
        tickers = self.api.get_ticker_names(self.min_volume)
        return tickers


class GetOKEXData(GetData):
    @abstractmethod
    def get_data(self, *args, **kwargs):
        pass
