import numpy as np
import pandas as pd
from abc import abstractmethod

import models.cryptocurrency
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
        self.timestamp_dict = dict()
        dt = datetime.now()
        for tf in self.timeframe_div.keys():
            self.timestamp_dict[tf] = dt

    @abstractmethod
    def get_data(self, df, ticker, timeframe):
        """ Get candle and volume data from exchange """
        pass

    @abstractmethod
    def get_tickers(self):
        """ Get list of available ticker names """
        pass

    @staticmethod
    def process_data(crypto_currency: models.cryptocurrency.CryptoCurrency, df: pd.DataFrame) -> pd.DataFrame:
        """ Update dataframe for current ticker or create new dataframe if it's first run """
        # Create dataframe and fill it with data
        tmp = pd.DataFrame()
        tmp['time'] = np.asarray(crypto_currency.time)
        tmp['open'] = np.asarray(crypto_currency.open_values)
        tmp['high'] = np.asarray(crypto_currency.high_values)
        tmp['low'] = np.asarray(crypto_currency.low_values)
        tmp['close'] = np.asarray(crypto_currency.close_values)
        tmp['volume'] = np.asarray(crypto_currency.volume_values)
        tmp['time'] = pd.to_datetime(tmp['time'], unit='ms')
        # convert time to UTC+3
        tmp['time'] = tmp['time'] + pd.to_timedelta(3, unit='h')
        # If dataframe is empty - fill it with the new data
        if df.shape[0] == 0:
            df = tmp
        else:
            # Update dataframe with new candles if it's not empty
            latest_time = df['time'].iloc[-1]
            tmp = tmp[tmp['time'] > latest_time]
            df = pd.concat([df, tmp])
            df = df.iloc[tmp.shape[0]:].reset_index(drop=True)
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
    key = "7arxKITvadhYavxsQr5dZelYK4kzyBGM4rsjDCyJiPzItNlAEdlqOzibV7yVdnNy"
    secret = "3NvopCGubDjCkF4SzqP9vj9kU2UIhE4Qag9ICUdESOBqY16JGAmfoaUIKJLGDTr4"

    def __init__(self, **params):
        super(GetBinanceData, self).__init__(**params)
        self.api.connect_to_api(self.key, self.secret)

    def get_limit(self, df: pd.DataFrame, timeframe: str) -> int:
        """ Get interval needed to download from exchange according to difference between current time and
            time of previous download"""
        if df.shape[0] == 0:
            return self.limit
        else:
            # get time passed from previous download and select appropriate interval
            time_diff_sec = (datetime.now() - self.timestamp_dict[timeframe]).total_seconds()
            limit = int(time_diff_sec/self.timeframe_div[timeframe]) + 1
            # if time passed more than one interval - get it
            return min(self.limit, limit)

    def get_data(self, df: pd.DataFrame, ticker: str, timeframe: str):
        """ Get data from Binance exchange """
        limit = self.get_limit(df, timeframe)
        # get data from exchange only when there is at least one interval to get
        if limit > 1:
            crypto_currency = self.api.get_crypto_currency(ticker, timeframe, limit)
            df = self.process_data(crypto_currency, df)
            # update timestamp for current timeframe
            self.timestamp_dict[timeframe] = datetime.now()
            return df, limit, True
        return df, limit, False

    def get_tickers(self):
        """ Get list of available ticker names """
        ticker_names = self.api.get_ticker_names()
        df = self.api.get_ticker_volume(ticker_names)
        tickers = df.loc[df['volume'] >= self.min_volume, 'ticker'].to_list()
        return tickers


class GetOKEXData(GetData):
    @abstractmethod
    def get_data(self, *args, **kwargs):
        pass
