import numpy as np
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
        self.interval = self.params['interval']
        # parameter to convert seconds to intervals
        self.timeframe_div = self.params['timeframe_div']
        # dict to store timestamp for every timeframe
        self.timestamp_dict = dict()
        dt = datetime.now()
        for tf in self.timeframe_div.keys():
            self.timestamp_dict[tf] = dt

    @abstractmethod
    def get_data(self, df, ticker, timeframe):
        """ Get candle and volume data from exchange """
        pass

    @staticmethod
    def process_data(crypto_currency, df):
        """ Update dataframe for current ticker or create new dataframe if it's first run """
        # Create dataframe and fill it with data
        tmp = pd.DataFrame()
        tmp['time'] = np.asarray(crypto_currency.time)
        tmp['open'] = np.asarray(crypto_currency.open_values)
        tmp['close'] = np.asarray(crypto_currency.close_values)
        tmp['high'] = np.asarray(crypto_currency.high_values)
        tmp['low'] = np.asarray(crypto_currency.low_values)
        tmp['volume'] = np.asarray(crypto_currency.volume_values)
        tmp['time'] = pd.to_datetime(tmp['time'], unit='ms')
        tmp['time'] = tmp['time'] + pd.to_timedelta(3, unit='h')
        # If dataframe is empty - fill it with new data
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
    def add_indicator_data(df, indicators, ticker, timeframe):
        """ Add indicator data to cryptocurrency dataframe """
        for indicator in indicators:
            df = indicator.get_indicator(df, ticker, timeframe)
        return df


class GetBinanceData(GetData):
    name = 'Binance'
    api = Binance()
    key = "7arxKITvadhYavxsQr5dZelYK4kzyBGM4rsjDCyJiPzItNlAEdlqOzibV7yVdnNy"
    secret = "3NvopCGubDjCkF4SzqP9vj9kU2UIhE4Qag9ICUdESOBqY16JGAmfoaUIKJLGDTr4"

    def __int__(self, **params):
        super(GetBinanceData, self).__init__(**params)

    def get_interval(self, df: pd.DataFrame, timeframe: str) -> int:
        """ Get interval needed to download from exchange according to difference between current time and
            time of previous download"""
        if df.shape == (0, 0):
            return self.interval
        else:
            # get time passed from previous download and select appropriate interval
            time_diff_sec = (datetime.now() - self.timestamp_dict[timeframe]).seconds
            interval = int(time_diff_sec/self.timeframe_div[timeframe]) + 1
            # if time passed more than one interval - get it
            if interval > 1:
                return min(self.interval, interval)
            return 0

    def get_data(self, df: pd.DataFrame, ticker: str, timeframe: str):
        """ Get data from Binance exchange """
        self.api.connect_to_api(self.key, self.secret)
        interval = self.get_interval(df, timeframe)
        # get data from exchange only when there is at least one interval to get
        if interval > 0:
            print(interval)
            crypto_currency = self.api.get_crypto_currency(ticker, timeframe, interval)
            df = self.process_data(crypto_currency, df)
            # update timestamp for current timeframe
            self.timestamp_dict[timeframe] = datetime.now()
        return df


class GetOKEXData(GetData):
    @abstractmethod
    def get_data(self, *args, **kwargs):
        pass
