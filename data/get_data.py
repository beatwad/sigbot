import numpy as np
import pandas as pd
from abc import abstractmethod
from api.binance_api import Binance


class DataFactory(object):
    @staticmethod
    def factory(exchange):
        if exchange == 'Binance':
            return GetBinanceData()
        elif exchange == 'OKEX':
            return GetOKEXData()


class GetData:
    type = 'Data'

    @abstractmethod
    def get_data(self, df, ticker, timeframe, interval):
        """ Get candle and volume data from exchange """
        pass

    @staticmethod
    def process_data(crypto_currency, df, ticker, timeframe):
        """ Update dataframe for current ticker or create new dataframe if it's first run """
        # Create dataframe and fill it with data
        tmp = pd.DataFrame()
        tmp[f'{ticker}_{timeframe}_time'] = np.asarray(crypto_currency.time)
        tmp[f'{ticker}_{timeframe}_open'] = np.asarray(crypto_currency.open_values)
        tmp[f'{ticker}_{timeframe}_close'] = np.asarray(crypto_currency.close_values)
        tmp[f'{ticker}_{timeframe}_high'] = np.asarray(crypto_currency.high_values)
        tmp[f'{ticker}_{timeframe}_low'] = np.asarray(crypto_currency.low_values)
        tmp[f'{ticker}_{timeframe}_volume'] = np.asarray(crypto_currency.volume_values)
        tmp[f'{ticker}_{timeframe}_time'] = pd.to_datetime(tmp[f'{ticker}_{timeframe}_time'], unit='ms')
        # If dataframe is empty - fill it with new data
        if df.shape[0] == 0:
            df = tmp
        else:
            # Update dataframe with new candles if it's not empty
            latest_time = df['BTCUSDT_5m_time'].iloc[-1]
            tmp = tmp[tmp['BTCUSDT_5m_time'] > latest_time]
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
    api = Binance()
    key = "7arxKITvadhYavxsQr5dZelYK4kzyBGM4rsjDCyJiPzItNlAEdlqOzibV7yVdnNy"
    secret = "3NvopCGubDjCkF4SzqP9vj9kU2UIhE4Qag9ICUdESOBqY16JGAmfoaUIKJLGDTr4"

    def get_data(self, df, ticker, timeframe, interval):
        self.api.connect_to_api(self.key, self.secret)
        crypto_currency = self.api.get_crypto_currency(ticker, timeframe, interval)
        df = self.process_data(crypto_currency, df, ticker, timeframe)
        return df


class GetOKEXData(GetData):
    @abstractmethod
    def get_data(self, *args, **kwargs):
        pass
