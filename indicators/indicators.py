import talib as ta
import numpy as np
from abc import abstractmethod


class Indicators:
    def __init__(self):
        self.type = 'Indicator'

    @abstractmethod
    def get_indicator(self, *args, **kwargs):
        pass


class RSI(Indicators):
    def __init__(self, params):
        super().__init__()
        self.kind = 'RSI'
        # timeperiod: 14
        print(params[self.type][self.kind])
        self.params = params[self.type][self.kind]['params']

    def get_indicator(self, cc_df, ticker, timeframe):
        rsi = ta.RSI(cc_df[f'{ticker}_{timeframe}_close'], **self.params)
        return rsi


class STOCH(Indicators):
    def __init__(self, params):
        super().__init__()
        self.kind = 'STOCH'
        # default fastk_period: 14, slowk_period: 3, slowd_period:  3
        self.params = params[self.type][self.kind]['params']

    def get_indicator(self, cc_df, ticker, timeframe):
        slowk, slowd = ta.STOCH(cc_df[f'{ticker}_{timeframe}_high'], cc_df[f'{ticker}_{timeframe}_low'],
                               cc_df[f'{ticker}_{timeframe}_close'], **self.params)
        return slowk, slowd


class MACD(Indicators):
    def __init__(self, params):
        super().__init__()
        self.kind = 'RSI'
        # timeperiod: 14
        self.params = params[self.type][self.kind]['params']

    def get_indicator(self, cc_df, ticker, timeframe):
        macd, macdsignal, macdhist = ta.MACD(cc_df[f'{ticker}_{timeframe}_close'], **self.params)
        return macd, macdsignal, macdhist

