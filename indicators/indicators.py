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
        self.params = params[self.type][self.kind]

    def get_indicator(self, close):
        rsi = ta.RSI(close, **self.params)
        return rsi


class STOCH(Indicators):
    def __init__(self, params):
        super().__init__()
        self.kind = 'STOCH'
        # fastk_period: 14, slowk_period: 3, slowd_period:  3
        self.params = params[self.type][self.kind]

    def get_indicator(self, high, low, close):
        lowk, slowd = ta.STOCH(high, low, close, **self.params)
        return lowk, slowd


class MACD(Indicators):
    def __init__(self, params):
        super().__init__()
        self.kind = 'RSI'
        # timeperiod: 14
        self.params = params[self.type][self.kind]

    def get_indicator(self, close):
        macd, macdsignal, macdhist = ta.MACD(close, **self.params)
        return macd, macdsignal, macdhist

