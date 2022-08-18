import talib as ta
from abc import abstractmethod


class IndicatorFactory(object):
    """ Return indicator according to 'indicator' variable value """
    @staticmethod
    def factory(indicator, params):
        if indicator == 'RSI':
            return RSI(params)
        elif indicator == 'STOCH':
            return STOCH(params)
        elif indicator == 'MACD':
            return MACD(params)


class Indicator:
    """ Abstract indicator class """
    def __init__(self):
        self.type = 'Indicator'

    """ Get indicator data and write it to the dataframe """
    @abstractmethod
    def get_indicator(self, *args, **kwargs):
        pass


class RSI(Indicator):
    """ RSI indicator, default settings: timeperiod: 14"""
    def __init__(self, params):
        super(RSI, self).__init__()
        self.kind = 'RSI'
        self.params = params[self.type][self.kind]['params']

    def get_indicator(self, cc_df, ticker, timeframe):
        rsi = ta.RSI(cc_df[f'{ticker}_{timeframe}_close'], **self.params)
        cc_df[f'{ticker}_{timeframe}_rsi'] = rsi
        return cc_df


class STOCH(Indicator):
    """ STOCH indicator, default settings: fastk_period: 14, slowk_period: 3, slowd_period:  3 """
    def __init__(self, params):
        super(STOCH, self).__init__()
        self.kind = 'STOCH'
        self.params = params[self.type][self.kind]['params']

    def get_indicator(self, cc_df, ticker, timeframe):
        slowk, slowd = ta.STOCH(cc_df[f'{ticker}_{timeframe}_high'], cc_df[f'{ticker}_{timeframe}_low'],
                                cc_df[f'{ticker}_{timeframe}_close'], **self.params)
        cc_df[f'{ticker}_{timeframe}_stoch_slowk'] = slowk
        cc_df[f'{ticker}_{timeframe}_stoch_slowd'] = slowd
        return cc_df


class MACD(Indicator):
    """ MACD indicator, default settings: fastperiod: 12, slowperiod: 26, signalperiod: 9 """
    def __init__(self, params):
        super(MACD, self).__init__()
        self.kind = 'MACD'
        self.params = params[self.type][self.kind]['params']

    def get_indicator(self, cc_df, ticker, timeframe):
        macd, macdsignal, macdhist = ta.MACD(cc_df[f'{ticker}_{timeframe}_close'], **self.params)
        cc_df[f'{ticker}_{timeframe}_macd'] = macd
        cc_df[f'{ticker}_{timeframe}_macdsignal'] = macdsignal
        cc_df[f'{ticker}_{timeframe}_macdhist'] = macdhist
        return cc_df
