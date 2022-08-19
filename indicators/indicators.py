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
    type = 'Indicator'
    name = 'Base'

    def __init__(self, params):
        self.params = params[self.type][self.name]['params']

    """ Get indicator data and write it to the dataframe """
    @abstractmethod
    def get_indicator(self, *args, **kwargs):
        pass


class RSI(Indicator):
    """ RSI indicator, default settings: timeperiod: 14"""
    name = 'RSI'

    def __init__(self, params):
        super(RSI, self).__init__(params)

    def get_indicator(self, df, ticker, timeframe):
        rsi = ta.RSI(df['close'], **self.params)
        df['rsi'] = rsi
        return df


class STOCH(Indicator):
    """ STOCH indicator, default settings: fastk_period: 14, slowk_period: 3, slowd_period:  3 """
    name = 'STOCH'

    def __init__(self, params):
        super(STOCH, self).__init__(params)

    def get_indicator(self, df, ticker, timeframe):
        slowk, slowd = ta.STOCH(df['high'], df['low'],
                                df['close'], **self.params)
        df['stoch_slowk'] = slowk
        df['stoch_slowd'] = slowd
        return df


class MACD(Indicator):
    """ MACD indicator, default settings: fastperiod: 12, slowperiod: 26, signalperiod: 9 """
    name = 'MACD'

    def __init__(self, params):
        super(MACD, self).__init__(params)

    def get_indicator(self, df, ticker, timeframe):
        macd, macdsignal, macdhist = ta.MACD(df[f'{ticker}_{timeframe}_close'], **self.params)
        df['macd'] = macd
        df['macdsignal'] = macdsignal
        df['macdhist'] = macdhist
        return df
