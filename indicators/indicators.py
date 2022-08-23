import numpy as np
import pandas as pd
import talib as ta
from abc import abstractmethod


class IndicatorFactory(object):
    """ Return indicator according to 'indicator' variable value """
    @staticmethod
    def factory(indicator, params):
        if indicator.startswith('RSI'):
            return RSI(params)
        elif indicator.startswith('STOCH'):
            return STOCH(params)
        elif indicator.startswith('MACD'):
            return MACD(params)
        elif indicator.startswith('SUP_RES'):
            return SupRes(params)


class Indicator:
    """ Abstract indicator class """
    type = 'Indicator'
    name = 'Base'

    def __init__(self, params):
        self.params = params[self.type][self.name]['params']

    @abstractmethod
    def get_indicator(self, *args, **kwargs):
        """ Get indicator data and write it to the dataframe """
        pass


class RSI(Indicator):
    """ RSI indicator, default settings: timeperiod: 14"""
    name = 'RSI'

    def __init__(self, params: dict):
        super(RSI, self).__init__(params)

    def get_indicator(self, df, ticker: str, timeframe: str) -> pd.DataFrame:
        rsi = ta.RSI(df['close'], **self.params)
        df['rsi'] = rsi
        return df


class STOCH(Indicator):
    """ STOCH indicator, default settings: fastk_period: 14, slowk_period: 3, slowd_period:  3 """
    name = 'STOCH'

    def __init__(self, params):
        super(STOCH, self).__init__(params)

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str) -> pd.DataFrame:
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

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str) -> pd.DataFrame:
        macd, macdsignal, macdhist = ta.MACD(df['close'], **self.params)
        df['macd'] = macd
        df['macdsignal'] = macdsignal
        df['macdhist'] = macdhist
        return df


class DACD(Indicator):
    """ MACD indicator, default settings: fastperiod: 12, slowperiod: 26, signalperiod: 9 """
    name = 'MACD'

    def __init__(self, params):
        super(DACD, self).__init__(params)

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str) -> pd.DataFrame:
        macd, macdsignal, macdhist = ta.MACD(df['close'], **self.params)
        df['macd'] = macd
        df['macdsignal'] = macdsignal
        df['macdhist'] = macdhist
        return df


class SupRes(Indicator):
    """ Find support and resistance levels on the candle plot """
    name = 'SUP_RES'

    def __init__(self, params):
        super(SupRes, self).__init__(params)
        self.alpha = self.params.get('alpha', 0.7)
        self.merge_level_multiplier = self.params.get('merge_level_multiplier', 1)

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str, higher_levels, merge=False) -> list:
        # level proximity measure * multiplier (from configs)
        level_proximity = np.mean(df['high'] - df['low']) * self.merge_level_multiplier
        levels = self.find_levels(df, level_proximity)
        if merge:
            levels = self.add_higher_levels(levels, higher_levels, level_proximity)
        return levels

    @staticmethod
    def is_support(df: pd.DataFrame, i):
        """ Find support levels """
        support = df['low_roll'][i] < df['low_roll'][i - 1] < df['low_roll'][i - 2] < df['low_roll'][i - 3] and \
                  df['low_roll'][i] < df['low_roll'][i + 1] < df['low_roll'][i + 2] < df['low_roll'][i + 3]
        return support

    @staticmethod
    def is_resistance(df, i):
        """ Find resistance levels """
        resistance = df['high_roll'][i] > df['high_roll'][i - 1] > df['high_roll'][i - 2] > df['high_roll'][i - 3] and \
                     df['high_roll'][i] > df['high_roll'][i + 1] > df['high_roll'][i + 2] > df['high_roll'][i + 3]

        return resistance

    def find_levels(self, df, level_proximity):
        """ Find levels and save their value and their importance """
        levels = list()
        df['high_roll'] = df['high'].rolling(3).mean()
        df['low_roll'] = df['low'].rolling(3).mean()

        # find levels and increase importance of those where price changed direction twice or more
        for index, row in df.iterrows():
            if 2 <= index <= df.shape[0] - 4:
                distinct_level = True
                sup, res = self.is_support(df, index), self.is_resistance(df, index)
                if sup or res:
                    if sup:
                        level = row['low']
                    else:
                        level = row['high']
                    for i in range(len(levels)):
                        if abs(level - levels[i][0]) < level_proximity:
                            levels[i][1] = 2
                            distinct_level = False
                    if distinct_level:
                        levels.append([level, 1])

        return levels

    @staticmethod
    def add_higher_levels(levels, ticker_levels, s):
        """ Merge levels with the levels from higher timeframe. If layers from lower and higher timeframe are
            coincided - increase the importance value of lower timeframe """
        for t_level in ticker_levels:
            distinct_level = True
            for i in range(len(levels)):
                if abs(t_level[0] - levels[i][0]) < s:
                    levels[i][2] = 3
                    distinct_level = False
            if distinct_level:
                levels.append([t_level, 1])
        return levels
