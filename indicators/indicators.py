import numpy as np
import pandas as pd
import talib as ta
from abc import abstractmethod
from collections import Counter


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
        elif indicator.startswith('PriceChange'):
            return PriceChange(params)


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
    def is_support(df: pd.DataFrame, i: int) -> bool:
        """ Find support levels """
        support = df['low'][i] < df['low'][i-1] < df['low'][i-2] and \
                  df['low'][i] < df['low'][i+1] < df['low'][i+2]
        support_roll = df['low_roll'][i] < df['low_roll'][i-1] < df['low_roll'][i-2] < df['low_roll'][i-3] and \
                       df['low_roll'][i] < df['low_roll'][i+1] < df['low_roll'][i+2] < df['low_roll'][i+3]
        return support or support_roll

    @staticmethod
    def is_resistance(df: pd.DataFrame, i: int) -> bool:
        """ Find resistance levels """
        resistance = df['high'][i] > df['high'][i-1] > df['high'][i-2] and \
                     df['high'][i] > df['high'][i+1] > df['high'][i+2]
        resistance_roll = df['high_roll'][i] > df['high_roll'][i-1] > df['high_roll'][i-2] > df['high_roll'][i-3] and \
                          df['high_roll'][i] > df['high_roll'][i+1] > df['high_roll'][i+2] > df['high_roll'][i+3]
        return resistance or resistance_roll

    def find_levels(self, df: pd.DataFrame, level_proximity: float) -> list:
        """ Find levels and save their value and their importance """
        levels = list()
        df['high_roll'] = df['high'].rolling(3).mean()
        df['low_roll'] = df['low'].rolling(3).mean()

        # find levels and increase importance of those where price changed direction twice or more
        for index, row in df[::-1].iterrows():
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
        return sorted(levels, key=lambda x: x[0])

    @staticmethod
    def add_higher_levels(levels: list, higher_levels: list, s: list) -> list:
        """ Merge levels with the levels from higher timeframe. If layers from lower and higher timeframe are
            coincided - increase the importance value of lower timeframe """
        levels_to_add = list()
        if not levels:
            return higher_levels
        for h_level in higher_levels:
            distinct_level = True
            # if level from higher timeframe too small or too big for our current level range - continue
            if levels[0][0] - h_level[0] > s or h_level[0] - levels[-1][0] > s:
                continue
            # if level from higher timeframe too close to one of current levels - increase it importance
            # else just add it to level list
            for i in range(len(levels)):
                # if one of current level is already bigger than level that we want to add -
                # there is no sense to continue
                if levels[i][0] - h_level[0] > s:
                    break
                if abs(h_level[0] - levels[i][0]) < s:
                    levels[i][1] = 3
                    distinct_level = False
            if distinct_level:
                levels_to_add.append(h_level)
        return levels + levels_to_add


class PriceChange(Indicator):
    """ Find big changes of price in both directions """
    name = 'PriceChange'

    def __init__(self, params):
        super(PriceChange, self).__init__(params)
        self.low_price_quantile = self.params.get('low_price_quantile', 5)
        self.high_price_quantile = self.params.get('high_price_quantile', 95)
        self.round_decimals = self.params.get('decimals', 4)
        self.close_prices_lag_1 = Counter()
        self.close_prices_lag_2 = Counter()
        self.close_prices_lag_3 = Counter()

    def get_price_change(self, df: pd.DataFrame, lag: int) -> list:
        close_prices = (df['close'] - df['close'].shift(1)) / df['close'].shift(lag) * 100
        df['close_price_change'] = np.round(close_prices.values, 4)
        return df['close_price_change'].values

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str) -> pd.DataFrame:
        # get frequency counter
        close_prices_lag_1 = self.get_price_change(df, lag=1)
        close_prices_lag_2 = self.get_price_change(df, lag=2)
        close_prices_lag_3 = self.get_price_change(df, lag=3)
        self.close_prices_lag_1 += Counter(close_prices_lag_1)
        self.close_prices_lag_2 += Counter(close_prices_lag_2)
        self.close_prices_lag_3 += Counter(close_prices_lag_3)
        # get lag 1 quantiles and save to dataframe
        q_lag_1 = pd.Series(data=self.close_prices_lag_1.keys(), index=self.close_prices_lag_1.values())
        q_low_lag_1 = q_lag_1.sort_index().quantile(self.low_price_quantile / 100)
        q_high_lag_1 = q_lag_1.sort_index().quantile(self.high_price_quantile / 100)
        df[['q_low_lag_1', 'q_high_lag_1']] = q_low_lag_1, q_high_lag_1
        # get lag 2 quantiles and save to dataframe
        q_lag_2 = pd.Series(data=self.close_prices_lag_2.keys(), index=self.close_prices_lag_2.values())
        q_low_lag_2 = q_lag_2.sort_index().quantile(self.low_price_quantile / 100)
        q_high_lag_2 = q_lag_2.sort_index().quantile(self.high_price_quantile / 100)
        df[['q_low_lag_2', 'q_high_lag_2']] = q_low_lag_2, q_high_lag_2
        # get lag 3 quantiles and save to dataframe
        q_lag_3 = pd.Series(data=self.close_prices_lag_3.keys(), index=self.close_prices_lag_3.values())
        q_low_lag_3 = q_lag_3.sort_index().quantile(self.low_price_quantile / 100)
        q_high_lag_3 = q_lag_3.sort_index().quantile(self.high_price_quantile / 100)
        df[['q_low_lag_3', 'q_high_lag_3']] = q_low_lag_3, q_high_lag_3
        return df
