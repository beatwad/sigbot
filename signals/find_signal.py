import numpy as np
import pandas as pd
from abc import abstractmethod


class SignalFactory(object):
    """ Return indicator according to 'indicator' variable value """
    @staticmethod
    def factory(indicator, configs):
        if indicator == 'RSI':
            return RSISignal(**configs)
        elif indicator == 'STOCH':
            return STOCHSignal(**configs)
        elif indicator == 'MACD':
            return MACDSignal(**configs)
        elif indicator == 'SUP_RES':
            return SupResSignal(**configs)
        elif indicator == 'SUP_RES_Robust':
            return SupResSignalRobust(**configs)
        elif indicator == 'PriceChange':
            return PriceChangeSignal(**configs)
        elif indicator == 'LinearReg':
            return LinearRegSignal(**configs)


class SignalBase:
    """ Base signal searching class """
    type = 'Indicator_signal'
    name = 'Base'

    def __init__(self, configs):
        self.configs = configs[self.type][self.name]['params']

    @abstractmethod
    def find_signal(self, *args, **kwargs):
        return False, '', []

    @staticmethod
    def lower_bound(indicator: pd.Series, i: int, low_bound: float) -> bool:
        """ Returns True if at least two of three last points of indicator are lower than low_bound param """
        try:
            res = [indicator.loc[i] < low_bound, indicator.loc[i-1] < low_bound, indicator.loc[i-2] < low_bound]
        except KeyError:
            return False
        if sum(res) >= 2:
            return True
        return False

    @staticmethod
    def higher_bound(indicator: pd.Series, i: int, high_bound: float) -> bool:
        """ Returns True if at least two of three last points of indicator are higher than high_bound param """
        try:
            res = [indicator.loc[i] > high_bound, indicator.loc[i-1] > high_bound, indicator.loc[i-2] > high_bound]
        except KeyError:
            return False
        if sum(res) >= 2:
            return True
        return False

    @staticmethod
    def lower_bound_robust(indicator: pd.Series, i: int, low_bound: float) -> bool:
        """ Returns True if indicator is lower than low bound """
        try:
            if indicator.loc[i] < low_bound:
                return True
        except KeyError:
            return False
        return False

    @staticmethod
    def higher_bound_robust(indicator: pd.Series, i: int, high_bound: float) -> bool:
        """ Returns True if indicator is higher than high bound """
        try:
            if indicator.loc[i] > high_bound:
                return True
        except KeyError:
            return False
        return False

    @staticmethod
    def up_direction(indicator: pd.Series, i: int) -> bool:
        """ Returns True if indicator values are moving up """
        if indicator.loc[i] > 0:
            return True
        return False

    @staticmethod
    def down_direction(indicator: pd.Series, i: int) -> bool:
        """ Returns True if indicator values are moving down """
        if indicator.loc[i] < 0:
            return True
        return False


class STOCHSignal(SignalBase):
    """ Check if STOCH is in overbuy/oversell zone and is going to change its direction to opposite """
    type = 'Indicator_signal'
    name = 'STOCH'

    def __init__(self, **configs):
        super(STOCHSignal, self).__init__(configs)
        self.low_bound = self.configs.get('low_bound', 20)
        self.high_bound = self.configs.get('high_bound', 80)

    @staticmethod
    def crossed_lines(indicator: pd.Series, i: int, up: bool) -> bool:
        """ Returns True if slowk and slowd lines of RSI has crossed """
        if up:
            if (indicator.loc[i] < 0 and indicator.loc[i-1] > 0) or \
                    (indicator.loc[i] < 0 and indicator.loc[i-2] > 0):
                return True
        else:
            if (indicator.loc[i] > 0 and indicator.loc[i-1] < 0) or \
                    (indicator.loc[i] > 0 and indicator.loc[i-2] < 0):
                return True

        return False

    def find_signal(self, df: pd.DataFrame, index: int, *args) -> (bool, str, tuple):
        """ Return signal if RSI is higher/lower than high/low bound (overbuy/oversell zone),
            slowk and slowd lines have crossed and their direction is down/up """
        # Find STOCH signal
        if self.lower_bound(df['stoch_slowk'], index, self.low_bound) and \
                self.lower_bound(df['stoch_slowd'], index, self.low_bound):
            if self.crossed_lines(df['stoch_diff'], index, up=False):
                if self.up_direction(df['stoch_slowk_dir'], index) and \
                        self.up_direction(df['stoch_slowd_dir'], index):
                    return True, 'buy', (self.low_bound, self.high_bound)
        if self.higher_bound(df['stoch_slowk'], index, self.high_bound) and \
                self.higher_bound(df['stoch_slowd'], index, self.high_bound):
            if self.crossed_lines(df['stoch_diff'], index, up=True):
                if self.down_direction(df['stoch_slowk_dir'], index) and \
                        self.down_direction(df['stoch_slowd_dir'], index):
                    return True, 'sell', (self.low_bound, self.high_bound)
        return False, '', ()


class RSISignal(SignalBase):
    """ Check if RSI is in overbuy/oversell zone """
    type = 'Indicator_signal'
    name = "RSI"

    def __init__(self, **configs):
        super(RSISignal, self).__init__(configs)
        self.low_bound = self.configs.get('low_bound', 25)
        self.high_bound = self.configs.get('high_bound', 75)

    def find_signal(self, df: pd.DataFrame, index: int, *args) -> (bool, str, tuple):
        """ Return signal if RSI is higher/lower than high/low bound (overbuy/oversell zone),
            slowk and slowd lines have crossed and their direction is down/up """
        # Find RSI signal
        if self.lower_bound(df['rsi'], index, self.low_bound):
            return True, 'buy', (self.low_bound, self.high_bound)
        elif self.higher_bound(df['rsi'], index, self.high_bound):
            return True, 'sell', (self.low_bound, self.high_bound)
        return False, '', ()


class LinearRegSignal(SignalBase):
    """ Check if RSI is in overbuy/oversell zone """
    type = 'Indicator_signal'
    name = "LinearReg"

    def __init__(self, **configs):
        super(LinearRegSignal, self).__init__(configs)
        self.low_bound = self.configs.get('low_bound', -0.1)
        self.high_bound = self.configs.get('high_bound', 0.1)

    def find_signal(self, df: pd.DataFrame, index: int, *args) -> (bool, str, tuple):
        """ Return signal if RSI is higher/lower than high/low bound (overbuy/oversell zone),
            slowk and slowd lines have crossed and their direction is down/up """
        # Find LinearReg signal
        if self.lower_bound_robust(df['linear_reg_angle'], index, self.low_bound):
            return True, 'sell', ()
        if self.higher_bound_robust(df['linear_reg_angle'], index, self.high_bound):
            return True, 'buy', ()
        return False, '', ()


class PriceChangeSignal(SignalBase):
    type = 'Indicator_signal'
    name = 'PriceChange'

    def __init__(self, **configs):
        super(PriceChangeSignal, self).__init__(configs)

    def find_signal(self, df: pd.DataFrame, index: int, *args) -> (bool, str, tuple):
        """ Return signal if RSI is higher/lower than high/low bound (overbuy/oversell zone),
            slowk and slowd lines have crossed and their direction is down/up """
        # Find price change signal
        if self.lower_bound_robust(df['close_price_change_lag_1'], index, df['q_low_lag_1'].loc[index]):
            return True, 'buy', 1
        elif self.lower_bound_robust(df['close_price_change_lag_2'], index, df['q_low_lag_2'].loc[index]):
            return True, 'buy', 2
        elif self.lower_bound_robust(df['close_price_change_lag_3'], index, df['q_low_lag_3'].loc[index]):
            return True, 'buy', 3
        elif self.higher_bound_robust(df['close_price_change_lag_1'], index, df['q_high_lag_1'].loc[index]):
            return True, 'sell', 1
        elif self.higher_bound_robust(df['close_price_change_lag_2'], index, df['q_high_lag_2'].loc[index]):
            return True, 'sell', 2
        elif self.higher_bound_robust(df['close_price_change_lag_3'], index, df['q_high_lag_3'].loc[index]):
            return True, 'sell', 3
        return False, '', ()


class MACDSignal(SignalBase):
    type = 'Indicator_signal'
    name = 'MACD'

    def __init__(self, **configs):
        super(MACDSignal, self).__init__(configs)
        self.low_bound = self.configs.get('low_bound', 20)
        self.high_bound = self.configs.get('high_bound', 80)

    def find_signal(self, *args, **kwargs):
        return False, '', ()


class SupResSignal(SignalBase):
    """ Find support and resistance levels on the candle plot """
    type = 'Indicator_signal'
    name = 'SUP_RES'

    def __init__(self, **configs):
        super(SupResSignal, self).__init__(configs)
        self.proximity_multiplier = self.configs.get('proximity_multiplier', 1)

    def find_signal(self, *args, **kwargs):
        return True, '', ()

    @staticmethod
    def check_levels(df, index, levels, level_proximity, buy):
        """ Check if buy points are near support levels and sell points are near resistance levels"""
        for level in levels:
            if buy and abs(df.loc[index, 'low'] - level[0]) < level_proximity:
                return True
            if not buy and abs(df.loc[index, 'high'] - level[0]) < level_proximity:
                return True
        return False


class SupResSignalRobust(SupResSignal):
    """ Find support and resistance levels on the candle plot but return True only if support level is
        lower than buy point or resistance level is higher than sell point """
    name = 'SUP_RES_Robust'

    @staticmethod
    def check_levels(df, index, levels, level_proximity, buy):
        """ Check if buy points are near support levels and sell points are near resistance levels"""
        for level in levels:
            if buy and abs(df.loc[index, 'low'] - level[0]) < level_proximity and df.loc[index, 'low'] >= level[0]:
                return True
            if not buy and abs(df.loc[index, 'high'] -
                               level[0]) < level_proximity and df.loc[index, 'high'] <= level[0]:
                return True
        return False


class FindSignal:
    """ Class for searching of the indicator combination """
    def __init__(self, configs):
        self.configs = configs
        self.indicator_list = configs['Indicator_list']
        self.indicator_signals = self.prepare_indicator_signals()
        self.patterns = configs['Patterns']
        self.work_timeframe = configs['Timeframes']['work_timeframe']
        self.higher_timeframe = configs['Timeframes']['higher_timeframe']
        self.timeframe_div = configs['Data']['Basic']['params']['timeframe_div']

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Add all necessary indicator data to dataframe """
        for i in self.indicator_list:
            if i.startswith('STOCH'):
                df['stoch_slowk_dir'] = df['stoch_slowk'].pct_change().rolling(3).mean()
                df['stoch_slowd_dir'] = df['stoch_slowd'].pct_change().rolling(3).mean()
                df['stoch_diff'] = df['stoch_slowk'] - df['stoch_slowd']
                df['stoch_diff'] = df['stoch_diff'].rolling(3).mean()
                break

        for i in self.indicator_list:
            if i.startswith('SUP_RES'):
                df['high_roll'] = df['high'].rolling(3).mean()
                df['low_roll'] = df['low'].rolling(3).mean()
                break
        return df

    def prepare_indicator_signals(self) -> list:
        """ Get all indicator signal classes """
        indicator_signals = list()
        for indicator in self.indicator_list:
            indicator_signals.append(SignalFactory.factory(indicator, self.configs))
        return indicator_signals

    def find_signal(self, dfs: dict, ticker: str, timeframe: str, data_qty: int) -> list:
        """ Search for the signals through the dataframe, if found - add its index and trade type to the list.
            If dataset was updated - don't search through the whole dataset, only through updated part.
        """
        points = list()
        index_list = list()

        try:
            df_work = dfs[ticker][self.work_timeframe]['data']
            levels = dfs[ticker][self.work_timeframe]['levels']
            df_higher = dfs[ticker][self.higher_timeframe]['data']
        except KeyError:
            return points

        df_work = self.prepare_dataframe(df_work)
        # Support and Resistance indicator
        sup_res = None
        # Level proximity measure * multiplier (from configs)
        level_proximity = None

        for indicator_signal in self.indicator_signals:
            if indicator_signal.name == 'SUP_RES':
                sup_res = indicator_signal
                level_proximity = np.mean(df_work['high'] - df_work['low']) * sup_res.proximity_multiplier
                break

        for index in range(max(df_work.shape[0] - data_qty, 2), df_work.shape[0]):
            time = df_work.loc[index, 'time']
            sig_patterns = [p.copy() for p in self.patterns]
            # Check if current signal is found and add '1' to all patterns where it was added
            for indicator_signal in self.indicator_signals:
                # if indicator is linear regression - find its angle from dataframe from higher timeframe
                if indicator_signal.name == "LinearReg":
                    # get time ratio between higher timeframe and working timeframe
                    timeframe_ratio = int(self.timeframe_div[self.higher_timeframe] /
                                          self.timeframe_div[self.work_timeframe])
                    # transform working timeframe index to higher timeframe index according to timeframe_ratio
                    higher_index = df_higher.shape[0] - int((df_higher.shape[0] - index - 1)/timeframe_ratio) - 1
                    fs = indicator_signal.find_signal(df_higher, higher_index, levels)
                else:
                    fs = indicator_signal.find_signal(df_work, index, levels)
                if fs[0] is True:
                    for pattern in sig_patterns:
                        if indicator_signal.name in pattern:
                            pattern[pattern.index(indicator_signal.name)] = \
                                (pattern[pattern.index(indicator_signal.name)], fs[0], fs[1], fs[2])

            # If any pattern has all 1 - add corresponding point as signal
            for pattern in sig_patterns:
                point = None
                for p in pattern:
                    if (p[1], p[2]) != (True, 'buy') and (p[1], p[2]) != (True, ''):
                        break
                else:
                    if "SUP_RES" in [p[0] for p in pattern]:
                        if sup_res.check_levels(df_work, index, levels, level_proximity, True):
                            point = [ticker, timeframe, index, 'buy', time, [(p[0], p[3]) for p in pattern],
                                     [], [], [], []]
                    else:
                        point = [ticker, timeframe, index, 'buy', time, [(p[0], p[3]) for p in pattern],
                                 [], [], [], []]

                for p in pattern:
                    if (p[1], p[2]) != (True, 'sell') and (p[1], p[2]) != (True, ''):
                        break
                else:
                    if "SUP_RES" in [p[0] for p in pattern]:
                        if sup_res.check_levels(df_work, index, levels, level_proximity, False):
                            point = [ticker, timeframe, index, 'sell', time, [(p[0], p[3]) for p in pattern],
                                     [], [], [], []]
                    else:
                        point = [ticker, timeframe, index, 'sell', time, [(p[0], p[3]) for p in pattern],
                                 [], [], [], []]
                if point:  # and index not in index_list:
                    index_list.append(index)
                    points.append(point)

        return points
