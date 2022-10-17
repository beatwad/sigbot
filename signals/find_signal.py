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
    def lower_bound(low_bound: float, indicator: pd.Series, indicator_lag_1: pd.Series,
                         indicator_lag_2: pd.Series) -> np.ndarray:
        """ Returns True if at least two of three last points of indicator are higher than high_bound param """
        indicator = np.where(indicator < low_bound, 1, 0)
        indicator_lag_1 = np.where(indicator_lag_1 < low_bound, 1, 0)
        indicator_lag_2 = np.where(indicator_lag_2 < low_bound, 1, 0)
        indicator_sum = np.array([indicator, indicator_lag_1, indicator_lag_2]).sum(axis=0)
        return np.where(indicator_sum >= 2, 1, 0)

    @staticmethod
    def higher_bound(high_bound: float, indicator: pd.Series, indicator_lag_1: pd.Series,
                          indicator_lag_2: pd.Series) -> np.ndarray:
        """ Returns True if at least two of three last points of indicator are higher than high_bound param """
        indicator = np.where(indicator > high_bound, 1, 0)
        indicator_lag_1 = np.where(indicator_lag_1 > high_bound, 1, 0)
        indicator_lag_2 = np.where(indicator_lag_2 > high_bound, 1, 0)
        indicator_sum = np.array([indicator, indicator_lag_1, indicator_lag_2]).sum(axis=0)
        return np.where(indicator_sum >= 2, 1, 0)

    @staticmethod
    def lower_bound_robust(low_bound: float, indicator: pd.Series) -> np.ndarray:
        """ Returns True if indicator is lower than low bound """
        return np.where(indicator < low_bound, 1, 0)

    @staticmethod
    def higher_bound_robust(high_bound: float, indicator: pd.Series) -> np.ndarray:
        """ Returns True if indicator is lower than low bound """
        return np.where(indicator > high_bound, 1, 0)

    @staticmethod
    def up_direction(indicator: pd.Series) -> np.ndarray:
        """ Returns True if indicator is lower than low bound """
        return np.where(indicator > 0, 1, 0)

    @staticmethod
    def down_direction(indicator: pd.Series) -> np.ndarray:
        """ Return True if indicator values are moving down """
        return np.where(indicator < 0, 1, 0)


class STOCHSignal(SignalBase):
    """ Check if STOCH is in overbuy/oversell zone and is going to change its direction to opposite """
    type = 'Indicator_signal'
    name = 'STOCH'

    def __init__(self, **configs):
        super(STOCHSignal, self).__init__(configs)
        self.low_bound = self.configs.get('low_bound', 20)
        self.high_bound = self.configs.get('high_bound', 80)

    @staticmethod
    def crossed_lines(up: bool, indicator: pd.Series,
                           indicator_lag_1: pd.Series, indicator_lag_2: pd.Series) -> np.ndarray:
        """ Returns True if slowk and slowd lines of RSI has crossed """
        if up:
            indicator = np.where(indicator < 0, 1, 0)
            indicator_lag_1 = np.where(indicator_lag_1 > 0, 1, 0)
            indicator_lag_1 = np.array([indicator, indicator_lag_1]).sum(axis=0)
            indicator_lag_2 = np.where(indicator_lag_2 > 0, 1, 0)
            indicator_lag_2 = np.array([indicator, indicator_lag_2]).sum(axis=0)
        else:
            indicator = np.where(indicator > 0, 1, 0)
            indicator_lag_1 = np.where(indicator_lag_1 < 0, 1, 0)
            indicator_lag_1 = np.array([indicator, indicator_lag_1]).sum(axis=0)
            indicator_lag_2 = np.where(indicator_lag_2 < 0, 1, 0)
            indicator_lag_2 = np.array([indicator, indicator_lag_2]).sum(axis=0)
        indicator = np.maximum(indicator_lag_1, indicator_lag_2)
        return np.where(indicator > 1, 1, 0)

    def find_signal(self, df: pd.DataFrame, *args) -> (np.ndarray, np.ndarray):
        """ 1 - Return true if Stochastic is lower than low bound, lines have crossed and look up (buy signal)
            2 - Return true if Stochastic is higher than high bound, lines have crossed and look down (sell signal)  """
        # Find STOCH signal
        stoch_slowk = df['stoch_slowk']
        stoch_slowk_lag_1 = df['stoch_slowk'].shift(1)
        stoch_slowk_lag_2 = df['stoch_slowk'].shift(2)
        stoch_slowd = df['stoch_slowd']
        stoch_slowd_lag_1 = df['stoch_slowd'].shift(1)
        stoch_slowd_lag_2 = df['stoch_slowd'].shift(2)
        stoch_diff = df['stoch_diff']
        stoch_diff_lag_1 = df['stoch_diff'].shift(1)
        stoch_diff_lag_2 = df['stoch_diff'].shift(2)

        lower_bound_slowk = self.lower_bound(self.low_bound, stoch_slowk, stoch_slowk_lag_1, stoch_slowk_lag_2)
        lower_bound_slowd = self.lower_bound(self.low_bound, stoch_slowd, stoch_slowd_lag_1, stoch_slowd_lag_2)
        crossed_lines_down = self.crossed_lines(False, stoch_diff, stoch_diff_lag_1, stoch_diff_lag_2)
        up_direction_slowk = self.up_direction(df['stoch_slowk_dir'])
        up_direction_slowd = self.up_direction(df['stoch_slowd_dir'])
        stoch_up = lower_bound_slowk & lower_bound_slowd & crossed_lines_down & up_direction_slowk & up_direction_slowd

        higher_bound_slowk = self.higher_bound(self.high_bound, stoch_slowk, stoch_slowk_lag_1, stoch_slowk_lag_2)
        higher_bound_slowd = self.higher_bound(self.high_bound, stoch_slowd, stoch_slowd_lag_1, stoch_slowd_lag_2)
        crossed_lines_up = self.crossed_lines(True, stoch_diff, stoch_diff_lag_1, stoch_diff_lag_2)
        down_direction_slowk = self.down_direction(df['stoch_slowk_dir'])
        down_direction_slowd = self.down_direction(df['stoch_slowd_dir'])
        stoch_down = higher_bound_slowk & higher_bound_slowd & crossed_lines_up & down_direction_slowk & \
                     down_direction_slowd

        return stoch_up, stoch_down


class RSISignal(SignalBase):
    """ Check if RSI is in overbuy/oversell zone """
    type = 'Indicator_signal'
    name = "RSI"

    def __init__(self, **configs):
        super(RSISignal, self).__init__(configs)
        self.low_bound = self.configs.get('low_bound', 25)
        self.high_bound = self.configs.get('high_bound', 75)

    def find_signal(self, df: pd.DataFrame, *args) -> (bool, str, tuple):
        """ 1 - Return true if RSI is lower than low bound (buy signal)
            2 - Return true if RSI is higher than low bound (sell signal)  """
        # Find RSI signal
        rsi = df['rsi']
        rsi_lag_1 = df['rsi'].shift(1)
        rsi_lag_2 = df['rsi'].shift(2)
        rsi_lower = self.lower_bound(self.low_bound, rsi, rsi_lag_1, rsi_lag_2)
        rsi_higher = self.higher_bound(self.high_bound, rsi, rsi_lag_1, rsi_lag_2)
        return rsi_lower, rsi_higher


class LinearRegSignal(SignalBase):
    """ Check trend using linear regression indicator """
    type = 'Indicator_signal'
    name = "LinearReg"

    def __init__(self, **configs):
        super(LinearRegSignal, self).__init__(configs)
        self.low_bound = self.configs.get('low_bound', -0.1)
        self.high_bound = self.configs.get('high_bound', 0.1)

    def find_signal(self, df: pd.DataFrame, timeframe_ratio: int, working_df_len: int, points_shape: int,
                    *args) -> (bool, str, tuple):
        """ 1 - Return true if trend is moving up (buy signal)
            2 - Return true if trend is moving down (sell signal)  """
        # According to difference between working timeframe and higher timeframe for every point on working timeframe
        # find corresponding value of Linear Regression from higher timeframe
        higher_shape = int(working_df_len / timeframe_ratio) + 1
        linear_reg_angle = df.loc[max(df.shape[0] - higher_shape, 0):, 'linear_reg_angle']
        # Then find Linear Regression signal
        lr_lower_bound = self.lower_bound_robust(self.low_bound, linear_reg_angle)
        lr_higher_bound = self.higher_bound_robust(self.high_bound, linear_reg_angle)
        # Then for each working timeframe point add corresponding LR singal
        # and make size of the signal the same as size of working timeframe df
        lr_lower_bound = np.repeat(lr_lower_bound, timeframe_ratio)
        lr_lower_bound = lr_lower_bound[max(lr_lower_bound.shape[0] - working_df_len, 0):]
        lr_higher_bound = np.repeat(lr_higher_bound, timeframe_ratio)
        lr_higher_bound = lr_higher_bound[max(lr_higher_bound.shape[0] - working_df_len, 0):]
        # If signal shape differs from point shape - bring them to one shape
        if lr_lower_bound.shape[0] < points_shape:
            diff = points_shape - lr_lower_bound.shape[0]
            lr_lower_bound = np.concatenate([np.zeros(diff), lr_lower_bound])
            lr_higher_bound = np.concatenate([np.zeros(diff), lr_higher_bound])
        return lr_higher_bound, lr_lower_bound


class PriceChangeSignal(SignalBase):
    type = 'Indicator_signal'
    name = 'PriceChange'

    def __init__(self, **configs):
        super(PriceChangeSignal, self).__init__(configs)

    def find_signal(self, df: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """ 1 - Price rapidly moves down in one candle (buy signal)
            2 - Price rapidly moves up  in one candle (sell signal)  """
        # Find price change signal
        price_change_lower_1 = self.lower_bound_robust(df['q_low_lag_1'].loc[0], df['close_price_change_lag_1'])
        price_change_higher_1 = self.higher_bound_robust(df['q_high_lag_1'].loc[0], df['close_price_change_lag_1'])
        price_change_lower_2 = self.lower_bound_robust(df['q_low_lag_2'].loc[0], df['close_price_change_lag_2'])
        price_change_higher_2 = self.higher_bound_robust(df['q_high_lag_2'].loc[0], df['close_price_change_lag_2'])
        price_change_lower_3 = self.lower_bound_robust(df['q_low_lag_3'].loc[0], df['close_price_change_lag_3'])
        price_change_higher_3 = self.higher_bound_robust(df['q_high_lag_3'].loc[0], df['close_price_change_lag_3'])
        return price_change_lower_1 | price_change_lower_2 | price_change_lower_3, \
               price_change_higher_1 | price_change_higher_2 | price_change_higher_3


class MACDSignal(SignalBase):
    type = 'Indicator_signal'
    name = 'MACD'

    def __init__(self, **configs):
        super(MACDSignal, self).__init__(configs)
        self.low_bound = self.configs.get('low_bound', 20)
        self.high_bound = self.configs.get('high_bound', 80)

    def find_signal(self, *args, **kwargs):
        return False, '', ()


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

        try:
            df_work = dfs[ticker][self.work_timeframe]['data']
            df_higher = dfs[ticker][self.higher_timeframe]['data']
        except KeyError:
            return points

        df_work = self.prepare_dataframe(df_work)
        sig_patterns = [p.copy() for p in self.patterns]

        buy_points = pd.DataFrame()
        sell_points = pd.DataFrame()
        buy_points['time'] = sell_points['time'] = df_work['time']

        # Get signal points for each indicator
        for indicator_signal in self.indicator_signals:
            if indicator_signal.name == "LinearReg":
                # get time ratio between higher timeframe and working timeframe
                timeframe_ratio = int(self.timeframe_div[self.higher_timeframe] /
                                      self.timeframe_div[self.work_timeframe])
                fs_buy, fs_sell = indicator_signal.find_signal(df_higher, timeframe_ratio, df_work.shape[0],
                                                               buy_points.shape[0])
            else:
                fs_buy, fs_sell = indicator_signal.find_signal(df_work)

            buy_points[indicator_signal.name] = fs_buy
            sell_points[indicator_signal.name] = fs_sell

        # If any pattern has all 1 - add corresponding point as signal
        for pattern in sig_patterns:
            # find indexes of buy points
            pattern_points = buy_points[pattern]
            max_shape = pattern_points.shape[1]
            pattern_points = pattern_points.sum(axis=1)
            buy_indexes = pattern_points[pattern_points == max_shape].index
            buy_indexes = buy_indexes[df_work.shape[0] - buy_indexes < data_qty]
            # find indexes of sell points
            pattern_points = sell_points[pattern]
            pattern_points = pattern_points.sum(axis=1)
            sell_indexes = pattern_points[pattern_points == max_shape].index
            sell_indexes = sell_indexes[df_work.shape[0] - sell_indexes < data_qty]
            sig_pattern = '_'.join(pattern)
            points += [[ticker, timeframe, index, 'buy', buy_points.loc[index, 'time'], sig_pattern, [], [], [], []]
                       for index in buy_indexes]
            points += [[ticker, timeframe, index, 'sell', sell_points.loc[index, 'time'], sig_pattern, [], [], [], []]
                       for index in sell_indexes]

        return points
