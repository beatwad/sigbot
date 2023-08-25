import numpy as np
import pandas as pd
from abc import abstractmethod


class SignalFactory(object):
    """ Return indicator according to 'indicator' variable value """

    @staticmethod
    def factory(indicator, ttype, configs):
        if indicator == 'RSI':
            return RSISignal(ttype, **configs)
        elif indicator == 'STOCH':
            return STOCHSignal(ttype, **configs)
        elif indicator == 'MACD':
            return MACDSignal(ttype, **configs)
        elif indicator == 'Pattern':
            return PatternSignal(ttype, **configs)
        elif indicator == 'PumpDump':
            return PumpDumpSignal(ttype, **configs)
        elif indicator == 'Trend':
            return TrendSignal(ttype, **configs)
        elif indicator == 'HighVolume':
            return HighVolumeSignal(ttype, **configs)


class SignalBase:
    """ Base signal searching class """
    type = 'Indicator_signal'
    name = 'Base'

    def __init__(self, ttype, configs):
        self.ttype = ttype
        self.configs = configs[self.type][self.ttype]
        self.vol_q_high = 0
        self.vol_q_low = 0
        self.vol_window = 0
        self.first_candle = 0
        self.second_candle = 0

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

    def two_good_candles(self, df: pd.DataFrame, ttype: str) -> np.ndarray:
        """ Get two candles that confirm pattern movement """
        # use high/low volume to confirm pattern
        vol_q_high = df['volume'].rolling(self.vol_window).quantile(self.vol_q_high)
        vol_q_low = df['volume'].rolling(self.vol_window).quantile(self.vol_q_low)
        first_candle_vol = df['volume'].shift(1)
        second_candle_vol = df['volume'].shift(2)
        # find two candles
        if ttype == 'buy':
            sign_1 = np.where(df['close'].shift(2) > df['open'].shift(2), 1, -1)
            sign_2 = np.where(df['close'].shift(1) > df['open'].shift(1), 1, -1)
            first_candle = (df['close'].shift(2) - df['low'].shift(2)) / \
                           (df['high'].shift(2) - df['low'].shift(2)) * sign_1
            second_candle = (df['close'].shift(1) - df['low'].shift(1)) / \
                            (df['high'].shift(1) - df['low'].shift(1)) * sign_2
        else:
            sign_1 = np.where(df['close'].shift(2) < df['open'].shift(2), 1, -1)
            sign_2 = np.where(df['close'].shift(1) < df['open'].shift(1), 1, -1)
            first_candle = (df['high'].shift(2) - df['close'].shift(2)) / \
                           (df['high'].shift(2) - df['low'].shift(2)) * sign_1
            second_candle = (df['high'].shift(1) - df['close'].shift(1)) / \
                            (df['high'].shift(1) - df['low'].shift(1)) * sign_2
        return np.where(
                        (first_candle >= self.first_candle) &
                        (((first_candle_vol >= vol_q_high) & (second_candle_vol >= vol_q_high)) |
                         ((first_candle_vol <= vol_q_low) & (second_candle_vol <= vol_q_low))) &
                        (second_candle >= self.second_candle),
                        1, 0)


class STOCHSignal(SignalBase):
    """ Check if STOCH is in overbuy/oversell zone and is going to change its direction to opposite """
    type = 'Indicator_signal'
    name = 'STOCH'

    def __init__(self, ttype: str, **configs):
        super(STOCHSignal, self).__init__(ttype, configs)
        self.configs = self.configs[self.name]['params']
        self.low_bound = self.configs.get('low_bound', 20)
        self.high_bound = self.configs.get('high_bound', 80)

    def find_signal(self, df: pd.DataFrame, *args) -> np.ndarray:
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

        if self.ttype == 'sell':
            lower_bound_slowk = self.lower_bound(self.low_bound, stoch_slowk, stoch_slowk_lag_1, stoch_slowk_lag_2)
            lower_bound_slowd = self.lower_bound(self.low_bound, stoch_slowd, stoch_slowd_lag_1, stoch_slowd_lag_2)
            crossed_lines_down = self.crossed_lines(False, stoch_diff, stoch_diff_lag_1, stoch_diff_lag_2)
            up_direction_slowk = self.up_direction(df['stoch_slowk_dir'])
            up_direction_slowd = self.up_direction(df['stoch_slowd_dir'])
            stoch_up = (lower_bound_slowk & lower_bound_slowd & crossed_lines_down & up_direction_slowk &
                        up_direction_slowd)
            return stoch_up

        higher_bound_slowk = self.higher_bound(self.high_bound, stoch_slowk, stoch_slowk_lag_1, stoch_slowk_lag_2)
        higher_bound_slowd = self.higher_bound(self.high_bound, stoch_slowd, stoch_slowd_lag_1, stoch_slowd_lag_2)
        crossed_lines_up = self.crossed_lines(True, stoch_diff, stoch_diff_lag_1, stoch_diff_lag_2)
        down_direction_slowk = self.down_direction(df['stoch_slowk_dir'])
        down_direction_slowd = self.down_direction(df['stoch_slowd_dir'])
        stoch_down = (higher_bound_slowk & higher_bound_slowd & crossed_lines_up & down_direction_slowk &
                      down_direction_slowd)
        return stoch_down


class RSISignal(SignalBase):
    """ Check if RSI is in overbuy/oversell zone """
    type = 'Indicator_signal'
    name = "RSI"

    def __init__(self, ttype: str, **configs):
        super(RSISignal, self).__init__(ttype, configs)
        self.configs = self.configs[self.name]['params']
        self.low_bound = self.configs.get('low_bound', 25)
        self.high_bound = self.configs.get('high_bound', 75)

    def find_signal(self, df: pd.DataFrame, *args) -> np.ndarray:
        """ 1 - Return true if RSI is lower than low bound (buy signal)
            2 - Return true if RSI is higher than low bound (sell signal)  """
        # Find RSI signal
        rsi = df['rsi']
        rsi_lag_1 = df['rsi'].shift(1)
        rsi_lag_2 = df['rsi'].shift(2)
        if self.ttype == 'sell':
            rsi_lower = self.lower_bound(self.low_bound, rsi, rsi_lag_1, rsi_lag_2)
            return rsi_lower
        rsi_higher = self.higher_bound(self.high_bound, rsi, rsi_lag_1, rsi_lag_2)
        return rsi_higher


class TrendSignal(SignalBase):
    """ Check trend using linear regression indicator """
    type = 'Indicator_signal'
    name = "Trend"

    def __init__(self, ttype, **configs):
        super(TrendSignal, self).__init__(ttype, configs)
        self.configs = self.configs[self.name]['params']
        self.low_bound = self.configs.get('low_bound', 0)
        self.high_bound = self.configs.get('high_bound', 0)

    def find_signal(self, df: pd.DataFrame, trade_points: pd.DataFrame, *args) -> np.ndarray:
        """ 1 - Return true if trend is moving up (buy signal)
            2 - Return true if trend is moving down (sell signal)  """
        # According to difference between working timeframe and higher timeframe for every point on working timeframe
        # find corresponding value of Linear Regression from higher timeframe
        linear_reg_angle = pd.merge(trade_points, df, how='left', left_on='time_higher',
                                    right_on='time')['linear_reg_angle']
        x = pd.merge(trade_points, df, how='left', left_on='time_higher', right_on='time')
        # buy trade
        if self.ttype == 'buy':
            # find Linear Regression signal
            lr_higher_bound = self.higher_bound_robust(self.high_bound, linear_reg_angle)
            return lr_higher_bound

        # same for the sell trade
        # find Linear Regression signal
        lr_lower_bound = self.lower_bound_robust(self.low_bound, linear_reg_angle)
        return lr_lower_bound


class PumpDumpSignal(SignalBase):
    type = 'Indicator_signal'
    name = 'PumpDump'

    def __init__(self, ttype, **configs):
        super(PumpDumpSignal, self).__init__(ttype, configs)

    def find_signal(self, df: pd.DataFrame) -> np.ndarray:
        """ 1 - Price rapidly moves down in one candle (buy signal)
            2 - Price rapidly moves up  in one candle (sell signal)  """
        # Find price change signal
        # buy trade
        if self.ttype == 'buy':
            price_change_lower_1 = self.lower_bound_robust(df['q_low_lag_1'].loc[0], df['price_change_1'])
            price_change_lower_2 = 0
            price_change_lower_3 = 0
            return price_change_lower_1 | price_change_lower_2 | price_change_lower_3
        # sell trade
        price_change_higher_1 = self.higher_bound_robust(df['q_high_lag_1'].loc[0], df['price_change_1'])
        price_change_higher_2 = 0
        price_change_higher_3 = 0
        return price_change_higher_1 | price_change_higher_2 | price_change_higher_3


class HighVolumeSignal(SignalBase):
    type = 'Indicator_signal'
    name = 'HighVolume'

    def __init__(self, ttype, **configs):
        super(HighVolumeSignal, self).__init__(ttype, configs)

    def find_signal(self, df: pd.DataFrame) -> np.ndarray:
        """ Find high volume signal  """
        # Find high volume signal
        high_vol = self.higher_bound_robust(df['quantile_vol'].loc[0], df['normalized_vol'])
        return high_vol


class MACDSignal(SignalBase):
    type = 'Indicator_signal'
    name = 'MACD'

    def __init__(self, ttype, **configs):
        super(MACDSignal, self).__init__(ttype, configs)
        self.configs = self.configs[self.name]['params']
        self.low_bound = self.configs.get('low_bound', 20)
        self.high_bound = self.configs.get('high_bound', 80)

    def find_signal(self, df: pd.DataFrame, trade_points: pd.DataFrame, timeframe_ratio: int) -> np.ndarray:
        """ Find MACD signal """
        macd_df = pd.merge(trade_points, df, how='left',
                           left_on='time_higher', right_on='time')[['macdhist', 'macd_dir', 'macdsignal_dir']]
        macd_df['macdhist_1'] = macd_df['macdhist'].shift(timeframe_ratio)
        macd_df['macdhist_2'] = macd_df['macdhist'].shift(timeframe_ratio * 2)

        if self.ttype == 'buy':
            crossed_lines_down = self.crossed_lines(False, macd_df['macdhist'],
                                                    macd_df['macdhist_1'], macd_df['macdhist_2'])
            up_direction_macd = self.up_direction(macd_df['macd_dir'])
            up_direction_macdsignal = self.up_direction(macd_df['macdsignal_dir'])
            macd_up = crossed_lines_down & up_direction_macd & up_direction_macdsignal
            return macd_up

        crossed_lines_up = self.crossed_lines(True, macd_df['macdhist'],
                                              macd_df['macdhist_1'], macd_df['macdhist_2'])
        down_direction_slowk = self.down_direction(macd_df['macd_dir'])
        down_direction_slowd = self.down_direction(macd_df['macdsignal_dir'])
        macd_down = crossed_lines_up & down_direction_slowk & down_direction_slowd
        return macd_down


class PatternSignal(SignalBase):
    type = 'Indicator_signal'
    name = 'Pattern'

    def __init__(self, ttype, **configs):
        super(PatternSignal, self).__init__(ttype, configs)
        self.ttype = ttype
        self.configs = self.configs[self.name]['params']
        self.vol_q_high = self.configs.get('vol_q_high', 0.75)
        self.vol_q_low = self.configs.get('vol_q_low', 0.25)
        self.vol_window = self.configs.get('vol_window', 48)
        self.window_low_bound = self.configs.get('window_low_bound', 1)
        self.window_high_bound = self.configs.get('window_high_bound', 6)
        self.first_candle = self.configs.get('first_candle', 0.667)
        self.second_candle = self.configs.get('second_candle', 0.5)

    def shrink_max_min(self, df: pd.DataFrame, high_max: [pd.DataFrame.index, list],
                       low_min: [pd.DataFrame.index, list]) -> (np.ndarray, np.ndarray):
        """ EM algorithm that allows to leave only important high and low extremums """
        temp_high_max = list()
        temp_low_min = list()
        for i in range(len(low_min) + 1):
            if i == 0:
                interval = high_max[(0 < high_max) & (high_max < low_min[i])]
            elif i == len(low_min):
                interval = high_max[(low_min[i - 1] < high_max) & (high_max < np.inf)]
            else:
                interval = high_max[(low_min[i - 1] < high_max) & (high_max < low_min[i])]
            if len(interval):
                idx = df.loc[interval, 'high'].argmax()
                temp_high_max.append(interval[idx])
        for i in range(len(high_max) + 1):
            if i == 0:
                interval = low_min[(0 < low_min) & (low_min < high_max[i])]
            elif i == len(high_max):
                interval = low_min[(high_max[i - 1] < low_min) & (low_min < np.inf)]
            else:
                interval = low_min[(high_max[i - 1] < low_min) & (low_min < high_max[i])]
            if len(interval):
                idx = df.loc[interval, 'low'].argmin()
                temp_low_min.append(interval[idx])
        if len(temp_high_max) == len(high_max) and len(temp_low_min) == len(low_min):
            if self.ttype == 'buy':
                low_min = low_min[low_min > min(high_max)]
            else:
                high_max = high_max[high_max > min(low_min)]
            return high_max, low_min
        return self.shrink_max_min(df, np.array(temp_high_max), np.array(temp_low_min))

    def get_min_max_indexes(self, df: pd.DataFrame, gmax: pd.DataFrame.index,
                            gmin: pd.DataFrame.index) -> (tuple, tuple):
        if self.ttype == 'buy':  # find H&S
            # find 3 last global maximum
            ei = gmax
            ci = np.append(ei[0], ei[:-1])
            ai = np.append(ei[0:2], ei[:-2])
            # find 3 last global minimum
            fi = gmin
            di = np.append(fi[0], fi[:-1])
            bi = np.append(fi[0:2], fi[:-2])
            # find according high/low values
            aiv = df.loc[ai, 'high'].values
            biv = df.loc[bi, 'low'].values
            civ = df.loc[ci, 'high'].values
            div = df.loc[di, 'low'].values
            eiv = df.loc[ei, 'high'].values
            fiv = df.loc[fi, 'low'].values
        else:  # find inverted H&S
            # find 3 last global minimum
            ei = gmin
            ci = np.append(ei[0], ei[:-1])
            ai = np.append(ei[0:2], ei[:-2])
            # find 3 last global maximum
            fi = gmax
            di = np.append(fi[0], fi[:-1])
            bi = np.append(fi[0:2], fi[:-2])
            # find according high/low values
            aiv = df.loc[ai, 'low'].values
            biv = df.loc[bi, 'high'].values
            civ = df.loc[ci, 'low'].values
            div = df.loc[di, 'high'].values
            eiv = df.loc[ei, 'low'].values
            fiv = df.loc[fi, 'high'].values
        return (ai, bi, ci, di, ei, fi), (aiv, biv, civ, div, eiv, fiv)

    def create_pattern_vector(self, df: pd.DataFrame, res: np.ndarray):
        """ Create vector that shows potential places where we can enter the trade after pattern appearance """
        v = np.zeros(df.shape[0], dtype=int)
        for i in range(self.window_low_bound, self.window_high_bound):
            try:
                v[res[res > 0] + i] = 1
            except IndexError:
                break
        return v

    def head_and_shoulders(self, df: pd.DataFrame, min_max_idxs: tuple, min_max_vals: tuple,
                           avg_gap: float) -> np.ndarray:
        """ Find H&S/inverted H&S pattern """
        ai, bi, ci, di, ei, fi = min_max_idxs
        aiv, biv, civ, div, eiv, fiv = min_max_vals
        if self.ttype == 'buy':  # inverted H&S
            # find if global maximums and minimums make H&S pattern
            res = np.where((1.005 * div < biv) & (1.005 * div < fiv), fi, 0)
        else:  # H&S
            # find if global maximums and minimums make inverted H&S pattern
            res = np.where((div > 1.005 * biv) & (div > 1.005 * fiv), fi, 0)
        v = self.create_pattern_vector(df, res)
        return v

    def hlh_lhl(self, df: pd.DataFrame, min_max_idxs: tuple, min_max_vals: tuple) -> np.ndarray:
        """ Find HLH/LHL pattern """
        _, __, ci, di, ei, fi = min_max_idxs
        _, __, civ, div, eiv, fiv = min_max_vals
        if self.ttype == 'buy':  # LHL
            # find if global maximums and minimums make LHL pattern
            res = np.where((1.005 * civ < eiv) & (1.005 * div < fiv), fi, 0)
        else:  # find HLH
            # find if global maximums and minimums make HLH pattern
            res = np.where((civ > 1.005 * eiv) & (div > 1.005 * fiv), fi, 0)
        v = self.create_pattern_vector(df, res)
        return v

    def dt_db(self, df: pd.DataFrame, min_max_idxs: tuple, min_max_vals: tuple) -> np.ndarray:
        """ Find Double Top/Double Bottom pattern """
        _, __, ci, di, ei, fi = min_max_idxs
        _, __, civ, div, eiv, fiv = min_max_vals
        if self.ttype == 'buy':  # LHL
            # find if global maximums and minimums make DP pattern
            res = np.where((civ > eiv) & (np.abs(div - fiv)/div < 0.001), fi, 0)
        else:  # find HLH
            # find if global maximums and minimums make DB pattern
            res = np.where((civ < eiv) & (np.abs(div - fiv)/div < 0.001), fi, 0)
        v = self.create_pattern_vector(df, res)
        return v

    def triangle(self, df: pd.DataFrame, min_max_idxs: tuple, min_max_vals: tuple) -> np.ndarray:
        """ Find ascending/descending triangle pattern """
        ai, bi, ci, di, ei, fi = min_max_idxs
        aiv, biv, civ, div, eiv, fiv = min_max_vals
        if self.ttype == 'buy':  # ascending triangle
            # find if global maximums and minimums make ascending triangle pattern
            res = np.where((biv < div) & (div < fiv) & (np.abs(aiv - civ) / aiv < 0.001)
                           & (np.abs(civ - eiv) / civ < 0.001) & (np.abs(aiv - eiv) / aiv < 0.001), fi, 0)
        else:  # descending triangle
            # find if global maximums and minimums make descending triangle pattern
            res = np.where((biv > div) & (div > fiv) & (np.abs(aiv - civ) / aiv < 0.001)
                           & (np.abs(civ - eiv) / civ < 0.001) & (np.abs(aiv - eiv) / aiv < 0.001), fi, 0)
        v = self.create_pattern_vector(df, res)
        return v

    def swing(self, df: pd.DataFrame, min_max_idxs: tuple, min_max_vals: tuple, avg_gap: float) -> np.ndarray:
        """ Find swing patter """
        ai, bi, ci, di, ei, fi = min_max_idxs
        aiv, biv, civ, div, eiv, fiv = min_max_vals
        res = np.where((np.abs(aiv - civ) / aiv < 0.0025) & (np.abs(civ - eiv) / civ < 0.0025) &
                       (np.abs(aiv - eiv) / aiv < 0.0025) & (np.abs(biv - div) / biv < 0.0025) &
                       (np.abs(div - fiv) / div < 0.0025) & (np.abs(biv - fiv) / biv < 0.0025) &
                       (np.abs(aiv - fiv) <= avg_gap), fi, 0)
        v = self.create_pattern_vector(df, res)
        return v

    def find_signal(self, df: pd.DataFrame) -> np.ndarray:
        # Find one of TA patterns like H&S, HLH/LHL, DT/DB and good candles that confirm that pattern
        avg_gap = (df['high'] - df['low']).mean()
        high_max = df[df['high_max'] > 0].index
        low_min = df[df['low_min'] > 0].index
        # bring both global extremum lists to one length
        min_len = min(len(high_max), len(low_min))
        if min_len == 0:
            return np.zeros(df.shape[0])
        gmax, gmin = high_max[:min_len], low_min[:min_len]
        # find minimum and maximum indexes for patterns search
        min_max_idxs, min_max_vals = self.get_min_max_indexes(df, gmax, gmin)
        has = self.head_and_shoulders(df, min_max_idxs, min_max_vals, avg_gap)
        hlh = self.hlh_lhl(df, min_max_idxs, min_max_vals)
        dt = self.dt_db(df, min_max_idxs, min_max_vals)
        tgc = self.two_good_candles(df, self.ttype)
        sw = self.swing(df, min_max_idxs, min_max_vals, avg_gap)
        pattern_signal = (has | hlh | dt | sw) & tgc
        return pattern_signal


class FindSignal:
    """ Class for searching of the indicator combination """

    def __init__(self, ttype, configs):
        self.ttype = ttype
        self.configs = configs
        self.indicator_list = configs['Indicator_list']
        self.indicator_signals = self.prepare_indicator_signals()
        self.patterns = configs['Patterns']
        self.work_timeframe = configs['Timeframes']['work_timeframe']
        self.higher_timeframe = configs['Timeframes']['higher_timeframe']
        self.timeframe_div = configs['Data']['Basic']['params']['timeframe_div']

    def prepare_indicator_signals(self) -> list:
        """ Get all indicator signal classes """
        indicator_signals = list()
        for indicator in self.indicator_list:
            if (indicator == 'HighVolume' and self.ttype == 'sell') or indicator == 'ATR':
                continue
            indicator_signals.append(SignalFactory.factory(indicator, self.ttype, self.configs))
        return indicator_signals

    def find_signal(self, dfs: dict, ticker: str, timeframe: str, data_qty: int, data_qty_higher: int) -> list:
        """ Search for the signals through the dataframe, if found - add its index and trade type to the list.
            If dataset was updated - don't search through the whole dataset, only through updated part.
        """
        points = list()

        try:
            if self.ttype == 'buy':
                df_work = dfs[ticker][self.work_timeframe]['data']['buy'].copy()
                df_higher = dfs[ticker][self.higher_timeframe]['data']['buy'].copy()
            else:
                df_work = dfs[ticker][self.work_timeframe]['data']['sell'].copy()
                df_higher = dfs[ticker][self.higher_timeframe]['data']['sell'].copy()
        except KeyError:
            return points

        sig_patterns = [p.copy() for p in self.patterns]
        timeframe_ratio = int(self.timeframe_div[self.higher_timeframe] / self.timeframe_div[self.work_timeframe])
        # Create signal point df for each indicator 
        trade_points, tmp = pd.DataFrame(), pd.DataFrame()
        trade_points['time'] = df_work['time']
        # merge higher dataframe timestamps and working dataframe timestamps in one dataframe
        tmp['time'] = df_higher['time']
        tmp['time_higher'] = tmp['time']
        # merge work timeframe with higher timeframe, so we can work with indicator values from higher timeframe
        trade_points = pd.merge(trade_points[['time']], tmp[['time', 'time_higher']], how='left')
        trade_points['time_higher'] = trade_points['time_higher'].fillna(method='ffill')
        trade_points['time_higher'] = trade_points['time_higher'].fillna(trade_points['time_higher'].min() -
                                                                         pd.to_timedelta(int(self.higher_timeframe[:-1]),
                                                                                         self.higher_timeframe[-1]))
        # Fill signal point df with signals from indicators
        for indicator_signal in self.indicator_signals:
            # if indicators work with higher timeframe - we should treat them differently
            if indicator_signal.name == "Trend":
                fs = indicator_signal.find_signal(df_higher, trade_points)
            elif indicator_signal.name == "MACD":
                # check higher timeframe signals every hour
                if data_qty_higher > 1:
                    fs = indicator_signal.find_signal(df_higher, trade_points, timeframe_ratio)
                else:
                    fs = np.zeros(trade_points.shape[0])
            else:
                fs = indicator_signal.find_signal(df_work)

            trade_points[indicator_signal.name] = fs

        # If any pattern has all 1 - add corresponding point as a signal
        for pattern in sig_patterns:
            # find indexes of trade points
            if self.ttype == 'sell' and pattern == ['HighVolume']:
                continue
            # check if pattern has all 1
            pattern_points = trade_points[pattern]
            max_shape = pattern_points.shape[1]
            pattern_points = pattern_points.sum(axis=1)
            # get trade indexes
            trade_indexes = pattern_points[pattern_points == max_shape].index
            # save only recent trade indexes
            trade_indexes = trade_indexes[df_work.shape[0] - trade_indexes < data_qty]
            sig_pattern = '_'.join(pattern)
            if sig_pattern == 'MACD':  # or sig_pattern == 'Pattern_Trend':
                # sparse signal indexes for higher timeframe
                trade_indexes = [trade_indexes[i] for i in range(0, len(trade_indexes), timeframe_ratio)]
                points += [[ticker, self.higher_timeframe, index, self.ttype, trade_points.loc[index, 'time_higher'],
                            sig_pattern, [], [], [], []] for index in trade_indexes]
            else:
                points += [[ticker, self.work_timeframe, index, self.ttype, trade_points.loc[index, 'time'],
                            sig_pattern, [], [], [], []] for index in trade_indexes]
        return points
