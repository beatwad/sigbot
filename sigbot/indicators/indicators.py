import json
import numpy as np
import pandas as pd
import talib as ta
from abc import abstractmethod
from collections import Counter


class IndicatorFactory(object):
    """ Return indicator according to 'indicator' variable value """
    @staticmethod
    def factory(indicator, ttype, configs):
        if indicator.startswith('RSI'):
            return RSI(ttype, configs)
        elif indicator.startswith('STOCH'):
            return STOCH(ttype, configs)
        elif indicator.startswith('MACD'):
            return MACD(ttype, configs)
        elif indicator.startswith('PriceChange'):
            return PriceChange(ttype, configs)
        elif indicator.startswith('LinearReg'):
            return LinearReg(ttype, configs)
        elif indicator.startswith('HighVolume'):
            return HighVolume(ttype, configs)
        elif indicator.startswith('Pattern'):
            return Pattern(ttype, configs)
        elif indicator.startswith('ATR'):
            return ATR(ttype, configs)


class Indicator:
    """ Abstract indicator class """
    type = 'Indicator'
    name = 'Base'

    def __init__(self, ttype, configs):
        self.ttype = ttype
        self.configs = configs[self.type][self.ttype][self.name]['params']

    @abstractmethod
    def get_indicator(self, *args, **kwargs):
        """ Get indicator data and write it to the dataframe """
        pass


class RSI(Indicator):
    """ RSI indicator, default settings: timeperiod: 14"""
    name = 'RSI'

    def __init__(self, ttype: str, configs: dict):
        super(RSI, self).__init__(ttype, configs)

    def get_indicator(self, df, ticker: str, timeframe: str, data_qty: int, *args) -> pd.DataFrame:
        # if mean close price value is too small, RSI indicator can become zero,
        # so we should increase it to at least 1e-4
        if df['close'].mean() < 1e-4:
            multiplier = int(1e-4/df['close'].mean()) + 1
            rsi = ta.RSI(df['close'] * multiplier, **self.configs)
        else:
            rsi = ta.RSI(df['close'], **self.configs)
        df['rsi'] = rsi
        return df


class STOCH(Indicator):
    """ STOCH indicator, default settings: fastk_period: 14, slowk_period: 3, slowd_period:  3 """
    name = 'STOCH'

    def __init__(self, ttype: str, configs: dict):
        super(STOCH, self).__init__(ttype, configs)

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args) -> pd.DataFrame:
        slowk, slowd = ta.STOCH(df['high'], df['low'], df['close'], **self.configs)
        df['stoch_slowk'] = slowk
        df['stoch_slowd'] = slowd
        # add auxilary data
        df['stoch_slowk_dir'] = df['stoch_slowk'].pct_change().rolling(3).mean()
        df['stoch_slowd_dir'] = df['stoch_slowd'].pct_change().rolling(3).mean()
        df['stoch_diff'] = df['stoch_slowk'] - df['stoch_slowd']
        df['stoch_diff'] = df['stoch_diff'].rolling(3).mean()
        return df


class LinearReg(Indicator):
    """ Indicator of linear regression and its angle indicators, default settings: timeperiod: 14 """
    name = 'LinearReg'

    def __init__(self, ttype: str, configs: dict):
        super(LinearReg, self).__init__(ttype, configs)

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args) -> pd.DataFrame:
        adx = ta.ADX(df['high'], df['low'], df['close'], **self.configs)
        plus_di = ta.PLUS_DI(df['high'], df['low'], df['close'], **self.configs)
        minus_di = ta.MINUS_DI(df['high'], df['low'], df['close'], **self.configs)
        df['linear_reg'] = adx
        df['linear_reg_angle'] = plus_di - minus_di
        return df


class MACD(Indicator):
    """ MACD indicator, default settings: fastperiod: 12, slowperiod: 26, signalperiod: 9 """
    name = 'MACD'

    def __init__(self, ttype: str, configs: dict):
        super(MACD, self).__init__(ttype, configs)

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args) -> pd.DataFrame:
        macd, macdsignal, macdhist = ta.MACD(df['close'], **self.configs)
        df['macd'] = macd
        df['macdsignal'] = macdsignal
        df['macdhist'] = macdhist  # macd - macdsignal
        # add auxilary data
        df['macd_dir'] = df['macd'].pct_change().rolling(3).mean()
        df['macdsignal_dir'] = df['macdsignal'].pct_change().rolling(3).mean()
        # don't consider low diff speed macd signals
        df.loc[(df['macd_dir'] > -0.1) & (df['macd_dir'] < 0.1), 'macd_dir'] = 0
        return df


class ATR(Indicator):
    """ ATR indicator, default settings: timeperiod: 24 """
    name = 'ATR'

    def __init__(self, ttype: str, configs: dict):
        super(ATR, self).__init__(ttype, configs)

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args) -> pd.DataFrame:
        atr = ta.ATR(df['high'], df['low'], df['close'], **self.configs)
        df['atr'] = atr
        return df


class PivotPoints(Indicator):
    """ Pivot point indicator """
    name = 'ATR'

    def __init__(self, ttype: str, configs: dict):
        super(PivotPoints, self).__init__(ttype, configs)

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args) -> pd.DataFrame:
        """ Add PP indicator to the dataframe """
        PP = pd.Series((df['high'] + df['low'] + df['close']) / 3)
        R1 = pd.Series(2 * PP - df['low'])
        S1 = pd.Series(2 * PP - df['high'])
        R2 = pd.Series(PP + df['high'] - df['low'])
        S2 = pd.Series(PP - df['high'] + df['low'])
        R3 = pd.Series(df['high'] + 2 * (PP - df['low']))
        S3 = pd.Series(df['low'] - 2 * (df['high'] - PP))
        psr = {'PP': PP, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}
        PSR = pd.DataFrame(psr)
        df = df.join(PSR)
        return df


class PriceChange(Indicator):
    """ Find big changes of price in both directions """
    name = 'PriceChange'

    def __init__(self, ttype: str, configs: dict):
        super(PriceChange, self).__init__(ttype, configs)
        self.low_price_quantile = self.configs.get('low_price_quantile', 5)
        self.high_price_quantile = self.configs.get('high_price_quantile', 95)
        self.max_stat_size = self.configs.get('max_stat_size', 100000)
        self.round_decimals = self.configs.get('decimals', 6)
        self.stat_file_path = self.configs.get('stat_file_path')
        self.price_stat = {
                           'lag1': np.array([]),
                           'lag2': np.array([]),
                           'lag3': np.array([])
                           }
        self.price_tmp_stat = {
            'lag1': np.array([]),
            'lag2': np.array([]),
            'lag3': np.array([])
        }
        # self.get_price_stat()

    def get_price_stat(self) -> None:
        """ Load price statistics from file """
        try:
            with open(self.stat_file_path, 'r') as f:
                self.price_stat = json.load(f)
        except FileNotFoundError:
            pass
        else:
            for key in self.price_stat.keys():
                del self.price_stat[key]['NaN']
                self.price_stat[key] = {float(k): int(v) for k, v in self.price_stat[key].items()}
                self.price_stat[key] = Counter(self.price_stat[key])

    def save_price_stat(self) -> None:
        """ Save price statistics to file """
        with open(self.stat_file_path, 'w+') as f:
            json.dump(self.price_stat, f)

    def get_price_change(self, df: pd.DataFrame, data_qty: int, lag: int) -> list:
        """ Get difference between current price and previous price """
        close_prices = (df['close'] - df['close'].shift(lag)) / df['close'].shift(lag)
        df[f'close_price_change_lag_{lag}'] = np.round(close_prices.values, self.round_decimals)
        return df[f'close_price_change_lag_{lag}'][max(df.shape[0] - data_qty + 1, 0):].values

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args) -> pd.DataFrame:
        """ Measure degree of ticker price change """
        for i in range(1, 2):
            # get statistics
            close_prices = self.get_price_change(df, data_qty, lag=i)
            # add price statistics, if statistics size is enough - add data to temp file to prevent high load of CPU
            if len(self.price_stat[f'lag{i}']) < self.max_stat_size:
                self.price_stat[f'lag{i}'] = np.append(self.price_stat[f'lag{i}'], close_prices)
                # delete NaNs
                self.price_stat[f'lag{i}'] = self.price_stat[f'lag{i}'][~np.isnan(self.price_stat[f'lag{i}'])]
                # sort price values
                self.price_stat[f'lag{i}'] = np.sort(self.price_stat[f'lag{i}'])
            else:
                self.price_tmp_stat[f'lag{i}'] = np.append(self.price_tmp_stat['lag1'], close_prices)
            # if we accumulate enough data - add them to our main statistics and prune it to reasonable size
            if len(self.price_tmp_stat[f'lag{i}']) > self.max_stat_size * 0.2:
                # add new data
                self.price_stat[f'lag{i}'] = np.append(self.price_stat[f'lag{i}'], self.price_tmp_stat[f'lag{i}'])
                self.price_tmp_stat[f'lag{i}'] = np.array([])
                # delete NaNs
                self.price_stat[f'lag{i}'] = self.price_stat[f'lag{i}'][~np.isnan(self.price_stat[f'lag{i}'])]
                # prune statistics
                indices = np.where(self.price_stat[f'lag{i}'])[0]
                to_replace = np.random.permutation(indices)[:self.max_stat_size]
                self.price_stat[f'lag{i}'] = self.price_stat[f'lag{i}'][to_replace]
                # sort price values
                self.price_stat[f'lag{i}'] = np.sort(self.price_stat[f'lag{i}'])
            # get lag quantiles and save to the dataframe
            q_low_lag = np.quantile(self.price_stat[f'lag{i}'], self.low_price_quantile / 1000)
            q_high_lag = np.quantile(self.price_stat[f'lag{i}'], self.high_price_quantile / 1000)
            df[[f'q_low_lag_{i}', f'q_high_lag_{i}']] = q_low_lag, q_high_lag
        # save price statistics to the file
        # self.save_price_stat()
        return df


class HighVolume(Indicator):
    """ Find a high volume """
    name = 'HighVolume'

    def __init__(self, ttype: str, configs: dict):
        super(HighVolume, self).__init__(ttype, configs)
        self.high_volume_quantile = self.configs.get('high_volume_quantile', 995)
        self.round_decimals = self.configs.get('round_decimals', 6)
        self.max_stat_size = self.configs.get('self.max_stat_size', 100000)
        self.vol_stat_file_path = self.configs.get('vol_stat_file_path')
        self.vol_stat = np.array([])
        self.vol_tmp_stat = np.array([])
        # self.get_vol_stat()

    def get_vol_stat(self) -> None:
        """ Load volume statistics from file """
        try:
            self.vol_stat = np.load(self.vol_stat_file_path)
        except FileNotFoundError:
            pass

    def save_vol_stat(self) -> None:
        """ Save volume statistics to file """
        np.save(self.vol_stat_file_path)

    def get_volume(self, df: pd.DataFrame, data_qty: int) -> list:
        """ Get MinMax normalized volume """
        normalized_vol = df['volume'] / df['volume'].sum()
        df[f'normalized_vol'] = np.round(normalized_vol.values, self.round_decimals)
        return df[f'normalized_vol'][max(df.shape[0] - data_qty + 1, 0):].values

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args) -> pd.DataFrame:
        """ Measure degree of ticker volume change """
        # get frequency counter
        vol = self.get_volume(df, data_qty)
        # add price statistics, if statistics size is enough - add data to temp file to prevent high load of CPU
        if len(self.vol_stat) < self.max_stat_size:
            self.vol_stat = np.append(self.vol_stat, vol)
            # delete NaNs and small values
            self.vol_stat = self.vol_stat[~np.isnan(self.vol_stat)]
            # sort price values
            self.vol_stat = np.sort(self.vol_stat)
        else:
            self.vol_tmp_stat = np.append(self.vol_tmp_stat, vol)
        # if we accumulate enough data - add them to our main statistics and prune it to reasonable size
        if len(self.vol_tmp_stat) > self.max_stat_size * 0.2:
            # add new data
            self.vol_stat = np.append(self.vol_stat, self.vol_tmp_stat)
            self.vol_tmp_stat = np.array([])
            # delete NaNs
            self.vol_stat = self.vol_stat[~np.isnan(self.vol_stat)]
            # prune statistics
            indices = np.where(self.vol_stat)[0]
            to_replace = np.random.permutation(indices)[:self.max_stat_size]
            self.vol_stat = self.vol_stat[to_replace]
            # sort price values
            self.vol_stat = np.sort(self.vol_stat)
        # get volume quantile and save to the dataframe
        quantile_vol = np.quantile(self.vol_stat, self.high_volume_quantile / 1000)
        df['quantile_vol'] = quantile_vol
        # save volume statistics to file
        # self.save_vol_stat()
        return df


class Pattern(Indicator):
    """ Find the minimum and maximum extremums """
    name = 'Pattern'

    def __init__(self, ttype: str, configs: dict):
        super(Pattern, self).__init__(ttype, configs)
        # number of last candle's extremums that will be updated (to increase performance) on second bot cycle and after
        self.last_candles_ext_num = self.configs.get('number_last_ext', 100)

    @staticmethod
    def get_high_max(df: pd.DataFrame) -> pd.Series:
        """ Get maximum high prices """
        df_high = df['high']
        high_max = df_high[(df_high >= df_high.shift(1)) &
                           (df_high >= df_high.shift(2)) &
                           (df_high >= df_high.shift(-1)) &
                           (df_high >= df_high.shift(-2))]
        return high_max

    @staticmethod
    def get_low_min(df: pd.DataFrame) -> pd.Series:
        """ Get minimum low prices """
        df_low = df['low']
        low_min = df_low[(df_low <= df_low.shift(1)) &
                         (df_low <= df_low.shift(2)) &
                         (df_low <= df_low.shift(-1)) &
                         (df_low <= df_low.shift(-2))]
        return low_min

    def shrink_max_min(self, df: pd.DataFrame, high_max: pd.DataFrame.index,
                       low_min: pd.DataFrame.index) -> (np.ndarray, np.ndarray):
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

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args) -> pd.DataFrame:
        """ Get main minimum and maximum extremums """
        high_max = self.get_high_max(df).index
        low_min = self.get_low_min(df).index
        df_len = df.shape[0]
        # on second and next cycles don't update all maximums and minimums to increase performance
        if data_qty <= 100:
            high_max = high_max[high_max >= df_len - self.last_candles_ext_num]
            low_min = low_min[low_min >= df_len - self.last_candles_ext_num]
        # leave only global maximums and minimums
        try:
            high_max, low_min = self.shrink_max_min(df, high_max, low_min)
        except IndexError:
            high_max, low_min = [], []
        df['high_max'] = 0
        df.loc[high_max, 'high_max'] = 1
        df['low_min'] = 0
        df.loc[low_min, 'low_min'] = 1
        return df
