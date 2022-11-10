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
        slowk, slowd = ta.STOCH(df['high'], df['low'],
                                df['close'], **self.configs)
        df['stoch_slowk'] = slowk
        df['stoch_slowd'] = slowd
        return df


class LinearReg(Indicator):
    """ Indicator of linear regression and its angle indicators, default settings: timeperiod: 14 """
    name = 'LinearReg'

    def __init__(self, ttype: str, configs: dict):
        super(LinearReg, self).__init__(ttype, configs)

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args) -> pd.DataFrame:
        linear_reg = ta.LINEARREG(df['close'], **self.configs)
        linear_reg_angle = ta.LINEARREG_ANGLE(df['close'], **self.configs)
        df['linear_reg'] = linear_reg
        df['linear_reg_angle'] = linear_reg_angle
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
        df['macdhist'] = macdhist
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

    def get_volume(self, df: pd.DataFrame, data_qty: int, volume: int) -> list:
        """ Get MinMax normalized volume """
        normalized_vol = df['volume'] / volume
        df[f'normalized_vol'] = np.round(normalized_vol.values, self.round_decimals)
        return df[f'normalized_vol'][max(df.shape[0] - data_qty + 1, 0):].values

    def get_indicator(self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, volume: int) -> pd.DataFrame:
        """ Measure degree of ticker price change """
        # get frequency counter
        vol = self.get_volume(df, data_qty, volume)
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
