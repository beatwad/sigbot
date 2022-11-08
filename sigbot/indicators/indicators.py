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
        self.round_decimals = self.configs.get('decimals', 4)
        self.stat_file_path = self.configs.get('stat_file_path')
        self.price_stat = {'lag1': Counter(),
                           'lag2': Counter(),
                           'lag3': Counter()}
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
        # get frequency counter
        close_prices_lag_1 = self.get_price_change(df, data_qty, lag=1)
        close_prices_lag_2 = self.get_price_change(df, data_qty, lag=2)
        close_prices_lag_3 = self.get_price_change(df, data_qty, lag=3)
        self.price_stat['lag1'] += Counter(close_prices_lag_1)
        self.price_stat['lag2'] += Counter(close_prices_lag_2)
        self.price_stat['lag3'] += Counter(close_prices_lag_3)
        # get lag 1 quantiles and save to dataframe
        q_lag_1 = pd.Series(data=self.price_stat['lag1'].keys(), index=self.price_stat['lag1'].values())
        q_low_lag_1 = q_lag_1.sort_index().quantile(self.low_price_quantile / 1000)
        q_high_lag_1 = q_lag_1.sort_index().quantile(self.high_price_quantile / 1000)
        df[['q_low_lag_1', 'q_high_lag_1']] = q_low_lag_1, q_high_lag_1
        # get lag 2 quantiles and save to dataframe
        q_lag_2 = pd.Series(data=self.price_stat['lag2'].keys(), index=self.price_stat['lag2'].values())
        q_low_lag_2 = q_lag_2.sort_index().quantile(self.low_price_quantile / 1000)
        q_high_lag_2 = q_lag_2.sort_index().quantile(self.high_price_quantile / 1000)
        df[['q_low_lag_2', 'q_high_lag_2']] = q_low_lag_2, q_high_lag_2
        # get lag 3 quantiles and save to dataframe
        q_lag_3 = pd.Series(data=self.price_stat['lag3'].keys(), index=self.price_stat['lag3'].values())
        q_low_lag_3 = q_lag_3.sort_index().quantile(self.low_price_quantile / 1000)
        q_high_lag_3 = q_lag_3.sort_index().quantile(self.high_price_quantile / 1000)
        df[['q_low_lag_3', 'q_high_lag_3']] = q_low_lag_3, q_high_lag_3
        # save price statistics to file
        # self.save_price_stat()
        return df


class HighVolume(Indicator):
    """ Find a high volume """
    name = 'HighVolume'

    def __init__(self, ttype: str, configs: dict):
        super(HighVolume, self).__init__(ttype, configs)
        self.high_volume_quantile = self.configs.get('high_volume_quantile', 990)
        self.round_decimals = self.configs.get('round_decimals', 4)
        self.vol_stat_file_path = self.configs.get('vol_stat_file_path')
        self.vol_stat = np.array([])
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
        self.vol_stat = np.append(self.vol_stat, vol)
        self.vol_stat = np.unique(self.vol_stat)
        np.sort(self.vol_stat)
        # get quantiles and save to dataframe
        quantile_vol = np.quantile(self.vol_stat, self.high_volume_quantile / 1000)
        df['quantile_vol'] = quantile_vol
        # save volume statistics to file
        # self.save_vol_stat()
        return df
