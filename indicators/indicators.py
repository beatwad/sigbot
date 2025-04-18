"""
This module provides various technical indicators for analyzing financial data.
It defines an abstract base class `Indicator` and multiple concrete classes
representing specific indicators such as RSI, MACD, STOCH, ATR, and others.
Each indicator class is designed to calculate and append its respective indicator
values to a given DataFrame of financial data. The `IndicatorFactory` class is
responsible for dynamically selecting and returning the appropriate indicator
class based on the provided input.
"""

import json
import warnings
from abc import abstractmethod
from collections import Counter

import numpy as np
import pandas as pd
import talib as ta
from scipy.signal import argrelmax, argrelmin

warnings.simplefilter(action="ignore", category=FutureWarning)


class IndicatorFactory:
    """Factory class to return specific indicator object based on input value."""

    @staticmethod
    def factory(indicator, ttype, configs):
        """
        Factory method to select the appropriate indicator class.

        Parameters
        ----------
        indicator : str
            Name of the indicator to return.
        ttype : str
            Type of the trade ('buy', 'sell').
        configs : dict
            Configuration settings for the indicators.

        Returns
        -------
        Indicator
            The corresponding indicator object.
        """
        if indicator.startswith("RSI"):
            return RSI(ttype, configs)
        if indicator.startswith("STOCH"):
            return STOCH(ttype, configs)
        if indicator.startswith("MACD"):
            return MACD(ttype, configs)
        if indicator.startswith("PumpDump"):
            return PumpDump(ttype, configs)
        if indicator.startswith("Trend"):
            return Trend(ttype, configs)
        if indicator.startswith("HighVolume"):
            return HighVolume(ttype, configs)
        if indicator.startswith("Pattern"):
            return Pattern(ttype, configs)
        if indicator.startswith("ATR"):
            return ATR(ttype, configs)
        if indicator.startswith("SMA"):
            return SMA(ttype, configs)
        if indicator.startswith("Volume24"):
            return Volume24(ttype, configs)
        if indicator.startswith("CCI"):
            return CCI(ttype, configs)
        return SAR(ttype, configs)


class Indicator:
    """Abstract base class for indicators"""

    type = "Indicator"
    name = "Base"

    def __init__(self, ttype, configs):
        """Initialize the Indicator class.
        Parameters
        ----------
        ttype : str
            Type of the trade ('buy', 'sell').
        configs : dict
            Configuration settings for the indicators.
        """
        self.ttype = ttype
        self.configs = configs[self.type][self.ttype][self.name]["params"]

    @abstractmethod
    def get_indicator(
        self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args
    ) -> pd.DataFrame:
        """Get indicator data and write it to the dataframe"""
        return pd.DataFrame()


class RSI(Indicator):
    """RSI indicator, default settings: timeperiod: 14"""

    name = "RSI"

    def get_indicator(self, df, ticker: str, timeframe: str, data_qty: int, *args) -> pd.DataFrame:
        """
        Calculate RSI indicator and append to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        ticker : str
            Ticker symbol.
        timeframe : str
            Timeframe for data (e.g., 5m, 1H, 4H, 1D).
        data_qty : int
            The number of the most recent data to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with calculated RSI values.
        """
        # if mean close price value is too small, RSI indicator can become zero,
        # so we should increase it to at least 1e-4
        try:
            if df["close"].mean() < 1e-4:
                multiplier = int(1e-4 / df["close"].mean()) + 1
                rsi = ta.RSI(df["close"] * multiplier, **self.configs)
            else:
                rsi = ta.RSI(df["close"], **self.configs)
        except BaseException:  # noqa
            rsi = 0
        df["rsi"] = rsi
        return df


class STOCH(Indicator):
    """
    STOCH indicator, default settings:
    fastk_period: 14, slowk_period: 3, slowd_period:  3
    """

    name = "STOCH"

    def get_indicator(
        self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args
    ) -> pd.DataFrame:
        """
        Calculate STOCH indicator and append to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        ticker : str
            Ticker symbol.
        timeframe : str
            Timeframe for data (e.g., 5m, 1H, 4H, 1D).
        data_qty : int
            The number of the most recent data to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with calculated STOCH values.
        """
        try:
            slowk, slowd = ta.STOCH(df["high"], df["low"], df["close"], **self.configs)
        except BaseException:  # noqa
            slowk, slowd = 0, 0
        df["stoch_slowk"] = slowk
        df["stoch_slowd"] = slowd
        # add auxilary data
        df["stoch_slowk_dir"] = df["stoch_slowk"].pct_change().rolling(3).mean()
        df["stoch_slowd_dir"] = df["stoch_slowd"].pct_change().rolling(3).mean()
        df["stoch_diff"] = df["stoch_slowk"] - df["stoch_slowd"]
        df["stoch_diff"] = df["stoch_diff"].rolling(3).mean()
        return df


class Trend(Indicator):
    """
    Indicator of linear regression and its angle indicators,
    default settings: timeperiod: 14
    """

    name = "Trend"

    def get_indicator(
        self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args
    ) -> pd.DataFrame:
        """
        Calculate Trend indicator and append to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        ticker : str
            Ticker symbol.
        timeframe : str
            Timeframe for data (e.g., 5m, 1H, 4H, 1D).
        data_qty : int
            The number of the most recent data to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with calculated STOCH values.
        """
        try:
            adx = ta.ADX(df["high"], df["low"], df["close"], **self.configs)
            plus_di = ta.PLUS_DI(df["high"], df["low"], df["close"], **self.configs)
            minus_di = ta.MINUS_DI(df["high"], df["low"], df["close"], **self.configs)
        except BaseException:  # noqa
            adx, plus_di, minus_di = 0, 0, 0
        df["linear_reg"] = adx
        df["linear_reg_angle"] = plus_di - minus_di
        return df


class MACD(Indicator):
    """
    MACD indicator, default settings:
    fastperiod: 12, slowperiod: 26, signalperiod: 9
    """

    name = "MACD"

    def get_indicator(
        self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args
    ) -> pd.DataFrame:
        """
        Calculate MACD indicator and append to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        ticker : str
            Ticker symbol.
        timeframe : str
            Timeframe for data (e.g., 5m, 1H, 4H, 1D).
        data_qty : int
            The number of the most recent data to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with calculated STOCH values.
        """
        try:
            macd, macdsignal, macdhist = ta.MACD(df["close"], **self.configs)
        except BaseException:  # noqa
            macd, macdsignal, macdhist = 0, 0, 0
        df["macd"] = macd
        df["macdsignal"] = macdsignal
        df["macdhist"] = macdhist  # macd - macdsignal
        # add auxilary data
        df["macd_dir"] = df["macd"].pct_change().rolling(3).mean()
        df["macdsignal_dir"] = df["macdsignal"].pct_change().rolling(3).mean()
        # don't consider low diff speed macd signals
        df.loc[(df["macd_dir"] > -0.1) & (df["macd_dir"] < 0.1), "macd_dir"] = 0
        return df


class ATR(Indicator):
    """Average True Range indicator"""

    name = "ATR"

    def get_indicator(
        self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args
    ) -> pd.DataFrame:
        """
        Calculate ATR indicator and append to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        ticker : str
            Ticker symbol.
        timeframe : str
            Timeframe for data (e.g., 5m, 1H, 4H, 1D).
        data_qty : int
            The number of the most recent data to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with calculated STOCH values.
        """
        try:
            atr = ta.ATR(df["high"], df["low"], df["close"], **self.configs)
        except BaseException:  # noqa
            atr = 0
        df["atr"] = atr
        df["close_smooth"] = df["close"].rolling(self.configs["timeperiod"]).mean()
        return df


class SMA(Indicator):
    """
    Simple Moving Average indicator,
    default settings: timeperiod: 24
    """

    name = "SMA"

    def get_indicator(
        self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args
    ) -> pd.DataFrame:
        """
        Calculate SMA indicator and append to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        ticker : str
            Ticker symbol.
        timeframe : str
            Timeframe for data (e.g., 5m, 1H, 4H, 1D).
        data_qty : int
            The number of the most recent data to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with calculated STOCH values.
        """
        timeperiod = self.configs["timeperiod"]
        try:
            sma = ta.SMA(df["close"], timeperiod)
            sma_7 = ta.SMA(df["close"], timeperiod * 7)
        except BaseException:  # noqa
            sma = 0
            sma_7 = 0
        df["sma"] = sma
        df["sma_7"] = sma_7
        return df


class CCI(Indicator):
    """
    Commodity Channel Index (Momentum Indicators),
    default settings: timeperiod: 24
    """

    name = "CCI"

    def get_indicator(
        self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args
    ) -> pd.DataFrame:
        """
        Calculate CCI indicator and append to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        ticker : str
            Ticker symbol.
        timeframe : str
            Timeframe for data (e.g., 5m, 1H, 4H, 1D).
        data_qty : int
            The number of the most recent data to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with calculated STOCH values.
        """
        timeperiod = self.configs["timeperiod"]
        try:
            cci = ta.CCI(df["high"], df["low"], df["close"], timeperiod)
        except BaseException:  # noqa
            cci = 0
        df["cci"] = cci
        return df


class SAR(Indicator):
    """
    Parabolic Stop and Reverse indicator,
    default settings: acceleration: 0.02, maximum: 0.2
    """

    name = "SAR"

    def get_indicator(
        self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args
    ) -> pd.DataFrame:
        """
        Calculate SAR indicator and append to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        ticker : str
            Ticker symbol.
        timeframe : str
            Timeframe for data (e.g., 5m, 1H, 4H, 1D).
        data_qty : int
            The number of the most recent data to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with calculated STOCH values.
        """
        acceleration = self.configs["acceleration"]
        maximum = self.configs["maximum"]
        try:
            sar = ta.SAR(df["high"], df["low"], acceleration, maximum)
        except BaseException:  # noqa
            sar = 0
        df["sar"] = sar
        return df


class PumpDump(Indicator):
    """Find big changes of price in both directions"""

    name = "PumpDump"

    def __init__(self, ttype: str, configs: dict):
        super().__init__(ttype, configs)
        self.low_price_quantile = self.configs.get("low_price_quantile", 5)
        self.high_price_quantile = self.configs.get("high_price_quantile", 95)
        self.max_stat_size = self.configs.get("max_stat_size", 100000)
        self.round_decimals = self.configs.get("decimals", 6)
        self.stat_file_path = self.configs.get("stat_file_path")
        self.price_stat = {
            "lag1": np.array([]),
            "lag2": np.array([]),
            "lag3": np.array([]),
        }
        self.price_tmp_stat = {
            "lag1": np.array([]),
            "lag2": np.array([]),
            "lag3": np.array([]),
        }
        # self.get_price_stat()

    def get_price_stat(self) -> None:
        """
        Load price statistics from file.
        This method attempts to read price statistics from a JSON file.
        If the file is not found, it will silently pass.
        Otherwise, it updates the statistics.
        """
        try:
            with open(self.stat_file_path, "r") as f:
                self.price_stat = json.load(f)
        except FileNotFoundError:
            pass
        else:
            for key in self.price_stat.keys():
                del self.price_stat[key]["NaN"]
                self.price_stat[key] = {float(k): int(v) for k, v in self.price_stat[key].items()}
                self.price_stat[key] = Counter(self.price_stat[key])

    def save_price_stat(self) -> None:
        """
        Save price statistics to file.

        This method writes the current price statistics to a
        JSON file specified by the stat_file_path attribute.
        """
        with open(self.stat_file_path, "w+") as f:
            json.dump(self.price_stat, f)

    def get_price_change(self, df: pd.DataFrame, data_qty: int, lag: int) -> list:
        """
        Get difference between current price and previous price.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        data_qty : int
            The number of the most recent data to return.
        lag : int
            The number of periods to look back for price changes.

        Returns
        -------
        list
            A list of rounded price changes.
        """
        close_prices = (df["close"] - df["close"].shift(lag)) / df["close"].shift(lag)
        df[f"price_change_{lag}"] = np.round(close_prices.values, self.round_decimals)
        return df[f"price_change_{lag}"][max(df.shape[0] - data_qty + 1, 0) :].values  # noqa

    def get_indicator(
        self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args
    ) -> pd.DataFrame:
        """
        Measure degree of ticker price change.

        This method calculates price changes, updates the price statistics, and computes
        the quantiles for the specified lag periods.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        ticker : str
            The ticker symbol of the asset.
        timeframe : str
            Timeframe for data (e.g., 5m, 1H, 4H, 1D).
        data_qty : int
            The number of the most recent data to return.
        args : tuple
            Additional arguments (currently unused).

        Returns
        -------
        pd.DataFrame
            The updated DataFrame with quantile values for the specified lags.
        """
        for i in range(1, 2):
            # get statistics
            close_prices = self.get_price_change(df, data_qty, lag=i)
            # add price statistics, if statistics size is enough -
            # add data to temp file to prevent high load of CPU
            if len(self.price_stat[f"lag{i}"]) < self.max_stat_size:
                self.price_stat[f"lag{i}"] = np.append(self.price_stat[f"lag{i}"], close_prices)
                # delete NaNs
                self.price_stat[f"lag{i}"] = self.price_stat[f"lag{i}"][
                    ~np.isnan(self.price_stat[f"lag{i}"])
                ]
                # sort price values
                self.price_stat[f"lag{i}"] = np.sort(self.price_stat[f"lag{i}"])
            else:
                self.price_tmp_stat[f"lag{i}"] = np.append(
                    self.price_tmp_stat[f"lag{i}"], close_prices
                )
            # if we accumulate enough data -
            # add them to our main statistics and prune it to reasonable size
            if len(self.price_tmp_stat[f"lag{i}"]) > self.max_stat_size * 0.2:
                # add new data
                self.price_stat[f"lag{i}"] = np.append(
                    self.price_stat[f"lag{i}"], self.price_tmp_stat[f"lag{i}"]
                )
                self.price_tmp_stat[f"lag{i}"] = np.array([])
                # delete NaNs
                self.price_stat[f"lag{i}"] = self.price_stat[f"lag{i}"][
                    ~np.isnan(self.price_stat[f"lag{i}"])
                ]
                # prune statistics
                indices = np.where(self.price_stat[f"lag{i}"])[0]
                to_replace = np.random.permutation(indices)[: self.max_stat_size]
                self.price_stat[f"lag{i}"] = self.price_stat[f"lag{i}"][to_replace]
                # sort price values
                self.price_stat[f"lag{i}"] = np.sort(self.price_stat[f"lag{i}"])
            # get lag quantiles and save to the dataframe
            q_low_lag = np.quantile(self.price_stat[f"lag{i}"], self.low_price_quantile / 1000)
            q_high_lag = np.quantile(self.price_stat[f"lag{i}"], self.high_price_quantile / 1000)
            df[[f"q_low_lag_{i}", f"q_high_lag_{i}"]] = q_low_lag, q_high_lag
        # save price statistics to the file
        # self.save_price_stat()
        return df


class HighVolume(Indicator):
    """Find a high volume"""

    name = "HighVolume"

    def __init__(self, ttype: str, configs: dict):
        super().__init__(ttype, configs)
        self.high_volume_quantile = self.configs.get("high_volume_quantile", 995)
        self.round_decimals = self.configs.get("round_decimals", 6)
        self.max_stat_size = self.configs.get("self.max_stat_size", 100000)
        self.vol_stat_file_path = self.configs.get("vol_stat_file_path")
        self.vol_stat = np.array([])
        self.vol_tmp_stat = np.array([])
        # self.get_vol_stat()

    def get_vol_stat(self) -> None:
        """
        Load volume statistics from file.

        This method attempts to read volume statistics from a NumPy binary file.
        If the file is not found, it will silently pass.
        """
        try:
            self.vol_stat = np.load(self.vol_stat_file_path)
        except FileNotFoundError:
            pass

    def save_vol_stat(self, quantile_vol: np.array) -> None:
        """
        Save volume statistics to file.

        Parameters
        ----------
        quantile_vol : np.array
            The quantile volume statistics to save.
        """
        np.save(self.vol_stat_file_path, quantile_vol)

    def get_volume(self, df: pd.DataFrame, data_qty: int) -> list:
        """
        Get MinMax normalized volume.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        data_qty : int
            The number of the most recent data to return.

        Returns
        -------
        list
            A list of normalized volumes.
        """
        normalized_vol = df["volume"] / df["volume"].sum()
        df["normalized_vol"] = np.round(normalized_vol.values, self.round_decimals)
        return df["normalized_vol"][max(df.shape[0] - data_qty + 1, 0) :].values  # noqa

    def get_indicator(
        self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args
    ) -> pd.DataFrame:
        """
        Measure degree of ticker volume change.

        This method calculates normalized volumes,
        updates the volume statistics, and computes
        the quantile for high volume.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        ticker : str
            Ticker symbol.
        timeframe : str
            Timeframe for data (e.g., 5m, 1H, 4H, 1D).
        data_qty : int
            The number of the most recent data to return.
        args : tuple
            Additional arguments (currently unused).

        Returns
        -------
        pd.DataFrame
            The updated DataFrame with quantile volume value.
        """
        # get frequency counter
        vol = self.get_volume(df, data_qty)
        # add price statistics, if statistics size is enough -
        # add data to temp file to prevent high load of CPU
        if len(self.vol_stat) < self.max_stat_size:
            self.vol_stat = np.append(self.vol_stat, vol)
            # delete NaNs and small values
            self.vol_stat = self.vol_stat[~np.isnan(self.vol_stat)]
            # sort price values
            self.vol_stat = np.sort(self.vol_stat)
        else:
            self.vol_tmp_stat = np.append(self.vol_tmp_stat, vol)
        # if we accumulate enough data -
        # add them to our main statistics and prune it to reasonable size
        if len(self.vol_tmp_stat) > self.max_stat_size * 0.2:
            # add new data
            self.vol_stat = np.append(self.vol_stat, self.vol_tmp_stat)
            self.vol_tmp_stat = np.array([])
            # delete NaNs
            self.vol_stat = self.vol_stat[~np.isnan(self.vol_stat)]
            # prune statistics
            indices = np.where(self.vol_stat)[0]
            to_replace = np.random.permutation(indices)[: self.max_stat_size]
            self.vol_stat = self.vol_stat[to_replace]
            # sort price values
            self.vol_stat = np.sort(self.vol_stat)
        # get volume quantile and save to the dataframe
        quantile_vol = np.quantile(self.vol_stat, self.high_volume_quantile / 1000)
        df["quantile_vol"] = quantile_vol
        # save volume statistics to file
        # self.save_vol_stat(quantile_vol)
        return df


class Pattern(Indicator):
    """Find the minimum and maximum extremums"""

    name = "Pattern"

    def __init__(self, ttype: str, configs: dict):
        super().__init__(ttype, configs)
        # number of last candle's extremums that will be updated
        # (to increase performance) on second bot cycle and after
        self.last_candles_ext_num = self.configs.get("number_last_ext", 100)

    def get_indicator(
        self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args
    ) -> pd.DataFrame:
        """
        Get main minimum and maximum extremes.

        This method identifies the relative maxima and minima
        in the price data and updates the DataFrame with these
        extreme points.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        ticker : str
            Ticker symbol.
        timeframe : str
            Timeframe for data (e.g., 5m, 1H, 4H, 1D).
        data_qty : int
            The number of the most recent data to return.
        args : tuple
            Additional arguments (currently unused).

        Returns
        -------
        pd.DataFrame
            The updated DataFrame with extremum indicators.
        """
        high_max = argrelmax(df["high"].values, order=4)[0]
        low_min = argrelmin(df["low"].values, order=4)[0]
        min_max_len = min(high_max.shape[0], low_min.shape[0])
        high_max = high_max[-min_max_len:]
        low_min = low_min[-min_max_len:]

        df["high_max"] = 0
        df.loc[high_max, "high_max"] = 1
        df["low_min"] = 0
        df.loc[low_min, "low_min"] = 1

        return df


class Volume24(Indicator):
    """Find ticker volume for the last 24 hours"""

    name = "Volume24"

    def __init__(self, ttype: str, configs: dict):
        super().__init__(ttype, configs)
        self.timeframe_div = configs["Data"]["Basic"]["params"]["timeframe_div"]

    def get_indicator(
        self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int, *args
    ) -> pd.DataFrame:
        """
        Calculate indicator of high volume for the last 24 hours
        and append to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with candlestick data.
        ticker : str
            Ticker symbol.
        timeframe : str
            Timeframe for data (e.g., 5m, 1H, 4H, 1D).
        data_qty : int
            The number of the most recent data to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with calculated STOCH values.
        """
        # get quantity of candles in 24 hours
        avg_period = int(24 / (self.timeframe_div[timeframe] / 3600))
        # get average volume for 24 hours
        df["volume_24"] = df["volume"].rolling(avg_period).sum()
        return df
