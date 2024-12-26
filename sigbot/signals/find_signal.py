"""
This module provides a functionality for searching of the
technical indicator combination (e.g. RSI + Stochastic, MACD + Volume, etc.)
or technical analysys patterns
(e.g. Upper Triangle, Double Bottom, Head & Shoulders, etc.)
"""

from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd


class SignalBase:
    """
    Base signal searching class

    Attributes
    ----------
    ttype : str
        Trade type ('buy' or 'sell').
    configs : dict
        Dictionary with parameters for the Trend indicator.
    """

    type = "Indicator_signal"
    name = "Base"

    def __init__(self, ttype, configs):
        self.ttype = ttype
        self.configs = configs[self.type][self.ttype]
        self.vol_q_high = 0
        self.vol_q_low = 0
        self.vol_window = 0
        self.first_candle = 0
        self.second_candle = 0

    @abstractmethod
    def find_signal(self, df: pd.DataFrame, *args):
        """
        Abstract method for signal detection,
        to be implemented by derived classes.
        """
        return False, "", []

    @staticmethod
    def lower_bound(
        low_bound: float,
        indicator: pd.Series,
        indicator_lag_1: pd.Series,
        indicator_lag_2: pd.Series,
    ) -> np.ndarray:
        """
        Returns an array indicating if at least two of the last
        three points of the indicator are below the given low bound.

        Parameters
        ----------
        low_bound : float
            The lower bound threshold.
        indicator : pd.Series
            The current indicator series.
        indicator_lag_1 : pd.Series
            The indicator series lagged by one time step.
        indicator_lag_2 : pd.Series
            The indicator series lagged by two time steps.

        Returns
        -------
        np.ndarray
            An array indicating where the condition is met.
        """
        indicator = np.where(indicator < low_bound, 1, 0)
        indicator_lag_1 = np.where(indicator_lag_1 < low_bound, 1, 0)
        indicator_lag_2 = np.where(indicator_lag_2 < low_bound, 1, 0)
        indicator_sum = np.array([indicator, indicator_lag_1, indicator_lag_2]).sum(
            axis=0
        )
        return np.where(indicator_sum >= 2, 1, 0)

    @staticmethod
    def higher_bound(
        high_bound: float,
        indicator: pd.Series,
        indicator_lag_1: pd.Series,
        indicator_lag_2: pd.Series,
    ) -> np.ndarray:
        """
        Returns an array indicating if at least two of the last
        three points of the indicator are above the given low bound.

        Parameters
        ----------
        high_bound : float
            The lower bound threshold.
        indicator : pd.Series
            The current indicator series.
        indicator_lag_1 : pd.Series
            The indicator series lagged by one time step.
        indicator_lag_2 : pd.Series
            The indicator series lagged by two time steps.

        Returns
        -------
        np.ndarray
            An array indicating where the condition is met.
        """
        indicator = np.where(indicator > high_bound, 1, 0)
        indicator_lag_1 = np.where(indicator_lag_1 > high_bound, 1, 0)
        indicator_lag_2 = np.where(indicator_lag_2 > high_bound, 1, 0)
        indicator_sum = np.array([indicator, indicator_lag_1, indicator_lag_2]).sum(
            axis=0
        )
        return np.where(indicator_sum >= 2, 1, 0)

    @staticmethod
    def crossed_lines(
        up: bool,
        indicator: pd.Series,
        indicator_lag_1: pd.Series,
        indicator_lag_2: pd.Series,
    ) -> np.ndarray:
        """
        Returns an array indicating if slowk and slowd lines
        have crossed based on the given direction (up or down).

        Parameters
        ----------
        up : bool
            Direction of the crossing; True for upward cross,
            False for downward cross.
        indicator : pd.Series
            The current indicator series.
        indicator_lag_1 : pd.Series
            The indicator series lagged by one time step.
        indicator_lag_2 : pd.Series
            The indicator series lagged by two time steps.

        Returns
        -------
        np.ndarray
            An array indicating whether the lines have crossed.
        """
        if up:
            indicator = np.where(indicator > 0, 1, 0)
            indicator_lag_1 = np.where(indicator_lag_1 < 0, 1, 0)
            indicator_lag_1 = np.array([indicator, indicator_lag_1]).sum(axis=0)
            indicator_lag_2 = np.where(indicator_lag_2 < 0, 1, 0)
            indicator_lag_2 = np.array([indicator, indicator_lag_2]).sum(axis=0)
        else:
            indicator = np.where(indicator < 0, 1, 0)
            indicator_lag_1 = np.where(indicator_lag_1 > 0, 1, 0)
            indicator_lag_1 = np.array([indicator, indicator_lag_1]).sum(axis=0)
            indicator_lag_2 = np.where(indicator_lag_2 > 0, 1, 0)
            indicator_lag_2 = np.array([indicator, indicator_lag_2]).sum(axis=0)
        indicator = np.maximum(indicator_lag_1, indicator_lag_2)
        return np.where(indicator > 1, 1, 0)

    @staticmethod
    def lower_bound_robust(low_bound: float, indicator: pd.Series) -> np.ndarray:
        """
        Returns an array indicating if the indicator is below the given low bound.

        Parameters
        ----------
        low_bound : float
            The lower bound threshold.
        indicator : pd.Series
            The current indicator series.

        Returns
        -------
        np.ndarray
            An array indicating where the condition is met.
        """
        return np.where(indicator < low_bound, 1, 0)

    @staticmethod
    def higher_bound_robust(high_bound: float, indicator: pd.Series) -> np.ndarray:
        """
        Returns an array indicating if the indicator is above the given high bound.

        Parameters
        ----------
        high_bound : float
            The upper bound threshold.
        indicator : pd.Series
            The current indicator series.

        Returns
        -------
        np.ndarray
            An array indicating where the condition is met.
        """
        return np.where(indicator > high_bound, 1, 0)

    @staticmethod
    def up_direction(indicator: pd.Series) -> np.ndarray:
        """
        Returns an array indicating if the indicator is moving in an upward direction.

        Parameters
        ----------
        indicator : pd.Series
            The indicator series.

        Returns
        -------
        np.ndarray
            An array indicating where upward movement happens.
        """
        return np.where(indicator > 0, 1, 0)

    @staticmethod
    def down_direction(indicator: pd.Series) -> np.ndarray:
        """
        Returns an array indicating if the indicator is moving in a downward direction.

        Parameters
        ----------
        indicator : pd.Series
            The indicator series.

        Returns
        -------
        np.ndarray
            An array indicating where downward movement happens.
        """
        return np.where(indicator < 0, 1, 0)

    def two_good_candles(self, df: pd.DataFrame, ttype: str) -> np.ndarray:
        """
        Returns an array indicating if two candles confirm a pattern movement
        and corresponding volume is high or low enough.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing candlestick and volume data.
        ttype : str
            Trade type ('buy' or 'sell').

        Returns
        -------
        np.ndarray
            An array indicating where two good candles pattern is present.
        """
        # use high/low volume to confirm pattern
        vol_q_high = df["volume"].rolling(self.vol_window).quantile(self.vol_q_high)
        vol_q_low = df["volume"].rolling(self.vol_window).quantile(self.vol_q_low)
        first_candle_vol = df["volume"].shift(1)
        second_candle_vol = df["volume"].shift(2)
        # find two candles
        if ttype == "buy":
            sign_1 = np.where(df["close"].shift(2) > df["open"].shift(2), 1, -1)
            sign_2 = np.where(df["close"].shift(1) > df["open"].shift(1), 1, -1)
            first_candle = (
                (df["close"].shift(2) - df["low"].shift(2))
                / (df["high"].shift(2) - df["low"].shift(2))
                * sign_1
            )
            second_candle = (
                (df["close"].shift(1) - df["low"].shift(1))
                / (df["high"].shift(1) - df["low"].shift(1))
                * sign_2
            )
        else:
            sign_1 = np.where(df["close"].shift(2) < df["open"].shift(2), 1, -1)
            sign_2 = np.where(df["close"].shift(1) < df["open"].shift(1), 1, -1)
            first_candle = (
                (df["high"].shift(2) - df["close"].shift(2))
                / (df["high"].shift(2) - df["low"].shift(2))
                * sign_1
            )
            second_candle = (
                (df["high"].shift(1) - df["close"].shift(1))
                / (df["high"].shift(1) - df["low"].shift(1))
                * sign_2
            )
        return np.where(
            (first_candle >= self.first_candle)
            & (
                ((first_candle_vol >= vol_q_high) & (second_candle_vol >= vol_q_high))
                | ((first_candle_vol <= vol_q_low) & (second_candle_vol <= vol_q_low))
            )
            & (second_candle >= self.second_candle),
            1,
            0,
        )


class SignalFactory:
    """
    Factory class to return the appropriate signal object
    based on the 'indicator' value.
    """

    @staticmethod
    def factory(indicator: str, ttype: str, configs: dict) -> SignalBase:
        """
        Factory method to create and return an instance of
        a signal class based on the indicator type.

        Parameters
        ----------
        indicator : str
            The type of indicator (e.g., 'RSI', 'STOCH').
        ttype : str
            Trade type ('buy', 'sell').
        configs : dict
            Dictionary with indicator parameters.

        Returns
        -------
        SignalBase
            An instance of the appropriate signal class.
        """
        if indicator == "RSI":
            return RSISignal(ttype, **configs)
        if indicator == "STOCH":
            return STOCHSignal(ttype, **configs)
        if indicator == "MACD":
            return MACDSignal(ttype, **configs)
        if indicator == "Pattern":
            return PatternSignal(ttype, **configs)
        if indicator == "PumpDump":
            return PumpDumpSignal(ttype, **configs)
        if indicator == "Trend":
            return TrendSignal(ttype, **configs)
        if indicator == "AntiTrend":
            return AntiTrendSignal(ttype, **configs)
        if indicator == "HighVolume":
            return HighVolumeSignal(ttype, **configs)
        return Volume24Signal(ttype, **configs)


class STOCHSignal(SignalBase):
    """
    Check if STOCH is in overbuy/oversell zone and
    is going to change its direction to opposite

    Attributes
    ----------
    ttype : str
        Trade type ('buy' or 'sell').
    configs : dict
        Dictionary with parameters for the Trend indicator.
    """

    type = "Indicator_signal"
    name = "STOCH"

    def __init__(self, ttype: str, **configs):
        """
        Initialize the STOCHSignal class with trade type and configuration parameters.

        Parameters
        ----------
        ttype : str
            Trade type ('buy' or 'sell').
        configs : dict
            Dictionary with parameters for the STOCH indicator.
        """
        super().__init__(ttype, configs)
        self.configs = self.configs[self.name]["params"]
        self.low_bound = self.configs.get("low_bound", 20)
        self.high_bound = self.configs.get("high_bound", 80)

    def find_signal(self, df: pd.DataFrame, *args) -> np.ndarray:
        """
        Find STOCH signal.
        Return true if Stochastic is lower than low bound, lines
        have crossed and look up (buy signal)
        Return true if Stochastic is higher than high bound, lines
        have crossed and look down (sell signal)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing STOCH indicator data.

        Returns
        -------
        np.ndarray
            Boolean array indicating the presence of a STOCH signal.
        """
        # Find STOCH signal
        stoch_slowk = df["stoch_slowk"]
        stoch_slowk_lag_1 = df["stoch_slowk"].shift(1)
        stoch_slowk_lag_2 = df["stoch_slowk"].shift(2)
        stoch_slowd = df["stoch_slowd"]
        stoch_slowd_lag_1 = df["stoch_slowd"].shift(1)
        stoch_slowd_lag_2 = df["stoch_slowd"].shift(2)
        # stoch_diff = df["stoch_diff"]
        # stoch_diff_lag_1 = df["stoch_diff"].shift(1)
        # stoch_diff_lag_2 = df["stoch_diff"].shift(2)

        if self.ttype == "buy":
            lower_bound_slowk = self.lower_bound(
                self.low_bound, stoch_slowk, stoch_slowk_lag_1, stoch_slowk_lag_2
            )
            lower_bound_slowd = self.lower_bound(
                self.low_bound, stoch_slowd, stoch_slowd_lag_1, stoch_slowd_lag_2
            )
            # crossed_lines_up = self.crossed_lines(
            #     True, stoch_diff, stoch_diff_lag_1, stoch_diff_lag_2
            # )
            # up_direction_slowk = self.up_direction(df["stoch_slowk_dir"])
            # up_direction_slowd = self.up_direction(df["stoch_slowd_dir"])
            down_direction_slowk = self.down_direction(df["stoch_slowk_dir"])
            down_direction_slowd = self.down_direction(df["stoch_slowd_dir"])
            stoch_up = (
                lower_bound_slowk
                & lower_bound_slowd
                & down_direction_slowk
                & down_direction_slowd
                # & crossed_lines_up
                # & up_direction_slowk
                # & up_direction_slowd
            )
            return stoch_up

        higher_bound_slowk = self.higher_bound(
            self.high_bound, stoch_slowk, stoch_slowk_lag_1, stoch_slowk_lag_2
        )
        higher_bound_slowd = self.higher_bound(
            self.high_bound, stoch_slowd, stoch_slowd_lag_1, stoch_slowd_lag_2
        )
        # crossed_lines_down = self.crossed_lines(
        #     False, stoch_diff, stoch_diff_lag_1, stoch_diff_lag_2
        # )
        # down_direction_slowk = self.down_direction(df["stoch_slowk_dir"])
        # down_direction_slowd = self.down_direction(df["stoch_slowd_dir"])
        up_direction_slowk = self.up_direction(df["stoch_slowk_dir"])
        up_direction_slowd = self.up_direction(df["stoch_slowd_dir"])
        stoch_down = (
            higher_bound_slowk
            & higher_bound_slowd
            & up_direction_slowk
            & up_direction_slowd
            # & crossed_lines_down
            # & down_direction_slowk
            # & down_direction_slowd
        )
        return stoch_down


class RSISignal(SignalBase):
    """
    Check if RSI is in overbuy/oversell zone

    Attributes
    ----------
    ttype : str
        Trade type ('buy' or 'sell').
    configs : dict
        Dictionary with parameters for the Trend indicator.
    """

    type = "Indicator_signal"
    name = "RSI"

    def __init__(self, ttype: str, **configs):
        super().__init__(ttype, configs)
        self.configs = self.configs[self.name]["params"]
        self.low_bound = self.configs.get("low_bound", 25)
        self.high_bound = self.configs.get("high_bound", 75)

    def find_signal(self, df: pd.DataFrame, *args) -> np.ndarray:
        """
        Find RSI signal.
        Return true if RSI is lower than low bound (buy signal)
        Return true if RSI is higher than low bound (sell signal)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing RSI indicator data.

        Returns
        -------
        np.ndarray
            Boolean array indicating the presence of a RSI signal.
        """
        # Find RSI signal
        rsi = df["rsi"]
        rsi_lag_1 = df["rsi"].shift(1)
        rsi_lag_2 = df["rsi"].shift(2)
        if self.ttype == "buy":
            rsi_lower = self.lower_bound(self.low_bound, rsi, rsi_lag_1, rsi_lag_2)
            return rsi_lower
        rsi_higher = self.higher_bound(self.high_bound, rsi, rsi_lag_1, rsi_lag_2)
        return rsi_higher


class TrendSignal(SignalBase):
    """
    Check trend using linear regression indicator

    Attributes
    ----------
    ttype : str
        Trade type ('buy' or 'sell').
    configs : dict
        Dictionary with parameters for the Trend indicator.
    """

    type = "Indicator_signal"
    name = "Trend"

    def __init__(self, ttype, **configs):
        super().__init__(ttype, configs)
        self.configs = self.configs[self.name]["params"]
        self.low_bound = self.configs.get("low_bound", 0)
        self.high_bound = self.configs.get("high_bound", 0)

    def find_signal(self, df: pd.DataFrame, *args) -> np.ndarray:
        """
        Find Trend signal.
        Return true if trend is moving up (buy signal)
        Return true if trend is moving down (sell signal)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing Trend indicator data.

        Returns
        -------
        np.ndarray
            Boolean array indicating the presence of a Trend signal.
        """
        # According to difference between working timeframe and higher timeframe
        # for every point on working timeframe find corresponding value of
        # Linear Regression from higher timeframe
        # buy trade
        if self.ttype == "buy":
            # find Linear Regression signal
            lr_higher_bound = self.higher_bound_robust(
                self.high_bound, df["linear_reg_angle"]
            )
            return lr_higher_bound

        # same for the sell trade
        # find Linear Regression signal
        lr_lower_bound = self.lower_bound_robust(self.low_bound, df["linear_reg_angle"])
        return lr_lower_bound


class AntiTrendSignal(SignalBase):
    """
    Check trend using linear regression indicator

    Attributes
    ----------
    ttype : str
        Trade type ('buy' or 'sell').
    configs : dict
        Configuration dictionary for the AntiTrend indicator.
    """

    type = "Indicator_signal"
    name = "AntiTrend"

    def __init__(self, ttype, **configs):
        super().__init__(ttype, configs)
        self.configs = self.configs[self.name]["params"]
        self.low_bound = self.configs.get("low_bound", 0)
        self.high_bound = self.configs.get("high_bound", 0)

    def find_signal(self, df: pd.DataFrame, *args) -> np.ndarray:
        """
        Find Trend signal.
        Return true if trend is moving down (buy signal)
        Return true if trend is moving up (sell signal)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing Trend indicator data.

        Returns
        -------
        np.ndarray
            Boolean array indicating the presence of an AntiTrend signal.
        """
        # According to difference between working timeframe and higher timeframe
        # for every point on working timeframe find corresponding value of
        # Linear Regression from higher timeframe
        # buy trade
        if self.ttype == "sell":
            # find Linear Regression signal
            lr_higher_bound = self.higher_bound_robust(
                self.high_bound, df["linear_reg_angle"]
            )
            return lr_higher_bound

        # same for the sell trade
        # find Linear Regression signal
        lr_lower_bound = self.lower_bound_robust(self.low_bound, df["linear_reg_angle"])
        return lr_lower_bound


class PumpDumpSignal(SignalBase):
    """
    Find situations when price
    rapidly moves up/down in one candle

    Attributes
    ----------
    ttype : str
        Trade type ('buy' or 'sell').
    configs : dict
        Configuration dictionary for the PumpDump indicator.
    """

    type = "Indicator_signal"
    name = "PumpDump"

    def __init__(self, ttype, **configs):
        super().__init__(ttype, configs)

    def find_signal(self, df: pd.DataFrame, *args) -> np.ndarray:
        """
        Return true if the price rapidly moves down in one candle (buy signal)
        Return true if the price rapidly moves up in one candle (sell signal)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing Trend indicator data.

        Returns
        -------
        np.ndarray
            Boolean array indicating the presence of a PumpDump signal.
        """
        # Find price change signal
        # buy trade
        if self.ttype == "buy":
            try:
                price_change_lower_1 = self.lower_bound_robust(
                    df["q_low_lag_1"].loc[0], df["price_change_1"]
                )
            except KeyError:
                return np.zeros(df.shape[0])
            price_change_lower_2 = 0
            price_change_lower_3 = 0
            return price_change_lower_1 | price_change_lower_2 | price_change_lower_3
        # sell trade
        try:
            price_change_higher_1 = self.higher_bound_robust(
                df["q_high_lag_1"].loc[0], df["price_change_1"]
            )
        except KeyError:
            return np.zeros(df.shape[0])
        price_change_higher_2 = 0
        price_change_higher_3 = 0
        return price_change_higher_1 | price_change_higher_2 | price_change_higher_3


class HighVolumeSignal(SignalBase):
    """
    Find candles with high volume

    Attributes
    ----------
    ttype : str
        Trade type ('buy' or 'sell').
    configs : dict
        Configuration dictionary for the HighVolume indicator.
    """

    type = "Indicator_signal"
    name = "HighVolume"

    def __init__(self, ttype, **configs):
        """
        Initialize the HighVolumeSignal class with trade type and
        configuration parameters.


        """
        super().__init__(ttype, configs)

    def find_signal(self, df: pd.DataFrame, *args) -> np.ndarray:
        """
        Find if ticker volume for the last 24 hours exceeds the min_volume_24 threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing 'volume_24' column.

        Returns
        -------
        np.ndarray
            Boolean array indicating the presence of a HighVolume signal.
        """
        # Find high volume signal
        try:
            high_vol = self.higher_bound_robust(
                df["quantile_vol"].loc[0], df["normalized_vol"]
            )
        except KeyError:
            return np.zeros(df.shape[0])
        return high_vol


class MACDSignal(SignalBase):
    """
    Find MACD signal

    Attributes
    ----------
    ttype : str
        Trade type ('buy' or 'sell').
    configs : dict
        Configuration dictionary for the MACD indicator.
    """

    type = "Indicator_signal"
    name = "MACD"

    def __init__(self, ttype, **configs):
        super().__init__(ttype, configs)
        self.configs = self.configs[self.name]["params"]
        self.low_bound = self.configs.get("low_bound", 20)
        self.high_bound = self.configs.get("high_bound", 80)

    def find_signal(self, df: pd.DataFrame, *args) -> np.ndarray:
        """
        Find MACD signal.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing 'macdhist', 'macd_dir', and 'macdsignal_dir' columns.
        timeframe_ratio : int
            Ratio of higher timeframe to working timeframe.

        Returns
        -------
        np.ndarray
            Boolean array indicating the presence of a MACD signal.
        """
        timeframe_ratio = args[0]
        macd_df = df[["macdhist", "macd_dir", "macdsignal_dir"]].copy()
        macd_df["macdhist_1"] = macd_df["macdhist"].shift(timeframe_ratio)
        macd_df["macdhist_2"] = macd_df["macdhist"].shift(timeframe_ratio * 2)
        macd_df["macdhist_3"] = macd_df["macdhist"].shift(timeframe_ratio * 3)

        if self.ttype == "buy":
            crossed_lines_up = self.crossed_lines(
                True, macd_df["macdhist"], macd_df["macdhist_1"], macd_df["macdhist_2"]
            )
            up_direction_macd = self.up_direction(macd_df["macd_dir"])
            up_direction_macdsignal = self.up_direction(macd_df["macdsignal_dir"])
            macd_up = crossed_lines_up & up_direction_macd & up_direction_macdsignal
            return macd_up

        crossed_lines_down = self.crossed_lines(
            False, macd_df["macdhist"], macd_df["macdhist_1"], macd_df["macdhist_2"]
        )
        down_direction_slowk = self.down_direction(macd_df["macd_dir"])
        down_direction_slowd = self.down_direction(macd_df["macdsignal_dir"])
        macd_down = crossed_lines_down & down_direction_slowk & down_direction_slowd
        return macd_down


class Volume24Signal(SignalBase):
    """
    Find Volume24 signal

    Attributes
    ----------
    ttype : str
        Trade type ('buy' or 'sell').
    configs : dict
        Configuration dictionary for the Volume24 indicator.
    """

    type = "Indicator_signal"
    name = "Volume24"

    def __init__(self, ttype, **configs):
        super().__init__(ttype, configs)
        self.configs = self.configs[self.name]["params"]
        self.min_volume_24 = self.configs.get("min_volume_24", 500000)

    def find_signal(self, df: pd.DataFrame, *args) -> np.ndarray:
        """
        Find if ticker volume for the last 24 hours exceeds the min_volume_24 threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing 'volume_24' column.

        Returns
        -------
        np.ndarray
            Boolean array indicating the presence of a Volume24 signal.
        """
        volume_24 = df["volume_24"]
        return np.where(volume_24 >= self.min_volume_24, 1, 0)


class PatternSignal(SignalBase):
    """
    Find TA patterns signal

    Attributes
    ----------
    ttype : str
        Trade type ('buy' or 'sell').
    configs : dict
        Configuration dictionary for the TA patterns indicators.
    """

    type = "Indicator_signal"
    name = "Pattern"

    def __init__(self, ttype, **configs):
        """
        Initialize PatternSignal.

        Parameters
        ----------
        ttype : str
            Trade type ('buy' or 'sell').
        configs : dict
            Configuration parameters for the Pattern signal.
        """
        super().__init__(ttype, configs)
        self.ttype = ttype
        self.configs = self.configs[self.name]["params"]
        self.vol_q_high = self.configs.get("vol_q_high", 0.75)
        self.vol_q_low = self.configs.get("vol_q_low", 0.25)
        self.vol_window = self.configs.get("vol_window", 48)
        self.window_low_bound = self.configs.get("window_low_bound", 1)
        self.window_high_bound = self.configs.get("window_high_bound", 6)
        self.first_candle = self.configs.get("first_candle", 0.667)
        self.second_candle = self.configs.get("second_candle", 0.5)

    def get_min_max_indexes(
        self, df: pd.DataFrame, gmax: pd.DataFrame.index, gmin: pd.DataFrame.index
    ) -> Tuple[Tuple, Tuple]:
        """
        Find min/max indexes based on trade type for H&S pattern detection.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing 'high' and 'low' columns.
        gmax : pd.DataFrame.index
            Indexes of global maximum points.
        gmin : pd.DataFrame.index
            Indexes of global minimum points.

        Returns
        -------
        tuple
            A tuple containing min and max indexes and corresponding values.
        """
        if self.ttype == "buy":  # find H&S
            # find 3 last global maximum
            ei = gmax
            ci = np.append(ei[0], ei[:-1])
            ai = np.append(ei[0:2], ei[:-2])
            # find 3 last global minimum
            fi = gmin
            di = np.append(fi[0], fi[:-1])
            bi = np.append(fi[0:2], fi[:-2])
            # find according high/low values
            aiv = df.loc[ai, "high"].values
            biv = df.loc[bi, "low"].values
            civ = df.loc[ci, "high"].values
            div = df.loc[di, "low"].values
            eiv = df.loc[ei, "high"].values
            fiv = df.loc[fi, "low"].values
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
            aiv = df.loc[ai, "low"].values
            biv = df.loc[bi, "high"].values
            civ = df.loc[ci, "low"].values
            div = df.loc[di, "high"].values
            eiv = df.loc[ei, "low"].values
            fiv = df.loc[fi, "high"].values
        return (ai, bi, ci, di, ei, fi), (aiv, biv, civ, div, eiv, fiv)

    def create_pattern_vector(self, df: pd.DataFrame, res: np.ndarray) -> np.ndarray:
        """
        Create vector showing potential trade entry points after pattern appearance.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe for the pattern vector creation.
        res : np.ndarray
            Results of the pattern search.

        Returns
        -------
        np.ndarray
            Boolean array indicating trade entry points.
        """
        v = np.zeros(df.shape[0], dtype=int)
        for i in range(self.window_low_bound, self.window_high_bound):
            try:
                v[res[res > 0] + i] = 1
            except IndexError:
                break
        return v

    def head_and_shoulders(
        self, df: pd.DataFrame, min_max_idxs: tuple, min_max_vals: tuple
    ) -> np.ndarray:
        """
        Find H&S or inverted H&S pattern.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe for the pattern detection.
        min_max_idxs : tuple
            Tuple of minimum and maximum indexes.
        min_max_vals : tuple
            Tuple of minimum and maximum values.

        Returns
        -------
        np.ndarray
            Boolean array indicating H&S pattern occurrences.
        """
        _, _, _, _, _, fi = min_max_idxs
        _, biv, _, div, _, fiv = min_max_vals
        if self.ttype == "buy":  # inverted H&S
            # find if global maximums and minimums make H&S pattern
            res = np.where((1.005 * div < biv) & (1.005 * div < fiv), fi, 0)
        else:  # H&S
            # find if global maximums and minimums make inverted H&S pattern
            res = np.where((div > 1.005 * biv) & (div > 1.005 * fiv), fi, 0)
        v = self.create_pattern_vector(df, res)
        return v

    def hlh_lhl(
        self, df: pd.DataFrame, min_max_idxs: tuple, min_max_vals: tuple
    ) -> np.ndarray:
        """
        Find HLH/LHL pattern.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe for the pattern detection.
        min_max_idxs : tuple
            Tuple of minimum and maximum indexes.
        min_max_vals : tuple
            Tuple of minimum and maximum values.

        Returns
        -------
        np.ndarray
            Boolean array indicating HLH/LHL pattern occurrences.
        """
        _, _, _, _, _, fi = min_max_idxs
        _, _, civ, div, eiv, fiv = min_max_vals
        if self.ttype == "buy":  # LHL
            # find if global maximums and minimums make LHL pattern
            res = np.where((1.005 * civ < eiv) & (1.005 * div < fiv), fi, 0)
        else:  # find HLH
            # find if global maximums and minimums make HLH pattern
            res = np.where((civ > 1.005 * eiv) & (div > 1.005 * fiv), fi, 0)
        v = self.create_pattern_vector(df, res)
        return v

    def dt_db(
        self, df: pd.DataFrame, min_max_idxs: tuple, min_max_vals: tuple
    ) -> np.ndarray:
        """
        Find Double Top/Double Bottom pattern.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe for the pattern detection.
        min_max_idxs : tuple
            Tuple of minimum and maximum indexes.
        min_max_vals : tuple
            Tuple of minimum and maximum values.

        Returns
        -------
        np.ndarray
            Boolean array indicating Double Top/Bottom pattern occurrences.
        """
        _, _, _, _, _, fi = min_max_idxs
        _, _, civ, div, eiv, fiv = min_max_vals
        if self.ttype == "buy":  # LHL
            # find if global maximums and minimums make DP pattern
            res = np.where((civ > eiv) & (np.abs(div - fiv) / div < 0.001), fi, 0)
        else:  # find HLH
            # find if global maximums and minimums make DB pattern
            res = np.where((civ < eiv) & (np.abs(div - fiv) / div < 0.001), fi, 0)
        v = self.create_pattern_vector(df, res)
        return v

    def triangle(
        self, df: pd.DataFrame, min_max_idxs: tuple, min_max_vals: tuple
    ) -> np.ndarray:
        """
        Find ascending/descending triangle pattern.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe for the pattern detection.
        min_max_idxs : tuple
            Tuple of minimum and maximum indexes.
        min_max_vals : tuple
            Tuple of minimum and maximum values.

        Returns
        -------
        np.ndarray
            Boolean array indicating ascending/descending triangle pattern occurrences.
        """
        _, _, _, _, _, fi = min_max_idxs
        aiv, biv, civ, div, eiv, fiv = min_max_vals
        if self.ttype == "buy":  # ascending triangle
            # find if global maximums and minimums make ascending triangle pattern
            res = np.where(
                (biv < div)
                & (div < fiv)
                & (np.abs(aiv - civ) / aiv < 0.001)
                & (np.abs(civ - eiv) / civ < 0.001)
                & (np.abs(aiv - eiv) / aiv < 0.001),
                fi,
                0,
            )
        else:  # descending triangle
            # find if global maximums and minimums make descending triangle pattern
            res = np.where(
                (biv > div)
                & (div > fiv)
                & (np.abs(aiv - civ) / aiv < 0.001)
                & (np.abs(civ - eiv) / civ < 0.001)
                & (np.abs(aiv - eiv) / aiv < 0.001),
                fi,
                0,
            )
        v = self.create_pattern_vector(df, res)
        return v

    def swing(
        self, df: pd.DataFrame, min_max_idxs: tuple, min_max_vals: tuple, avg_gap: float
    ) -> np.ndarray:
        """
        Find swing pattern.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe for the pattern detection.
        min_max_idxs : tuple
            Tuple of minimum and maximum indexes.
        min_max_vals : tuple
            Tuple of minimum and maximum values.
        avg_gap : float
            Average gap value for swing pattern detection.

        Returns
        -------
        np.ndarray
            Boolean array indicating swing pattern occurrences.
        """
        _, _, _, _, _, fi = min_max_idxs
        aiv, biv, civ, div, eiv, fiv = min_max_vals
        res = np.where(
            (np.abs(aiv - civ) / aiv < 0.0025)
            & (np.abs(civ - eiv) / civ < 0.0025)
            & (np.abs(aiv - eiv) / aiv < 0.0025)
            & (np.abs(biv - div) / biv < 0.0025)
            & (np.abs(div - fiv) / div < 0.0025)
            & (np.abs(biv - fiv) / biv < 0.0025)
            & (np.abs(aiv - fiv) <= avg_gap),
            fi,
            0,
        )
        v = self.create_pattern_vector(df, res)
        return v

    def find_signal(self, df: pd.DataFrame, *args) -> np.ndarray:
        """
        Find one of TA patterns like H&S, HLH/LHL, DT/DB and
        good candles that confirm that pattern.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing market data for pattern detection.

        Returns
        -------
        np.ndarray
            Boolean array indicating the detected trading signals based on patterns.
        """
        # Find one of TA patterns like H&S, HLH/LHL, DT/DB and
        # good candles that confirm that pattern
        avg_gap = (df["high"] - df["low"]).mean()
        high_max = df[df["high_max"] > 0].index
        low_min = df[df["low_min"] > 0].index
        # bring both global extremum lists to one length
        min_len = min(len(high_max), len(low_min))
        if min_len == 0:
            return np.zeros(df.shape[0])
        gmax, gmin = high_max[:min_len], low_min[:min_len]
        # find minimum and maximum indexes for patterns search
        min_max_idxs, min_max_vals = self.get_min_max_indexes(df, gmax, gmin)
        has = self.head_and_shoulders(df, min_max_idxs, min_max_vals)
        hlh = self.hlh_lhl(df, min_max_idxs, min_max_vals)
        dt = self.dt_db(df, min_max_idxs, min_max_vals)
        tgc = self.two_good_candles(df, self.ttype)
        sw = self.swing(df, min_max_idxs, min_max_vals, avg_gap)
        pattern_signal = (has | hlh | dt | sw) & tgc
        return pattern_signal


class FindSignal:
    """
    Class for searching of the indicator combination

    Attributes
    ----------
    ttype : str
        The type of trade ('buy' or 'sell').
    configs : dict
        Configuration parameters including indicators, patterns, and timeframes.
    """

    def __init__(self, ttype, configs):
        self.ttype = ttype
        self.configs = configs
        self.indicator_list = configs["Indicator_list"]
        self.indicator_signals = self.prepare_indicator_signals()
        self.patterns = configs["Patterns"]
        self.work_timeframe = configs["Timeframes"]["work_timeframe"]
        self.higher_timeframe = configs["Timeframes"]["higher_timeframe"]
        self.timeframe_div = configs["Data"]["Basic"]["params"]["timeframe_div"]

    def prepare_indicator_signals(self) -> List[SignalBase]:
        """
        Prepare all indicator signal classes.

        Returns
        -------
        list
            A list of initialized indicator signal objects.
        """
        indicator_signals = []
        for indicator in self.indicator_list:
            if (
                indicator == "HighVolume" and self.ttype == "sell"
            ) or indicator == "ATR":
                continue
            indicator_signals.append(
                SignalFactory.factory(indicator, self.ttype, self.configs)
            )
        return indicator_signals

    def find_signal(  # pylint: disable=R0912
        self,
        dfs: dict,
        ticker: str,
        timeframe: str,
        data_qty: int,
        data_qty_higher: int,
    ) -> List[List]:
        """
        Search for the signals through the dataframe, if found -
        add its index and trade type to the list.
        If dataset was updated - don't search through the whole dataset,
        only through updated part.

        Parameters
        ----------
        dfs : dict
            A dictionary containing dataframes for different tickers and timeframes.
        ticker : str
            The ticker symbol for the asset.
        timeframe : str
            The timeframe for the signals to be searched.
        data_qty : int
            The quantity of data to consider for finding signals.
        data_qty_higher : int
            The quantity of data from the higher timeframe to consider.

        Returns
        -------
        list
            A list of detected trading signals including
            their respective indexes and metadata.
        """
        points: List[List] = []

        try:
            if self.ttype == "buy":
                df_work = dfs[ticker][self.work_timeframe]["data"]["buy"].copy()
            else:
                df_work = dfs[ticker][self.work_timeframe]["data"]["sell"].copy()
        except KeyError:
            return points

        sig_patterns = [p.copy() for p in self.patterns]
        timeframe_ratio = int(
            self.timeframe_div[self.higher_timeframe]
            / self.timeframe_div[self.work_timeframe]
        )
        # Create signal point df for each indicator
        # merge higher dataframe timestamps and working dataframe
        # timestamps in one dataframe
        trade_points = pd.DataFrame()
        # Fill signal point df with signals from indicators
        for indicator_signal in self.indicator_signals:
            # if indicators work with higher timeframe -
            # we should treat them differently
            if indicator_signal.name == "Trend":
                if "linear_reg_angle" not in df_work.columns:
                    return points
                fs = indicator_signal.find_signal(df_work)
            elif indicator_signal.name == "MACD":
                # check higher timeframe signals every hour
                if data_qty_higher > 1:
                    if "macdsignal" not in df_work.columns:
                        return points
                    fs = indicator_signal.find_signal(df_work, timeframe_ratio)
                else:
                    fs = np.zeros(df_work.shape[0])
            else:
                fs = indicator_signal.find_signal(df_work)

            trade_points[indicator_signal.name] = fs

        # If any pattern has all 1 - add corresponding point as a signal
        for pattern in sig_patterns:
            # find indexes of trade points
            if self.ttype == "sell" and pattern == ["HighVolume"]:
                continue
            # check if pattern has all 1
            pattern_points = trade_points[pattern]
            max_shape = pattern_points.shape[1]
            pattern_points = pattern_points.sum(axis=1)
            # get trade indexes
            trade_indexes = pattern_points[pattern_points == max_shape].index
            # save only recent trade indexes
            trade_indexes = trade_indexes[df_work.shape[0] - trade_indexes < data_qty]
            sig_pattern = "_".join(pattern)
            points += [
                [
                    ticker,
                    self.work_timeframe,
                    index,
                    self.ttype,
                    df_work.loc[index, "time"],
                    sig_pattern,
                    [],
                    [],
                    [],
                    0,
                ]
                for index in trade_indexes
            ]
        return points
