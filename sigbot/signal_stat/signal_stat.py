"""
This module provides a functionality used to calculate and store signal statistics
for trading strategies. It`s capable of writing, processing, and loading statistical
data related to buy and sell signals for financial assets. The statistics include
Maximum Favorable Excursion (MFE), Maximum Adverse Excursion (MAE), and
price differentials for various time intervals after a signal.
"""

from typing import Tuple, Union

import pandas as pd


class SignalStat:
    """
    Class for acquiring signal statistics.

    Attributes
    ----------
    opt_type : str, optional
        Type of optimization ('ml', 'optimize'). Default is None.
    **configs : dict
        Configuration dictionary containing parameters for the SignalStat class.
    """

    type = "SignalStat"

    def __init__(self, opt_type=None, **configs):
        self.configs = configs[self.type]["params"]
        self.take_profit_multiplier = self.configs.get("take_profit_multiplier", 2)
        self.stop_loss_multiplier = self.configs.get("stop_loss_multiplier", 2)
        # the number of last candles for which statistics are calculated
        self.stat_range = self.configs.get("stat_range", 24)
        self.test = self.configs.get("test", False)
        self.stat_limit_hours = self.configs.get("stat_limit_hours", 48)
        # Minimal number of candles that should pass from previous
        # signal to add the new signal to statistics
        self.min_prev_candle_limit = self.configs.get("min_prev_candle_limit", 3)
        self.min_prev_candle_limit_higher = self.configs.get(
            "min_prev_candle_limit_higher", 2
        )
        # dictionary that is used to determine too late signals
        # according to current work_timeframe
        self.timeframe_div = configs["Data"]["Basic"]["params"]["timeframe_div"]
        # Get working and higher timeframes
        self.work_timeframe = configs["Timeframes"]["work_timeframe"]
        self.higher_timeframe = configs["Timeframes"]["higher_timeframe"]
        self.higher_tf_indicator_set = {
            i for i in configs["Higher_TF_indicator_list"] if i != "Trend"
        }
        if opt_type == "ml":
            self.buy_stat_path = f"../ml/signal_stat/buy_stat_{self.work_timeframe}.pkl"
            self.sell_stat_path = (
                f"../ml/signal_stat/sell_stat_{self.work_timeframe}.pkl"
            )
        elif opt_type == "optimize":
            self.buy_stat_path = (
                f"../optimizer/signal_stat/buy_stat_{self.work_timeframe}.pkl"
            )
            self.sell_stat_path = (
                f"../optimizer/signal_stat/sell_stat_{self.work_timeframe}.pkl"
            )
        else:
            self.buy_stat_path = f"signal_stat/buy_stat_{self.work_timeframe}.pkl"
            self.sell_stat_path = f"signal_stat/sell_stat_{self.work_timeframe}.pkl"

    def write_stat(
        self, dfs: dict, signal_points: list, data_qty_higher: Union[int, None] = None
    ) -> dict:
        """
        Write signal statistics for each signal point for the current ticker
        and timeframe.

        Parameters
        ----------
        dfs : dict
            Dictionary containing dataframes for different tickers and timeframes.
        signal_points : list
            List of signal points that contain information about the signals.
        data_qty_higher : int, optional
            The number of data points for higher timeframes. Default is None.

        Returns
        -------
        dict
            Updated dictionary with the written signal statistics.
        """
        for point in signal_points:
            (
                ticker,
                timeframe,
                index,
                ttype,
                time,
                pattern,
                _,
                _,
                _,
                _,
            ) = point
            # we don't write statistics for the High Volume pattern
            if pattern == "HighVolume":
                continue
            df = dfs[ticker][timeframe]["data"][ttype]
            # array of prices after signal
            index = df.loc[df["time"] == time, "close"].index[0]
            signal_price = df.iloc[index]["close"]
            signal_smooth_price = df.iloc[index]["close_smooth"]
            # If index of point was found too early - we shouldn't use it
            if index < 50:
                continue
            # Get statistics, process it and write into the database
            high_result_prices, low_result_prices, close_smooth_prices, atr = (
                self.get_result_price_after_period(df, index)
            )
            dfs = self.process_statistics(
                dfs,
                point,
                signal_price,
                signal_smooth_price,
                high_result_prices,
                low_result_prices,
                close_smooth_prices,
                atr,
            )
        return dfs

    def get_result_price_after_period(
        self, df: pd.DataFrame, index: int
    ) -> Tuple[list, list, list, float]:
        """
        Get result prices after every T minutes.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the signal data.
        index : int
            The index of the signal.

        Returns
        -------
        Tuple[list, list, float]
            A tuple containing lists of high, low, and
            smooth prices after the signal, and the ATR value.
        """
        high_result_prices = []
        low_result_prices = []
        close_smooth_prices = []
        atr = df["atr"].iloc[index]
        for t in range(1, self.stat_range + 1):
            try:
                high_result_prices.append(df["high"].iloc[index + t])
                low_result_prices.append(df["low"].iloc[index + t])
                close_smooth_prices.append(df["close_smooth"].iloc[index + t])
            except IndexError:
                high_result_prices.append(df["close"].iloc[index])
                low_result_prices.append(df["close"].iloc[index])
                close_smooth_prices.append(df["close_smooth"].iloc[index])
        return high_result_prices, low_result_prices, close_smooth_prices, atr

    @staticmethod
    def process_statistics(
        dfs: dict,
        point: list,
        signal_price: float,
        signal_smooth_price: float,
        high_result_prices: list,
        low_result_prices: list,
        close_smooth_prices: list,
        atr: float,
    ) -> dict:
        """
        Calculate and write statistics to the stat DataFrame if not already present.

        Parameters
        ----------
        dfs : dict
            Dictionary containing dataframes for different tickers and timeframes.
        point : list
            List containing information about the signal.
        signal_price : float
            The price of the asset at the time of the signal.
        signal_smooth_price : float
            The smoothed price at the time of the signal.
        high_result_prices : list
            List of high prices after the signal.
        low_result_prices : list
            List of low prices after the signal.
        close_smooth_prices : list
            List of smoothed close prices after the signal.
        atr : float
            The Average True Range (ATR) value at the time of the signal.

        Returns
        -------
        dict
            Updated dictionary with processed statistics.
        """
        # get data
        (
            ticker,
            timeframe,
            _,
            ttype,
            time,
            pattern,
            _,
            _,
            _,
            _,
        ) = point
        ticker = ticker.replace("-", "").replace("/", "").replace("SWAP", "")

        tmp = pd.DataFrame()
        tmp["time"] = [time]
        tmp["ticker"] = [ticker]
        tmp["timeframe"] = [timeframe]
        tmp["pattern"] = [pattern]
        # If current statistics is not in stat dataframe - write it
        if ttype == "buy":
            stat = dfs["stat"]["buy"]
        else:
            stat = dfs["stat"]["sell"]
        tmp["signal_price"] = [signal_price]
        tmp["signal_smooth_price"] = [signal_smooth_price]
        # write result price, MFE and MAE, if MAE is too low -
        # replace it with MFE/1000 to prevent zero division
        for i in range(len(high_result_prices)):
            # calculate price diff :
            # (close_smooth_price - signal_price) / close_smooth_price
            tmp["close_smooth_price"] = [close_smooth_prices[i]]
            tmp[f"pct_price_diff_{i+1}"] = (
                tmp["close_smooth_price"] - tmp["signal_smooth_price"]
            ) + 1e-8 / (tmp["close_smooth_price"] + 1e-8) * 100
            if ttype == "buy":
                # calculater MFE and MAE
                mfe = max(max(high_result_prices[: i + 1]) - signal_price, 0) / atr
                tmp[f"mfe_{i+1}"] = [mfe]
                tmp[f"mae_{i+1}"] = (
                    max(signal_price - min(low_result_prices[: i + 1]), 0) / atr
                )
            else:
                # calculater MFE and MAE
                mfe = max(signal_price - min(low_result_prices[: i + 1]), 0) / atr
                tmp[f"mfe_{i+1}"] = [mfe]
                tmp[f"mae_{i+1}"] = (
                    max(max(high_result_prices[: i + 1]) - signal_price, 0) / atr
                )
            # drop unnecessary columns
            tmp = tmp.drop(["close_smooth_price"], axis=1)
        # if can't find similar statistics in the dataset -
        # just add it to stat dataframe, else update stat columns
        if (
            stat[
                (stat["time"] == time)
                & (stat["ticker"] == ticker)
                & (stat["timeframe"] == timeframe)
                & (stat["pattern"] == pattern)
            ].shape[0]
            == 0
        ):
            stat = pd.concat([stat, tmp], ignore_index=True)
        else:
            mfe_cols = [f"mfe_{i+1}" for i in range(len(high_result_prices))]
            mae_cols = [f"mae_{i+1}" for i in range(len(high_result_prices))]
            pct_price_diff_cols = [
                f"pct_price_diff_{i+1}" for i in range(len(high_result_prices))
            ]
            stat.loc[
                (stat["time"] == time)
                & (stat["ticker"] == ticker)
                & (stat["timeframe"] == timeframe)
                & (stat["pattern"] == pattern),
                mfe_cols,
            ] = tmp[mfe_cols].values
            stat.loc[
                (stat["time"] == time)
                & (stat["ticker"] == ticker)
                & (stat["timeframe"] == timeframe)
                & (stat["pattern"] == pattern),
                mae_cols,
            ] = tmp[mae_cols].values
            stat.loc[
                (stat["time"] == time)
                & (stat["ticker"] == ticker)
                & (stat["timeframe"] == timeframe)
                & (stat["pattern"] == pattern),
                pct_price_diff_cols,
            ] = tmp[pct_price_diff_cols].values
        # updata database with new stat data
        if ttype == "buy":
            dfs["stat"]["buy"] = stat
        else:
            dfs["stat"]["sell"] = stat
        return dfs

    def save_statistics(self, dfs: dict) -> None:
        """
        Save statistics to the disk.

        Parameters
        ----------
        dfs : dict
            Dictionary containing dataframes for different tickers and timeframes.
        """
        if not self.test:
            # Write statistics to the dataframe dict
            dfs["stat"]["buy"].to_pickle(self.buy_stat_path)
            dfs["stat"]["sell"].to_pickle(self.sell_stat_path)

    def load_statistics(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load statistics from the disk.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames containing statistics for buy and sell signals.
        """
        try:
            buy_stat = pd.read_pickle(self.buy_stat_path)  # nosec
            sell_stat = pd.read_pickle(self.sell_stat_path)  # nosec
        except (FileNotFoundError, EOFError):
            buy_stat = pd.DataFrame(columns=["time", "ticker", "timeframe", "pattern"])
            sell_stat = pd.DataFrame(columns=["time", "ticker", "timeframe", "pattern"])
        return buy_stat, sell_stat

    def cut_stat_df(self, stat, pattern):
        """
        Get the latest signals not created earlier than a specified number of hours ago.

        Parameters
        ----------
        stat : pd.DataFrame
            DataFrame containing signal statistics.
        pattern : str
            The pattern for which the statistics are being cut.

        Returns
        -------
        pd.DataFrame
            DataFrame with signals created within the specified timeframe.
        """
        latest_time = stat["time"].max()
        stat = stat[
            latest_time - stat["time"]
            < pd.Timedelta(self.stat_limit_hours[pattern], "h")
        ]
        return stat

    def calculate_total_stat(
        self, dfs: dict, ttype: str, pattern: str
    ) -> Tuple[list, int]:
        """
        Calculate signal statistics for all found signals and tickers.

        Parameters
        ----------
        dfs : dict
            Dictionary containing dataframes for different tickers and timeframes.
        ttype : str
            Type of trade ('buy' or 'sell').
        pattern : str
            Pattern for which statistics are being calculated.

        Returns
        -------
        Tuple[list, int]
            List of statistics and the number of signals found.
        """
        stat = dfs["stat"][ttype]
        # get statistics by pattern
        stat = stat[stat["pattern"] == pattern]
        # get only last signals that has been created not earlier than
        # N hours ago (depends on pattern)
        stat = self.cut_stat_df(stat, pattern)
        if stat.shape[0] == 0:
            return [(0, 0, 0) for _ in range(1, self.stat_range + 1)], stat.shape[0]
        result_statistics = []
        # calculate E-ratio (MFE/MAE), median and standard deviation of price movement
        # for each time interval after signal
        for t in range(1, self.stat_range + 1):
            try:
                e_ratio = round(sum(stat[f"mfe_{t}"]) / sum(stat[f"mae_{t}"]), 4)
                e_ratio = min(e_ratio, 10)
            except ZeroDivisionError:
                if sum(stat[f"mfe_{t}"]) > 0:
                    e_ratio = 10
                else:
                    e_ratio = 1
            pct_price_diff_mean = round(
                stat.loc[
                    stat[f"pct_price_diff_{t}"] != 0, f"pct_price_diff_{t}"
                ].mean(),
                2,
            )
            pct_price_diff_std = round(
                stat.loc[stat[f"pct_price_diff_{t}"] != 0, f"pct_price_diff_{t}"].std(),
                2,
            )
            result_statistics.append((e_ratio, pct_price_diff_mean, pct_price_diff_std))
        return result_statistics, stat.shape[0]

    def check_close_trades(
        self,
        stat: pd.DataFrame,
        df_len: int,
        ticker: str,
        index: int,
        point_time: pd.Timestamp,
        pattern: str,
        prev_point: tuple,
    ) -> bool:
        """
        Check if the signal point wasn't too close to the previous one.

        Parameters
        ----------
        stat : pd.DataFrame
            DataFrame containing statistics.
        df_len : int
            The length of the ticker DataFrame.
        ticker : str
            Ticker symbol.
        index : int
            Index of the signal in the DataFrame.
        point_time : pd.Timestamp
            Time of the signal.
        pattern : str
            The pattern being checked.
        prev_point : tuple
            Previous signal point for comparison.

        Returns
        -------
        bool
            Whether the signal point can be used.
        """
        # if the same signal is in dataframe, and it appeared not too early -
        # add it again to update statistics
        same_signal = stat[
            (stat["ticker"] == ticker)
            & (stat["pattern"] == pattern)
            & (stat["time"] == point_time)
        ]
        if same_signal.shape[0] > 0:
            # if signal appeared too early - don't consider it
            if index < df_len - self.stat_range * 1.5 - 1:
                return False
            return True
        # else check the latest similar signal's time to prevent close signals
        same_signal_timestamps = stat.loc[
            (stat["ticker"] == ticker) & (stat["pattern"] == pattern), "time"
        ]
        # find the last timestamp when previous similar pattern appeared
        if same_signal_timestamps.shape[0] > 0:
            last_signal_timestamp = same_signal_timestamps.max()
        else:
            last_signal_timestamp = pd.Timestamp(0)
        # select the latest timestamp among similar signals from statistics
        # and previous signal
        prev_ticker, prev_time, prev_pattern = prev_point
        if prev_time is not None and ticker == prev_ticker and pattern == prev_pattern:
            prev_signal_timestamp = prev_time
            last_signal_timestamp = max(prev_signal_timestamp, last_signal_timestamp)
        # check if signal appeared too early after previous signal
        if set(pattern.split("_")).intersection(self.higher_tf_indicator_set):
            if (
                point_time - last_signal_timestamp
            ).total_seconds() > self.timeframe_div[
                self.higher_timeframe
            ] * self.min_prev_candle_limit_higher:
                return True
        else:
            if (
                point_time - last_signal_timestamp
            ).total_seconds() > self.timeframe_div[
                self.work_timeframe
            ] * self.min_prev_candle_limit:
                return True
        return False
