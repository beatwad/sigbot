"""
This module provides the `Visualizer` class, which is used to create visual
representations of financial trading signals, indicators, candlestick charts,
and other statistical information.
It is designed to support various indicators and patterns, providing
useful insights through graphical plots for trading analysis.
"""

import json
from typing import List, Union

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import rcParams

matplotlib.use("Agg")
rcParams["font.family"] = "DejaVu Sans"


class Visualizer:
    """
    Class for visualization of trading signals, ticker candlesticks,
    indicators and signal statistics

    Attributes
    ----------
    configs : dict
        A dictionary containing configuration parameters, including timeframes,
        plot settings, indicator settings, and file paths for previous statistics.
    """

    type = "Visualizer"
    ticker_color = "white"
    border_color = "white"
    background_color = "#010113"
    stat_color_1 = "#0ED6F1"
    stat_std_color_1 = "#19E729"
    stat_std_color_2 = "#E73B19"
    stat_color_2 = "#EE4B1A"

    def __init__(self, **configs):
        # Get working and higher timeframes
        self.working_timeframe = configs["Timeframes"]["work_timeframe"]
        self.higher_timeframe = configs["Timeframes"]["higher_timeframe"]
        # Get Visualizer parameters
        self.configs = configs[self.type]["params"]
        # Path to save plot files
        self.image_path = self.configs["image_path"]
        self.indicator_configs = configs["Indicator_signal"]
        self.plot_width = self.configs.get("plot_width", 10)
        self.indicator_dict = self.configs.get("indicator_dict", {})
        self.level_indicators = self.configs.get("level_indicators", [])
        self.boundary_indicators = self.configs.get("boundary_indicators", [])
        # Max number of previous candles for which signal can be searched for
        self.max_prev_candle_limit = self.configs.get("max_prev_candle_limit", 0)
        # dict for storing previous statistics values
        self.prev_e_ratio_path = self.configs["prev_e_ratio_path"]
        self.prev_mar_path = self.configs["prev_mar_path"]
        # list of indicator parameters that can be plotted
        self.indicator_params = configs["Indicator_signal"]
        self.indicators_to_plot = ["RSI", "STOCH"]

    def plot_indicator_parameters(
        self, point_type: str, index: int, axs: plt.axis, indicator_list: list
    ) -> None:
        """
        Plot parameters of indicator (like low or high boundary, etc.)

        Parameters
        ----------
        point_type : str
            The type of point to plot ('buy' or 'sell').
        index : int
            The index of the indicator in the indicator list.
        axs : plt.axis
            The axis to plot on.
        indicator_list : list
            A list of indicators to plot.
        """
        indicator = indicator_list[index]
        if indicator in self.indicators_to_plot:
            indicator_params = list(
                self.indicator_params[point_type][indicator]["params"].values()
            )
            if indicator_params:
                if indicator in self.boundary_indicators:
                    if point_type == "buy":
                        axs[index + 1].axhline(
                            y=indicator_params[0],
                            color="g",
                            linestyle="--",
                            linewidth=1.5,
                        )
                    else:
                        axs[index + 1].axhline(
                            y=indicator_params[1],
                            color="r",
                            linestyle="--",
                            linewidth=1.5,
                        )

    def plot_point(
        self,
        point_type: str,
        data: pd.DataFrame,
        ax: plt.axis,
        index: int = 0,
        higher: bool = False,
    ) -> None:
        """
        Plot trade point.

        Parameters
        ----------
        point_type : str
            The type of point to plot (e.g., 'buy' or 'sell').
        data : pd.DataFrame
            The data containing the price information.
        ax : plt.axis
            The axis to plot on.
        index : int, optional
            The index of the point to plot (default is 0).
        higher : bool, optional
            Whether to plot on a higher timeframe (default is False).
        """
        if index > 0:
            color = "blue"
        elif point_type == "buy":
            color = "green"
        else:
            color = "red"
        if higher:
            plot_width = self.plot_width * 2 - index - 1
        else:
            plot_width = self.plot_width - index
        if point_type == "buy":
            ax.scatter(plot_width, data["close"].iloc[-1 - index], s=50, color=color)
        else:
            ax.scatter(plot_width, data["close"].iloc[-1 - index], s=50, color=color)

    @staticmethod
    def plot_levels(data: pd.DataFrame, levels: list, axs: plt.axis) -> None:
        """
        Plot support and resistance levels.

        Parameters
        ----------
        data : pd.DataFrame
            The data containing price information.
        levels : list
            A list of levels to plot.
        axs : plt.axis
            The axis to plot on.
        """
        for level in levels:
            if (
                data["low"].min() <= level[0] <= data["high"].max()
            ):  # and level[1] == 3:
                axs[0].axhline(y=level[0], color="b", linestyle="dotted", linewidth=1.5)

    def save_plot(self, ticker, timeframe, pattern, data):
        """
        Save the plot as an image file.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the asset to plot.
        timeframe : str
            The timeframe of the data (e.g., '1h', '1d').
        pattern : str
            The trading pattern or signal represented by the plot.
        data : pd.DataFrame
            The data containing price information.

        Returns
        -------
        str
            The file path of the saved plot image.
        """
        filename = (
            f"{self.image_path}/{ticker}_{timeframe}_"
            f"{pattern}_{data['time'].iloc[-1]}.png"
        )
        plt.savefig(filename, bbox_inches="tight")
        return filename

    @staticmethod
    def process_ticker(ticker: str) -> str:
        """
        Bring ticker to more convenient view.

        Parameters
        ----------
        ticker : str
            The ticker symbol.

        Returns
        -------
        str
            The formatted ticker symbol.
        """
        if "-" in ticker:
            return ticker
        if "/" in ticker:
            ticker = ticker.replace("/", "-")
            return ticker
        ticker = ticker[:-4] + "/" + ticker[-4:]
        return ticker

    @staticmethod
    def get_statistics_dict_key(pattern: List[str]) -> str:
        """
        Get previous avg E-ratio coefficient and
        save current E-ratio to statistics dictionary.

        Parameters
        ----------
        pattern : list
            A list representing the pattern for which the key is generated.

        Returns
        -------
        str
            The generated key for the statistics dictionary.
        """
        if "PumpDump" in str(pattern[0][0]):
            key = str([pattern[0][0]] + pattern[1:])
        else:
            key = str(pattern)
        return key

    def load_prev_e_ratio_dict(self) -> dict:
        """
        Load E-ratio coefficients dict from file.

        Returns
        -------
        dict
            A dictionary containing previous E-ratio coefficients.
        """
        try:
            with open(self.prev_e_ratio_path, "r+") as f:
                prev_e_ratio_stat_dict = json.load(f)
        except FileNotFoundError:
            prev_e_ratio_stat_dict = {}
        return prev_e_ratio_stat_dict

    def load_prev_mar_dict(self) -> dict:
        """
        Load MAR coefficients dict from file.

        Returns
        -------
        dict
            A dictionary containing previous MAR coefficients.
        """
        try:
            with open(self.prev_mar_path, "r+") as f:
                prev_mar_stat_dict = json.load(f)
        except FileNotFoundError:
            prev_mar_stat_dict = {}
        return prev_mar_stat_dict

    def get_prev_avg_e_ratio_coef(
        self, key: str, point_type: str, avg_e_ratio_coef: float
    ) -> float:
        """
        Get previous E-ratio coefficient and
        fill the statistic dict by current value of E-ratio.

        Parameters
        ----------
        key : str
            The key for the E-ratio in the statistics dictionary.
        point_type : str
            The type of point ('buy' or 'sell').
        avg_e_ratio_coef : float
            The current average E-ratio coefficient.

        Returns
        -------
        float
            The previous average E-ratio coefficient.
        """
        prev_e_ratio_stat_dict = self.load_prev_e_ratio_dict()
        if key in prev_e_ratio_stat_dict:
            prev_avg_e_ratio_coef = prev_e_ratio_stat_dict[key][point_type]
        else:
            prev_avg_e_ratio_coef = avg_e_ratio_coef
            prev_e_ratio_stat_dict[key] = {"sell": None, "buy": None}
        # update previous E-ratio coefficients dict and save it to file
        if prev_e_ratio_stat_dict[key][point_type] != avg_e_ratio_coef:
            prev_e_ratio_stat_dict[key][point_type] = avg_e_ratio_coef
            with open(self.prev_e_ratio_path, "w+") as f:
                json.dump(prev_e_ratio_stat_dict, f)
        return prev_avg_e_ratio_coef

    def get_prev_avg_mar_coef(
        self, key: str, point_type: str, avg_mar_coef: float
    ) -> float:
        """
        Get previous MAR coefficient and
        fill the statistic dict by current value of MAR.

        Parameters
        ----------
        key : str
            The key for the MAR in the statistics dictionary.
        point_type : str
            The type of point ('buy' or 'sell').
        avg_mar_coef : float
            The current average MAR coefficient.

        Returns
        -------
        float
            The previous average MAR coefficient.
        """
        prev_mar_stat_dict = self.load_prev_mar_dict()
        if key in prev_mar_stat_dict:
            prev_avg_mar_coef = prev_mar_stat_dict[key][point_type]
        else:
            prev_avg_mar_coef = avg_mar_coef
            prev_mar_stat_dict[key] = {"sell": None, "buy": None}
        # update previous MAR coefficients dict and save it to file
        if prev_mar_stat_dict[key][point_type] != avg_mar_coef:
            prev_mar_stat_dict[key][point_type] = avg_mar_coef
            with open(self.prev_mar_path, "w+") as f:
                json.dump(prev_mar_stat_dict, f)
        return prev_avg_mar_coef

    @staticmethod
    def statistics_change(
        prev_avg_coef: Union[None, float], avg_coef: Union[None, float]
    ) -> str:
        """
        Measure statistics difference between previous signal and current signal.

        Parameters
        ----------
        prev_avg_coef : Union[None, float]
            The previous average coefficient.
        avg_coef : Union[None, float]
            The current average coefficient.

        Returns
        -------
        str
            A string representing the change in statistics.
        """
        if prev_avg_coef is None or avg_coef is None:
            return "= no change / без изменений"
        stat_diff = round(avg_coef - prev_avg_coef, 4)
        if stat_diff < 0:
            return f"= decreased on / уменьшилcя на {abs(stat_diff)}"
        if stat_diff > 0:
            return f"= increased on / вырос на {stat_diff}"
        return "= no change / без изменений"

    @staticmethod
    def round_price(df: pd.DataFrame) -> float:
        """
        Round the price.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing price information.

        Returns
        -------
        float
            The rounded price.
        """
        price = df["close"].iloc[-1]
        if price > 1:
            price = round(price, 3)
        else:
            price = round(price, 9)
        return price

    def create_plot(
        self, dfs: dict, point: list, levels: list
    ):  # pylint: disable=R0912, R0915
        """
        Create the plot

        Parameters
        ----------
        dfs : dict
            The dictionary containing price information about ticker.
        point : list
            List containing the information about signal (ticker, timeframe, type, etc.)
        levels : list
            The list of price levels.
        """
        # get necessary info
        (
            ticker,
            timeframe,
            point_index,
            point_type,
            _,
            pattern,
            _,
            _,
            statistics,
            _,
        ) = point
        df_working = dfs[ticker][self.working_timeframe]["data"][point_type]
        df_working = df_working.loc[point_index - self.plot_width : point_index]  # noqa
        ohlc = df_working[["time", "open", "high", "low", "close", "volume"]].set_index(
            "time"
        )
        # get indicator list
        indicator_list = [
            p for p in pattern.split("_") if p[0] not in self.level_indicators
        ]
        indicator_list_tmp = indicator_list.copy()

        # check if PumpDump indicator is in indicator list to make a special plot
        if "Volume24" in indicator_list_tmp:
            indicator_list.remove("Volume24")
        if "PumpDump" in indicator_list_tmp:
            plot_num = len(indicator_list) + 1
            indicator_list.remove("PumpDump")
            main_candleplot_ratio = 1.0
            plot_width_mult = 2.0
        elif "HighVolume" in indicator_list_tmp:
            plot_num = len(indicator_list) + 1
            main_candleplot_ratio = 3.0
            plot_width_mult = 5.0
        elif "MACD" in indicator_list_tmp:
            plot_num = len(indicator_list) + 2
            main_candleplot_ratio = 1.0
            plot_width_mult = 2.5
        elif "Pattern" in indicator_list_tmp:
            plot_num = len(indicator_list) + 1
            main_candleplot_ratio = 1.5
            plot_width_mult = 2.5
        else:
            plot_num = len(indicator_list) + 1
            main_candleplot_ratio = 3.0
            plot_width_mult = 1.8

        # Plot signals
        # make subfigs
        fig = plt.figure(
            constrained_layout=True,
            figsize=(plot_width_mult * (plot_num + 1), 3 * (plot_num + 1)),
        )
        fig.patch.set_facecolor(self.background_color)
        # add this to fix incorrect aspect ratio for Pump-Dump + Trend signal
        if "PumpDump" in indicator_list_tmp or "MACD" in indicator_list_tmp:
            plot_num -= 1
        # If linear regression is in indicator list -
        # remove it from list and plot one more plot with higher timeframe
        # candles and linear regression indicator
        if "Pattern" in indicator_list_tmp:
            indicator_columns = []
            for indicator in indicator_list:
                indicator_columns += self.indicator_dict[indicator]
            plot_num -= 1
            subfigs_num = 2
            # indicator_tmp = 'Pattern'
            # indicator_list.remove(indicator_tmp)
            subfigs = fig.subfigures(subfigs_num, 1, wspace=0, height_ratios=[2, 2])

            # plot higher timeframe with linear regression
            subfigs[0].patch.set_facecolor(self.background_color)
            axs_higher = subfigs[0].subplots(2, 1, sharex=True)
            df_higher = dfs[ticker][self.higher_timeframe]["data"][point_type]
            # get corresponding to signal_time index of dataframes
            # with working and higher timeframe candles
            df_working = df_working.reset_index(drop=True)
            df_higher = df_higher.loc[
                max(df_higher.shape[0] - self.plot_width, 0) :  # noqa
            ].reset_index(drop=True)
            ohlc = df_working[
                ["time", "open", "high", "low", "close", "volume"]
            ].set_index("time")
            # plot pattern indicators
            for i_c in indicator_columns:
                if i_c == "high_max":
                    point_max = df_working.loc[df_working["high_max"] > 0, "high"]
                    axs_higher[0].scatter(
                        point_max.index, point_max.values, s=30, color="blue"
                    )
                elif i_c == "low_min":
                    point_min = df_working.loc[df_working["low_min"] > 0, "low"]
                    axs_higher[0].scatter(
                        point_min.index, point_min.values, s=30, color="orange"
                    )
                else:
                    axs_higher[1].plot(df_higher[i_c], linewidth=2)
                    axs_higher[1].yaxis.set_label_position("right")
                    axs_higher[1].yaxis.tick_right()
            # plot grid for candles
            axs_higher[0].grid(which="both", linestyle="--", linewidth=0.3)
            # set ticker color
            axs_higher[0].tick_params(axis="x", colors=self.ticker_color)
            axs_higher[0].tick_params(axis="y", colors=self.ticker_color)
            # set background color
            axs_higher[0].patch.set_facecolor(self.background_color)
            # set border color
            axs_higher[0].spines["bottom"].set_color(self.border_color)
            axs_higher[0].spines["top"].set_color(self.border_color)
            axs_higher[0].spines["right"].set_color(self.border_color)
            axs_higher[0].spines["left"].set_color(self.border_color)
            # plot grid for the trend force
            axs_higher[1].grid(which="both", linestyle="--", linewidth=0.3)
            # set ticker color
            axs_higher[1].tick_params(axis="x", colors=self.ticker_color)
            axs_higher[1].tick_params(axis="y", colors=self.ticker_color)
            # set background color
            axs_higher[1].patch.set_facecolor(self.background_color)
            # set border color
            axs_higher[1].spines["bottom"].set_color(self.border_color)
            axs_higher[1].spines["top"].set_color(self.border_color)
            axs_higher[1].spines["right"].set_color(self.border_color)
            axs_higher[1].spines["left"].set_color(self.border_color)
            # plot titles
            price = self.round_price(df_working)
            axs_higher[0].set_title(
                f"{self.process_ticker(ticker)} - {timeframe} - ${price} - "
                f'{df_working["time"].iloc[-1].date().strftime("%d.%m.%Y")}',
                fontsize=14,
                color=self.ticker_color,
            )
            axs_higher[1].set_title(
                "Trend Force / Сила тренда", fontsize=14, color=self.ticker_color
            )
            # plot candles
            mpf.plot(
                ohlc,
                type="candle",
                ax=axs_higher[0],
                warn_too_much_data=1001,
                style="yahoo",
                ylabel="",
                returnfig=True,
            )

            # workaround to show the last xtick for the higher timeframe candle plot
            format_ = "%m-%d-%H:%M"
            newxticks = []
            newlabels = []

            # copy and format the existing xticks:
            for xt in axs_higher[0].get_xticks():
                p = int(xt)
                # if 0 <= p < len(ohlc_higher):
                if 0 <= p < len(ohlc):
                    # ts = ohlc_higher.index[p]
                    ts = ohlc.index[p]
                    newxticks.append(p)
                    newlabels.append(ts.strftime(format_))

            # Here we create the final tick and tick label:
            newxticks.append(len(ohlc) - 1)
            newlabels.append(ohlc.index[len(ohlc) - 1].strftime(format_))

            # set the xticks and labels with the new ticks and labels:
            axs_higher[1].set_xticks(newxticks)
            axs_higher[1].set_xticklabels(newlabels, rotation=30)

            # plot point of trade
            # self.plot_point(point_type, df_higher, axs_higher[0], higher=True)
            self.plot_point(point_type, df_working, axs_higher[0], higher=False)
        elif "Trend" in indicator_list_tmp or "MACD" in indicator_list_tmp:
            plot_num -= 1
            subfigs_num = 3
            if "Trend" in indicator_list_tmp:
                indicator_tmp = "Trend"
            else:
                indicator_tmp = "MACD"
            indicator_list.remove(indicator_tmp)

            subfigs = fig.subfigures(
                subfigs_num,
                1,
                wspace=0,
                height_ratios=[main_candleplot_ratio, 2.5, 2.5],
            )
            # plot higher timeframe with linear regression
            subfigs[1].patch.set_facecolor(self.background_color)
            axs_higher = subfigs[1].subplots(2, 1, sharex=True)
            df_higher = dfs[ticker][self.higher_timeframe]["data"][point_type]
            # get corresponding to signal_time index
            # of dataframe with higher timeframe candles
            df_higher = df_higher.loc[
                max(df_higher.shape[0] - self.plot_width * 2, 0) :  # noqa
            ].reset_index(drop=True)
            ohlc_higher = df_higher[
                ["time", "open", "high", "low", "close", "volume"]
            ].set_index("time")
            # plot higher timeframe indicator
            indicator_columns = self.indicator_dict[indicator_tmp]
            for i_c in indicator_columns:
                if i_c == "macdhist":
                    df_higher[i_c].plot(kind="bar", ax=axs_higher[1])
                    plt.xticks(
                        np.arange(min(df_higher.index), max(df_higher.index) + 1, 10)
                    )
                    axs_higher[1].yaxis.set_label_position("right")
                    axs_higher[1].yaxis.tick_right()
                else:
                    axs_higher[1].plot(df_higher[i_c], linewidth=2)
                    axs_higher[1].yaxis.set_label_position("right")
                    axs_higher[1].yaxis.tick_right()
            # plot grid for candles
            axs_higher[0].grid(which="both", linestyle="--", linewidth=0.3)
            # set ticker color
            axs_higher[0].tick_params(axis="x", colors=self.ticker_color)
            axs_higher[0].tick_params(axis="y", colors=self.ticker_color)
            # set background color
            axs_higher[0].patch.set_facecolor(self.background_color)
            # set border color
            axs_higher[0].spines["bottom"].set_color(self.border_color)
            axs_higher[0].spines["top"].set_color(self.border_color)
            axs_higher[0].spines["right"].set_color(self.border_color)
            axs_higher[0].spines["left"].set_color(self.border_color)
            # plot grid for the trend force
            axs_higher[1].grid(which="both", linestyle="--", linewidth=0.3)
            # set ticker color
            axs_higher[1].tick_params(axis="x", colors=self.ticker_color)
            axs_higher[1].tick_params(axis="y", colors=self.ticker_color)
            # set background color
            axs_higher[1].patch.set_facecolor(self.background_color)
            # set border color
            axs_higher[1].spines["bottom"].set_color(self.border_color)
            axs_higher[1].spines["top"].set_color(self.border_color)
            axs_higher[1].spines["right"].set_color(self.border_color)
            axs_higher[1].spines["left"].set_color(self.border_color)
            # plot titles
            if indicator_tmp in ("Trend", "MACD"):
                axs_higher[0].set_title(
                    f"{self.process_ticker(ticker)} - "
                    f"{self.higher_timeframe} - Trend / Тренд",
                    fontsize=14,
                    color=self.ticker_color,
                )
                axs_higher[1].set_title(
                    "Сила тренда", fontsize=14, color=self.ticker_color
                )
            # plot candles
            mpf.plot(
                ohlc_higher,
                type="candle",
                ax=axs_higher[0],
                warn_too_much_data=1001,
                style="yahoo",
                ylabel="",
                returnfig=True,
            )

            # workaround to show the last xtick for the higher timeframe candle plot
            format_ = "%m-%d-%H:%M"
            newxticks = []
            newlabels = []

            # copy and format the existing xticks:
            for xt in axs_higher[0].get_xticks():
                p = int(xt)
                if 0 <= p < len(ohlc_higher):
                    ts = ohlc_higher.index[p]
                    newxticks.append(p)
                    newlabels.append(ts.strftime(format_))

            # Here we create the final tick and tick label:
            newxticks.append(len(ohlc_higher) - 1)
            newlabels.append(ohlc_higher.index[len(ohlc_higher) - 1].strftime(format_))

            # set the xticks and labels with the new ticks and labels:
            axs_higher[1].set_xticks(newxticks)
            axs_higher[1].set_xticklabels(newlabels, rotation=30)
        elif "HighVolume" in indicator_list_tmp:
            subfigs_num = 2
            subfigs = fig.subfigures(
                subfigs_num,
                1,
                wspace=0,
                height_ratios=[main_candleplot_ratio / 3, 0.01],
            )
        else:
            subfigs_num = 2
            subfigs = fig.subfigures(
                subfigs_num, 1, wspace=0, height_ratios=[main_candleplot_ratio, 2.5]
            )

        subfigs[0].patch.set_facecolor(self.background_color)
        subfigs[-1].patch.set_facecolor(self.background_color)
        ap = []

        # plot candles
        if "Pattern" not in indicator_list_tmp:
            # make subplots
            axs1 = subfigs[0].subplots(plot_num, 1, sharex=True)

            try:
                axs1_0 = axs1[0]
            except TypeError:
                axs1_0 = axs1

            for index, indicator in enumerate(indicator_list):
                # plot indicator
                indicator_columns = self.indicator_dict[indicator]
                for i_c in indicator_columns:
                    if i_c == "volume":
                        m = mpf.make_addplot(
                            df_working[i_c],
                            panel=index + 1,
                            title=indicator,
                            ax=axs1[index + 1],
                            width=0.5,
                            type="bar",
                        )
                    else:
                        m = mpf.make_addplot(
                            df_working[i_c],
                            panel=index + 1,
                            title=indicator,
                            ax=axs1[index + 1],
                            width=2,
                        )
                    ap.append(m)
                # plot indicator parameters
                self.plot_indicator_parameters(point_type, index, axs1, indicator_list)
                # plot y-labels from right side
                axs1[index + 1].yaxis.set_label_position("right")
                axs1[index + 1].yaxis.tick_right()
                # plot grid
                axs1[index + 1].grid(which="both", linestyle="--", linewidth=0.3)
                # set title
                axs1[index + 1].set_title(
                    indicator, fontsize=14, color=self.ticker_color
                )
                # set ticker color
                axs1[index + 1].tick_params(axis="x", colors=self.ticker_color)
                axs1[index + 1].tick_params(axis="y", colors=self.ticker_color)
                # set background color
                axs1[index + 1].patch.set_facecolor(self.background_color)
                # set border color
                axs1[index + 1].spines["bottom"].set_color(self.border_color)
                axs1[index + 1].spines["top"].set_color(self.border_color)
                axs1[index + 1].spines["right"].set_color(self.border_color)
                axs1[index + 1].spines["left"].set_color(self.border_color)

            # plot candles
            axs1_0.grid(which="both", linestyle="--", linewidth=0.3)
            # set ticker color
            axs1_0.tick_params(axis="x", colors=self.ticker_color)
            axs1_0.tick_params(axis="y", colors=self.ticker_color)
            # set background color
            axs1_0.patch.set_facecolor(self.background_color)
            # set border color
            axs1_0.spines["bottom"].set_color(self.border_color)
            axs1_0.spines["top"].set_color(self.border_color)
            axs1_0.spines["right"].set_color(self.border_color)
            axs1_0.spines["left"].set_color(self.border_color)

            # plot all subplots
            mpf.plot(
                ohlc,
                type="candle",
                ax=axs1_0,
                addplot=ap,
                warn_too_much_data=1001,
                style="yahoo",
                ylabel="",
                returnfig=True,
                tz_localize=True,
            )

            # workaround to show the last xtick for the higher timeframe candle plot
            format_ = "%H:%M"
            newxticks = []
            newlabels = []

            # copy and format the existing xticks:
            for xt in axs1_0.get_xticks():
                p = int(xt)
                if 0 <= p < len(ohlc):
                    ts = ohlc.index[p]
                    newxticks.append(p)
                    newlabels.append(ts.strftime(format_))

            # here we create the final tick and tick label:
            newxticks.append(len(ohlc) - 1)
            newlabels.append(ohlc.index[len(ohlc) - 1].strftime(format_))

            # set the xticks and labels with the new ticks and labels:
            axs1_0.set_xticks(newxticks)
            axs1_0.set_xticklabels(newlabels, rotation=0)

            # plot titles
            price = self.round_price(df_working)
            axs1_0.set_title(
                f"{self.process_ticker(ticker)} - {timeframe} - ${price} - "
                f'{df_working["time"].iloc[-1].date().strftime("%d.%m.%Y")}',
                fontsize=14,
                color=self.ticker_color,
            )
            for index, indicator in enumerate(indicator_list):
                axs1[index + 1].set_title(
                    indicator, fontsize=14, color=self.ticker_color
                )

            # plot point of trade
            self.plot_point(point_type, df_working, axs1_0)
            # plot levels
            # self.plot_levels(data, levels, axs1)

        # Plot signal statistics
        if "HighVolume" not in indicator_list_tmp:
            e_ratio_coef = [s[0] for s in statistics[0]]
            mar_coef = [np.clip(s[1], -20, 20) for s in statistics[0]]
            mar_coef_std = [min(s[2], 20) for s in statistics[0]]
            mar_coef_plus_std = [a + b for a, b in zip(mar_coef, mar_coef_std)]
            mar_coef_minus_std = [a - b for a, b in zip(mar_coef, mar_coef_std)]

            # get previous percent of right forecast and
            # save current percent to statistics dictionary
            avg_e_ratio_coef = round(sum(e_ratio_coef) / len(e_ratio_coef), 4)
            avg_mar_coef = round(sum(mar_coef) / len(mar_coef), 4)

            # get key for statistics dict
            key = self.get_statistics_dict_key(pattern)

            # get previous mean percent of right forecasts and
            # check if pattern and trade type are in statistics dictionary
            prev_avg_e_ratio_coef = self.get_prev_avg_e_ratio_coef(
                key, point_type, avg_e_ratio_coef
            )
            prev_avg_mar_coef = self.get_prev_avg_mar_coef(
                key, point_type, avg_mar_coef
            )

            # get change of statistics
            e_ratio_stat_change = self.statistics_change(
                prev_avg_e_ratio_coef, avg_e_ratio_coef
            )
            mar_stat_change = self.statistics_change(prev_avg_mar_coef, avg_mar_coef)

            # make subplots
            axs2 = subfigs[-1].subplots(2, 1, sharex=True)

            # make plots
            axs2[0].plot(e_ratio_coef, linewidth=2, color=self.stat_color_1)
            axs2[0].axhline(y=1, color="white", linestyle="--", linewidth=1.5)
            axs2[0].yaxis.set_label_position("right")
            axs2[0].yaxis.tick_right()
            axs2[1].plot(
                mar_coef_plus_std,
                linewidth=1.5,
                linestyle="dotted",
                color=self.stat_std_color_1,
            )
            axs2[1].plot(
                mar_coef_minus_std,
                linewidth=1.5,
                linestyle="dotted",
                color=self.stat_std_color_2,
            )
            axs2[1].plot(mar_coef, linewidth=2, color=self.stat_color_2)
            axs2[1].axhline(y=0, color="white", linestyle="--", linewidth=1.5)
            axs2[1].yaxis.set_label_position("right")
            axs2[1].yaxis.tick_right()
            # plot grid
            axs2[0].grid(which="both", linestyle="--", linewidth=0.3)
            axs2[1].grid(which="both", linestyle="--", linewidth=0.3)

            # set title
            if point_type == "buy":
                title = (
                    "\nBUY SIGNAL STATISTICS / СTATИСТИКА СИГНАЛА НА ПОКУПКУ "
                    "\n(last 48 hours / за последние 48 часов)"
                )
                pro_trade = "buy / покупка"
                counter_trade = "sell / продажа"
            else:
                title = (
                    "\nSELL SIGNAL STATISTIC / СTATИСТИКА СИГНАЛА НА ПРОДАЖУ "
                    "\n(last 48 hours / за последние 48 часов)"
                )
                pro_trade = "sell / продажа"
                counter_trade = "buy / покупка"
            axs2[0].set_title(
                f"{title}\n\nE-ratio coefficient / Коэффициент E-ratio\n "
                f"E-ratio > 1 - {pro_trade}, E-ratio < 1 - {counter_trade}\n "
                f"average / в среднем: {avg_e_ratio_coef} {e_ratio_stat_change}",
                fontsize=13,
                color=self.ticker_color,
            )
            axs2[1].set_title(
                "MoR coefficient / Коэффициент MoR\n "
                "MoR > 0 - buy / покупка, MoR < 0 - sell / продажа\n "
                f"average / в среднем: {avg_mar_coef} {mar_stat_change}",
                fontsize=13,
                color=self.ticker_color,
            )

            # set x-ticks
            if "MACD" in indicator_list_tmp:  # or 'Pattern' in indicator_list_tmp:
                xticklabels = [
                    "8",
                    "16",
                    "24",
                    "32",
                    "40",
                    "48",
                    "56",
                    "64",
                    "72",
                    "80",
                    "88",
                    "96",
                ]
            else:
                xticklabels = [
                    "2",
                    "4",
                    "6",
                    "8",
                    "10",
                    "12",
                    "14",
                    "16",
                    "18",
                    "20",
                    "22",
                    "24",
                ]
            # set ticker color
            axs2[0].tick_params(axis="x", colors=self.ticker_color)
            axs2[0].tick_params(axis="y", colors=self.ticker_color)
            # set background color
            axs2[0].patch.set_facecolor(self.background_color)
            # set border color
            axs2[0].spines["bottom"].set_color(self.border_color)
            axs2[0].spines["top"].set_color(self.border_color)
            axs2[0].spines["right"].set_color(self.border_color)
            axs2[0].spines["left"].set_color(self.border_color)

            axs2[1].set_xticks(np.arange(1, 25, 2))
            axs2[1].set_xticklabels(xticklabels)

            # set x-labels
            axs2[1].set_xlabel(
                "time after signal, hours / время после сигнала, в часах",
                fontsize=12,
                color=self.ticker_color,
            )

            # set y-labels
            axs2[0].set_ylabel("E-ratio", fontsize=9.5, color=self.ticker_color)
            axs2[1].set_ylabel("MoR, %", fontsize=9.5, color=self.ticker_color)

            # set ticker color
            axs2[1].tick_params(axis="x", colors=self.ticker_color)
            axs2[1].tick_params(axis="y", colors=self.ticker_color)

            # set background color
            axs2[1].patch.set_facecolor(self.background_color)

            # set border color
            axs2[1].spines["bottom"].set_color(self.border_color)
            axs2[1].spines["top"].set_color(self.border_color)
            axs2[1].spines["right"].set_color(self.border_color)
            axs2[1].spines["left"].set_color(self.border_color)

        # save plot to file
        filename = self.save_plot(ticker, timeframe, pattern, df_working)

        # close figure
        plt.close()

        return filename


if __name__ == "__main__":
    from matplotlib import style

    matplotlib.use("Agg")
    style.use("dark_background")
    rcParams["font.family"] = "URW Gothic"
    print({f.name for f in matplotlib.font_manager.fontManager.ttflist})
