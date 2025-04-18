"""
This module provides functionality for collecting, processing and predicting
trading signals for various crypto exchanges like Binance, Bybit, OKEX, MEXC,
etc.
"""

import json
import multiprocessing
import os
from datetime import datetime
from os import environ
from typing import List, Tuple, Union

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from config.config import ConfigFactory
from data.get_data import DataFactory, GetData
from indicators.indicators import IndicatorFactory
from loguru import logger
from ml.inference import Model
from signal_stat.signal_stat import SignalStat
from signals.find_signal import FindSignal
from telegram_api.telegram_api import TelegramBot

# Get configs
configs_ = ConfigFactory.factory(environ).configs

# here we load environment variables from .env, must be called before init. class
load_dotenv(find_dotenv("../.env"), verbose=True)

# load Telegram token
env = os.getenv("ENV")
if env in ["debug", "optimize"]:
    telegram_token = os.getenv("TELEGRAM_TOKEN_DEBUG")
else:
    telegram_token = os.getenv("TELEGRAM_TOKEN")


class SigBot:
    """
    Class with methods for collecting and processing trading signals

    Parameters
    ----------
    main_class
        Link to the main class from file main.py.
    load_tickers
        Flag that shows if tickers is needed to be loaded.
        During optimzation process it's should be set to False.
    opt_type
        Flag that shows if class is used in optimization mode or not.
        If it's used in optimization mode
        than some class instances aren't need to be initialized
        (e.g. model for prediction)
    configs
        Dictionary of configs which is loaded from file config/config_*env*.json
    """

    def __init__(
        self,
        main_class,
        load_tickers: bool = True,
        opt_type: Union[str, None] = None,
        **configs,
    ):
        self.opt_type = opt_type
        self.configs = configs
        # Get main bot class
        self.main = main_class
        # Create statistics class
        self.stat = SignalStat(opt_type=opt_type, **configs)
        # Create find signal class
        self.find_signal_buy = FindSignal("buy", configs)
        self.find_signal_sell = FindSignal("sell", configs)
        # List that is used to avoid processing of ticker
        # that was already processed before
        self.used_tickers: List[str] = []
        # Get working and higher timeframes
        self.work_timeframe = configs["Timeframes"]["work_timeframe"]
        self.higher_timeframe = configs["Timeframes"]["higher_timeframe"]
        self.timeframes = [self.higher_timeframe, self.work_timeframe]
        self.higher_timeframe_hours = configs["Timeframes"]["higher_timeframe_hours"]
        self.higher_tf_indicator_list = configs["Higher_TF_indicator_list"]
        self.higher_tf_indicator_set = {i for i in self.higher_tf_indicator_list if i != "Trend"}
        # List of Futures Exchanges
        self.futures_exchanges = configs["Exchanges"]["futures_exchanges"]
        # Create indicators
        self.higher_tf_indicators, self.work_tf_indicators = self._create_indicators(configs)
        # Dict of available exchanges and their corresponding tickers
        self.exchanges = {
            "ByBitPerpetual": {
                "API": GetData(**configs),
                "tickers": [],
                "all_tickers": [],
            },
            "BinanceFutures": {
                "API": GetData(**configs),
                "tickers": [],
                "all_tickers": [],
            },
            "MEXCFutures": {
                "API": GetData(**configs),
                "tickers": [],
                "all_tickers": [],
            },
            "OKEXSwap": {"API": GetData(**configs), "tickers": [], "all_tickers": []},
            "Binance": {"API": GetData(**configs), "tickers": [], "all_tickers": []},
            "ByBit": {"API": GetData(**configs), "tickers": [], "all_tickers": []},
            "MEXC": {"API": GetData(**configs), "tickers": [], "all_tickers": []},
            "OKEX": {"API": GetData(**configs), "tickers": [], "all_tickers": []},
        }
        self.max_prev_candle_limit = configs["Signal_params"]["params"]["max_prev_candle_limit"]
        # Get API and ticker list for every exchange in the list
        if load_tickers:
            # Load statistics
            buy_stat, sell_stat = self.stat.load_statistics()
            self.database = {"stat": {"buy": buy_stat, "sell": sell_stat}}
            # Load tickers
            self.get_api_and_tickers()
            # Start Telegram bot
            self.trade_exchange = self.exchanges["ByBitPerpetual"]["API"]
            self.trade_mode = multiprocessing.Array("i", range(1))
            locker = multiprocessing.Lock()
            self.telegram_bot = TelegramBot(
                token=telegram_token,
                database=self.database,
                trade_mode=self.trade_mode,
                locker=locker,
                **configs,
            )
            # run polling in the separate process
            self.telegram_bot_process = multiprocessing.Process(target=self.telegram_bot.polling)
            self.telegram_bot_process.start()
        else:
            buy_stat = pd.DataFrame(columns=["time", "ticker", "timeframe", "pattern"])
            sell_stat = pd.DataFrame(columns=["time", "ticker", "timeframe", "pattern"])
            self.database = {"stat": {"buy": buy_stat, "sell": sell_stat}}
        # Set candle range in which signal stat update can happen
        self.stat_update_range = configs["SignalStat"]["params"]["stat_range"] + 1
        # Lists for storing exchange monitor threads (Spot and Futures)
        self.spot_ex_monitor_list: List[MonitorExchange] = []
        self.fut_ex_monitor_list: List[MonitorExchange] = []
        # dictionary that is used to determine
        # too late signals according to current work_timeframe
        self.timeframe_div = configs["Data"]["Basic"]["params"]["timeframe_div"]
        # indicators of BTC dominance
        self.btcd, self.btcdom = None, None
        # model for price prediction
        if not self.opt_type:
            self.model = Model(**configs)
        # model prediction threshold
        self.pred_thresh = configs["Model"]["params"]["pred_thresh"]

    def get_api_and_tickers(self) -> None:
        """Get API and ticker list for every exchange in the exchange list"""
        exchange_list = list(self.exchanges.keys())
        for i, ex in enumerate(exchange_list):
            # get exchange API
            exchange_api = DataFactory.factory(ex, **self.configs)
            self.exchanges[ex]["API"] = exchange_api
            # get ticker list
            try:
                tickers, ticker_vols, all_tickers = self.exchanges[ex]["API"].get_tickers()
            except Exception as exc:
                del self.exchanges[ex]
                logger.exception(
                    f"Catch an exception while accessing" f"the exchange {ex}. \nException is {exc}"
                )
                continue
            # filter tickers that were used by previous exchanges
            tickers, ticker_vols = self._filter_used_tickers(tickers, ticker_vols)
            # create dictionary to store info for each ticker (volume, funding, etc.)
            self.exchanges[ex]["tickers"] = {
                tickers[i]: [float(ticker_vols[i])] for i in range(len(tickers))
            }
            # create list of all available tickers
            self.exchanges[ex]["all_tickers"] = all_tickers
            # fill ticker dict of exchange API with tickers to store current time
            # for periodic updates of ticker information
            self.exchanges[ex]["API"].fill_ticker_dict(tickers)

    def _filter_used_tickers(self, tickers: list, ticker_vols: list) -> Tuple[list, list]:
        """
        Check if ticker was already used by previous exchange
        and add only unused tickers.

        Parameters
        ----------
        tickers
            List of tickers from the current exchange.
        ticker_vols
            List of corresponding ticker volumes.

        Returns
        -------
        not_used_tickers
            Lists of tickers that were not used by previous exchanges.
        not_used_ticker_vols
            Lists of ticker corresponding volumes that were not used
            by previous exchanges.
        """
        # create list of cleaned tickers from previous exchange
        not_used_tickers = []
        not_used_ticker_vols = []
        for ticker, ticker_vol in zip(tickers, ticker_vols):
            orig_ticker = ticker
            ticker = self.delete_redundant_symbols_from_ticker(ticker)[:-4]
            # if tickers is not used - add it to current exchange list
            if ticker not in self.used_tickers:
                self.used_tickers.append(ticker)
                not_used_tickers.append(orig_ticker)
                not_used_ticker_vols.append(ticker_vol)
        return not_used_tickers, not_used_ticker_vols

    def get_new_data(
        self, exchange_api: GetData, ticker: str, timeframe: str, dt_now: datetime
    ) -> Tuple[pd.DataFrame, int]:
        """
        Check if new data appeared. If it is - return dataframe with
        the new data and amount of data

        Parameters
        ----------
        exchange_api
            API class for current exchange access.
        ticker
            Name of ticker (e.g. BTCUSDT, ETHUSDT).
        timeframe
            Time frame value (e.g. 5m, 1h, 4h, 1d).
        dt_now
            Current datetime value.

        Returns
        -------
        df
            Dataframe that contains the historical data.
        data_qty
            Amount of these historical data.
        """
        df = (
            self.database.get(ticker, {})
            .get(timeframe, {})
            .get("data", pd.DataFrame())
            .get("buy", pd.DataFrame())
        )
        # Write data to the dataframe
        df, data_qty = exchange_api.get_data(df, ticker, timeframe, dt_now)
        return df, data_qty

    @staticmethod
    def _get_btc_dominance(exchange_api: GetData) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get two types of BTC dominance indicators

        Parameters
        ----------
        exchange_api
            Class for current exchange access through API.

        Returns
        -------
        btcd
            Dataframe that contains the indicator of BTC dominance of type 1 (CryptoCap)
        btcdom
            Dataframe that contains the indicator of BTC dominance of type 2 (Binance)
        """
        btcd, btcdom = exchange_api.get_btc_dom()
        return btcd, btcdom

    def get_historical_data(
        self,
        exchange_api: GetData,
        ticker: str,
        timeframe: str,
        min_time: Union[datetime, None],
    ) -> Tuple[pd.DataFrame, int]:
        """
        Collect historical candle data from min_time and until now

        Parameters
        ----------
        exchange_api
            API class for current exchange access.
        ticker
            Name of ticker (e.g. BTCUSDT, ETHUSDT).
        timeframe
            Time frame value (e.g. 5m, 1h, 4h, 1d).
        min_time
            Time before which data will be collected.

        Returns
        -------
        df
            Dataframe that contains the historical data.
        data_qty
            Amount of these historical data.
        """
        df = (
            self.database.get(ticker, {})
            .get(timeframe, {})
            .get("data", pd.DataFrame())
            .get("buy", pd.DataFrame())
        )
        # Write data to the dataframe
        df, data_qty = exchange_api.get_hist_data(df, ticker, timeframe, min_time)
        return df, data_qty

    def _create_indicators(self, configs: dict) -> Tuple[list, list]:
        """
        Create indicators list for higher and working timeframes

        Parameters
        ----------
        configs
            Configuration dictionary.

        Returns
        -------
        higher_tf_indicators
            List of higher timeframe indicators.
        working_tf_indicators
            list of working timeframe indicators.
        """
        higher_tf_indicators = []
        working_tf_indicators = []
        indicator_list = configs["Indicator_list"]
        # get indicators for higher timeframe
        for ttype in ["buy", "sell"]:
            ind_factory = IndicatorFactory.factory("ATR", ttype, configs)
            higher_tf_indicators.append(ind_factory)
            working_tf_indicators.append(ind_factory)

            for indicator in self.higher_tf_indicator_list:
                ind_factory = IndicatorFactory.factory(indicator, ttype, configs)
                if ind_factory:
                    higher_tf_indicators.append(ind_factory)
            # get indicators for working timeframe
            for indicator in indicator_list:
                if ttype == "sell" and indicator == "HighVolume":
                    continue
                ind_factory = IndicatorFactory.factory(indicator, ttype, configs)
                if ind_factory and indicator not in self.higher_tf_indicator_list:
                    working_tf_indicators.append(ind_factory)
        return higher_tf_indicators, working_tf_indicators

    def add_indicators(
        self,
        df: pd.DataFrame,
        ttype: str,
        ticker: str,
        timeframe: str,
        exchange_data: dict,
        data_qty: int,
        opt_flag: bool = False,
    ) -> Tuple[dict, int]:
        """
        Create indicators and add them to data

        Parameters
        ----------
        df
            Dataframe with candles.
        ttype
            Type of trade (buy or sell).
        ticker
            Name of ticker (e.g. BTCUSDT, ETHUSDT).
        timeframe
            Time frame value (e.g. 5m, 1h, 4h, 1d).
        exchange_data
            Dictionary that stores link to exchange API and names
            of all tickers from that exchange.
        data_qty
            Amount of data to which indicators will be added.
        opt_flag
            Flag that is used in optimization process.
            If it's True than Pattern indicator doesn't need to be added,
            if it's False - Pattern indicator will be added.
            This flag is True for the first optimization iteration
            and False for any next iterations.
            This was done because Pattern indicator doesn't have parameters
            and thus doesn't need to be optimized,
            so we may add it only one time during the first iteration.

        Returns
        -------
        database
            List of higher timeframe indicators, list of working timeframe indicators.
        data_qty
            Amount of these historical data.
        """
        if timeframe == self.work_timeframe:
            indicators = self.work_tf_indicators
        else:
            indicators = self.higher_tf_indicators
        # Write indicators to the dataframe, update dataframe dict
        exchange_api = exchange_data["API"]
        database = exchange_api.add_indicator_data(
            self.database, df, ttype, indicators, ticker, timeframe, data_qty, opt_flag
        )
        # If enough time has passed - update statistics
        if data_qty > 1 and self.main.cycle_number > 1 and timeframe == self.work_timeframe:
            data_qty = self.stat_update_range * 1.5
        return database, data_qty

    def get_buy_signals(
        self, ticker: str, timeframe: str, data_qty: int, data_qty_higher: int
    ) -> list:
        """
        Try to find the buy signals and if succeed - return them

        Parameters
        ----------
        ticker
            Name of ticker (e.g. BTCUSDT, ETHUSDT).
        timeframe
            Time frame value (e.g. 5m, 1h, 4h, 1d).
        data_qty
            Amount of data from working timeframe (default is 1h)
            to which indicators will be added.
        data_qty_higher
            Amount of data from higher timeframe (default is 4h)
            to which indicators will be added.

        Returns
        -------
        sig_points_buy
        List of signal buy points.
        """
        sig_points_buy = self.find_signal_buy.find_signal(
            self.database, ticker, timeframe, data_qty, data_qty_higher
        )
        return sig_points_buy

    def get_sell_signals(
        self, ticker: str, timeframe: str, data_qty: int, data_qty_higher: int
    ) -> list:
        """
        Try to find the sell signals and if succeed - return them

        Parameters
        ----------
        ticker
            Name of ticker (e.g. BTCUSDT, ETHUSDT).
        timeframe
            Time frame value (e.g. 5m, 1h, 4h, 1d).
        data_qty
            Amount of data from working timeframe (default is 1h)
            to which indicators will be added.
        data_qty_higher
            Amount of data from higher timeframe (default is 4h)
            to which indicators will be added.

        Returns
        -------
        sig_points_sell
            List of signal sell points.
        """
        sig_points_sell = self.find_signal_sell.find_signal(
            self.database, ticker, timeframe, data_qty, data_qty_higher
        )
        return sig_points_sell

    def filter_sig_points(self, sig_points: list) -> list:
        """
        Don't add signal if relatively fresh similar signal was already
        added to the statistics dataframe before

        Parameters
        ----------
        sig_points
            List of signal points to be filtered.

        Returns
        -------
        filtered_points
            List of filtered signal points.
        """
        filtered_points = []
        prev_point = (None, None, None)
        for point in sig_points:
            ticker, timeframe, index, ttype, timestamp, pattern = (
                point[0],
                point[1],
                point[2],
                point[3],
                point[4],
                point[5],
            )
            # pattern is PumpDump - we need only its name without settings
            if str(pattern[0][0]).startswith("PumpDump"):
                pattern = str([pattern[0][0]] + pattern[1:])
            else:
                pattern = str(pattern)
            # if earlier signal is already exists in the
            # signal list - don't add one more
            stat = self.database["stat"][ttype]
            df_len = self.database[ticker][timeframe]["data"][ttype].shape[0]
            if self.stat.check_close_trades(
                stat, df_len, ticker, index, timestamp, pattern, prev_point
            ):
                filtered_points.append(point)
                prev_point = (ticker, timestamp, pattern)
        return filtered_points

    def filter_old_signals(self, sig_points: list) -> list:
        """
        Don't send Telegram notification for the old signals
        (older than 1-2 candles ago)

        Parameters
        ----------
        sig_points
            List of signal points to be filtered.

        Returns
        -------
        filtered_points
            List of filtered signal points.
        """
        filtered_points = []
        # round datetime to hours, so time spent for data loading doesn't
        # influence the time span between current moment and signal time
        dt_now = datetime.now().replace(microsecond=0, second=0, minute=0)
        for point in sig_points:
            point_time = point[4]
            indicator_list = set(point[5].split("_"))
            if indicator_list.intersection(self.higher_tf_indicator_set):
                time_span = self.timeframe_div[self.higher_timeframe] * self.max_prev_candle_limit
            else:
                time_span = self.timeframe_div[self.work_timeframe] * self.max_prev_candle_limit
            # select only new signals
            if (dt_now - point_time).total_seconds() <= time_span:
                filtered_points.append(point)
        return filtered_points

    def filter_higher_tf_signals(self, sig_points: list) -> list:
        """
        If higher tf signal was found but new higher tf candle wasn't
        closed at the time of this signal - don't add it.

        Parameters
        ----------
        sig_points
            List of signal points to be filtered.

        Returns
        -------
        filtered_points
            List of filtered signal points.
        """
        filtered_points = []
        for point in sig_points:
            point_time = point[4]
            indicator_list = set(point[5].split("_"))
            if (
                indicator_list.intersection(self.higher_tf_indicator_set)
                and point_time.hour not in self.higher_timeframe_hours
            ):
                continue
            filtered_points.append(point)
        return filtered_points

    def sb_add_statistics(self, sig_points: list, data_qty_higher=None) -> dict:
        """
        Write statistics for signal points to the database

        Parameters
        ----------
        sig_points
            List of signal points to be filtered.
        data_qty_higher
            Amount of data from higher timeframe (default is 4h)
            to which indicators will be added.

        Returns
        -------
        filtered_points
            List of filtered signal points.
        """
        database = self.stat.write_stat(self.database, sig_points, data_qty_higher)
        return database

    def sb_save_statistics(self) -> None:
        """Save statistics to the disk"""
        self.stat.save_statistics(self.database)

    def calc_statistics(self, sig_points: list) -> list:
        """
        Calculate statistics and write it for every signal in signal points list

        Parameters
        ----------
        sig_points
            List of signal points.

        Returns
        -------
        sig_points
            List of signal points with added statistics.
        """
        for sig_point in sig_points:
            sig_type = sig_point[3]
            pattern = sig_point[5]
            result_statistics, _ = self.stat.calculate_total_stat(self.database, sig_type, pattern)
            sig_point[8].append(result_statistics)
        return sig_points

    def get_exchange_list(self, ticker: str, sig_points: list) -> list:
        """
        Add list of exchanges where this ticker can be traded

        Parameters
        ----------
        ticker
            Name of ticker (e.g. BTCUSDT, ETHUSDT).
        sig_points
            List of signal points.

        Returns
        -------
        sig_points
            List of signal points with added list of exchanges where
            corresponding to signal ticker can be traded.
        """
        for sig_point in sig_points:
            for exchange, exchange_data in self.exchanges.items():
                if not ticker.startswith("SWAP"):
                    ticker = ticker.replace("SWAP", "")
                if (
                    ticker in exchange_data["all_tickers"]
                    or ticker.replace("-", "") in exchange_data["all_tickers"]
                    or ticker.replace("_", "") in exchange_data["all_tickers"]
                    or ticker.replace("/", "") in exchange_data["all_tickers"]
                ):
                    sig_point[7].append(exchange)
        return sig_points

    def _create_exchange_monitors(self) -> Tuple[list, list]:
        """
        Create two lists of instances for ticker monitoring for every exchange

        Returns
        -------
        spot_ex_monitor_list
            List of spot exchanges.
        fut_ex_monitor_list
            List of futures exchanges.
        """
        spot_ex_monitor_list = []
        fut_ex_monitor_list = []
        for exchange, exchange_data in self.exchanges.items():
            monitor = MonitorExchange(self, exchange, exchange_data)
            if (
                exchange.endswith("Futures")
                or exchange.endswith("Swap")
                or exchange.endswith("Perpetual")
            ):
                fut_ex_monitor_list.append(monitor)
            else:
                spot_ex_monitor_list.append(monitor)
        return spot_ex_monitor_list, fut_ex_monitor_list

    def save_opt_dataframes(
        self,
        load: bool = False,
        historical: bool = False,
        min_time: Union[datetime, None] = None,
    ) -> None:
        """
        Save all ticker dataframes for further indicator/signal optimization

        Parameters
        ----------
        load
            Flag that defines if candle data must be loaded from the internet
        historical
            Flag that defines if we need to load only the latest candle data (False)
            or we need to load all available candle data (True)
            from min_time and until current time.
        min_time
            Time from which historical candle data will be loaded.
        """
        self.spot_ex_monitor_list, self.fut_ex_monitor_list = self._create_exchange_monitors()
        dt_now = datetime.now()
        if load:
            logger.info("\nLoad the datasets...")
            # start all futures exchange monitors
            for monitor in self.fut_ex_monitor_list:
                monitor.mon_save_opt_dataframes(dt_now, historical, min_time)
            # start all spot exchange monitors
            for monitor in self.spot_ex_monitor_list:
                monitor.mon_save_opt_dataframes(dt_now, historical, min_time)

    def save_opt_statistics(self, ttype: str, opt_limit: int, opt_flag: bool) -> None:
        """
        Save statistics in program memory for further indicator/signal optimization

        Parameters
        ----------
        ttype
            Type of trade (buy or sell).
        opt_limit
            Amount of the last data for which we look for the signals and
            collect signal statistics.
            This parameter is added to speed up finding the signals and signal
            statistic collection.
        opt_flag
            Flag that is used in optimization process.
            If it's True than Pattern indicator doesn't need to be added,
            if it's False - Pattern indicator will be added.
            This flag is True for the first optimization iteration and
            False for any next iteration.
            This was done because Pattern indicator doesn't have parameters and thus
            doesn't need to be optimized, so we may add it only one time during the
            first iteration.
        """
        self.spot_ex_monitor_list, self.fut_ex_monitor_list = self._create_exchange_monitors()
        # start all futures exchange monitors
        for monitor in self.fut_ex_monitor_list:
            monitor.mon_save_opt_statistics(ttype, opt_limit, opt_flag)
        # start all spot exchange monitors
        for monitor in self.spot_ex_monitor_list:
            monitor.mon_save_opt_statistics(ttype, opt_limit, opt_flag)

    def add_higher_time(self, ticker: str, ttype: str) -> None:
        """
        Add time from higher timeframe to dataframe with working timeframe data

        Parameters
        ----------
        ticker
            Name of ticker (e.g. BTCUSDT, ETHUSDT).
        ttype
            Type of trade (buy or sell).
        """
        # Create signal point df for each indicator
        df_work = self.database[ticker][self.work_timeframe]["data"][ttype]
        # add signals from higher timeframe
        try:
            df_higher = self.database[ticker][self.higher_timeframe]["data"][ttype]
        except KeyError:
            return

        # merge work timeframe with higher timeframe,
        # so we can work with indicator values from higher timeframe
        higher_features = [
            "time_4h",
            "linear_reg",
            "linear_reg_angle",
            "macd",
            "macdhist",
            "macd_dir",
            "macdsignal",
            "macdsignal_dir",
        ]
        df_higher["time_4h"] = df_higher["time"] + pd.to_timedelta(3, unit="h")
        df_work[["time"] + higher_features] = pd.merge(
            df_work[["time"]],
            df_higher[higher_features],
            how="left",
            left_on="time",
            right_on="time_4h",
        )
        df_work = df_work.drop(columns=["time_4h"])
        df_higher = df_higher.drop(columns=["time_4h"])

        df_work = df_work.ffill()
        df_work = df_work.reset_index(drop=True)
        self.database[ticker][self.work_timeframe]["data"][ttype] = df_work
        self.database[ticker][self.higher_timeframe]["data"][ttype] = df_higher
        return

    def make_prediction(self, sig_points: list, exchange_name: str) -> list:
        """
        Get dataset and use ML model to make price prediction for current signal points

        Parameters
        ----------
        sig_points
            List of signal points.
        exchange_name
            Name of the current exchange. We should predict and
            trade only if it's exchange we can trade on.

        Returns
        ----------
        sig_points
            List of signal points with added ML prediction for every signal point.
        """
        (
            ticker,
            timeframe,
            _,
            ttype,
            _,
            pattern,
            _,
            _,
            _,
            _,
        ) = sig_points[0]
        df = self.database[ticker][timeframe]["data"][ttype]
        # RSI_STOCH pattern is inverted with respect to the trade sides
        if pattern == "STOCH_RSI_Volume24":
            if ttype == "buy":
                ttype = "sell"
            else:
                ttype = "buy"
        if not self.opt_type:
            sig_points = self.model.make_prediction(
                df, self.btcd, self.btcdom, sig_points, ttype, exchange_name
            )
        return sig_points

    def main_cycle(self):
        """Create and run exchange monitors"""
        self.spot_ex_monitor_list, self.fut_ex_monitor_list = self._create_exchange_monitors()
        # get BTC dominance
        for ex in self.spot_ex_monitor_list:
            if ex.exchange_data["API"].name == "Binance":
                binance_exchange_data = ex.exchange_data["API"]
                btcd, btcdom = self._get_btc_dominance(binance_exchange_data)
                # update BTC dominance info only when there are new information
                if btcd.shape[0] > 0 or (btcd.shape[0] == 0 and self.btcd is None):
                    self.btcd = btcd
                if btcdom.shape[0] > 0 or (btcdom.shape[0] == 0 and self.btcdom is None):
                    self.btcdom = btcdom
        # start all futures exchange monitors
        for monitor in self.fut_ex_monitor_list:
            monitor.run_cycle()
        # start all spot exchange monitors
        for monitor in self.spot_ex_monitor_list:
            monitor.run_cycle()

    @staticmethod
    def delete_redundant_symbols_from_ticker(ticker: str) -> str:
        """
        Delete symbols like '-' or 'SWAP' from name of the ticker

        Parameters
        ----------
        ticker
            Name of ticker.

        Returns
        ----------
        ticker
            Cleaned name of the ticker.
        """
        ticker = ticker.replace("-", "").replace("_USDT", "USDT")
        if not ticker.startswith("SWAP"):
            ticker = ticker.replace("SWAP", "")
        return ticker


class MonitorExchange:
    """Class for monitoring of signals from current exchange"""

    def __init__(self, sigbot: SigBot, exchange: str, exchange_data: dict):
        # initialize separate thread the Telegram bot, so it can work independently
        # instance of main class
        self.sigbot = sigbot
        # exchange name
        self.exchange = exchange
        # exchange data
        self.exchange_data = exchange_data

    def mon_add_indicators(
        self,
        df: pd.DataFrame,
        ttype: str,
        ticker: str,
        timeframe: str,
        data_qty: int,
        opt_flag: bool = False,
    ) -> int:
        """
        Add indicators and return quantity of data

        Parameters
        ----------
        df
            Dataframe that contains the historical data.
        ttype
            Type of trade (buy or sell).
        ticker
            Name of ticker (e.g. BTCUSDT, ETHUSDT).
        timeframe
            Time frame value (e.g. 5m, 1h, 4h, 1d).
        data_qty
            Amount of data from working timeframe (default is 1h)
            to which indicators will be added.
        opt_flag
            Flag that is used in optimization process.
            If it's True than Pattern indicator doesn't need to be added,
            if it's False - Pattern indicator will be added.
            This flag is True for the first optimization iteration and False
            for any next iteration. This was done because Pattern indicator doesn't have
            parameters and thus doesn't need to be optimized,
            so we may add it only one time during the first iteration.

        Returns
        ----------
        data_qty
            Quantity of the new data (number of candles)
        """
        self.sigbot.database, data_qty = self.sigbot.add_indicators(
            df, ttype, ticker, timeframe, self.exchange_data, data_qty, opt_flag
        )
        return data_qty

    def mon_add_statistics(
        self, sig_points: list, data_qty_higher: Union[int, None] = None
    ) -> None:
        """
        Write statistics for signal points to the database

        Parameters
        ----------
        sig_points
            List of signal points to be added to statistics.
        data_qty_higher
            Amount of data from higher timeframe (default is 4h)
            to which indicators will be added.
        """
        self.sigbot.database = self.sigbot.sb_add_statistics(sig_points, data_qty_higher)

    def mon_save_statistics(self) -> None:
        """Save statistics for every ticker"""
        self.sigbot.sb_save_statistics()

    def mon_save_opt_dataframes(
        self, dt_now: datetime, historical: bool, min_time: Union[datetime, None]
    ) -> None:
        """
        Save dataframe for every ticker for further indicator/signal optimization

        Parameters
        ----------
        dt_now
            Current datetime value.
        historical
            Flag that defines if we need to load only the latest candle data (False)
            or we need to load all available candle data (True)
            from min_time and until current time.
        min_time
            Time from which historical candle data will be loaded.
        """
        exchange_api = self.exchange_data["API"]
        tickers = self.exchange_data["tickers"]
        logger.info(80 * "=")
        logger.info(f"{self.exchange}")
        if self.exchange == "ByBitPerpetual":
            with open("model/bybit_tickers.json", "w+") as f:
                json.dump(list(tickers.keys()), f)
        for ticker in tickers:
            logger.info(ticker)
            # For every timeframe get the data and find the signal
            for timeframe in self.sigbot.timeframes:
                if historical:
                    # get historical data for some period (before min_time)
                    df, data_qty = self.sigbot.get_historical_data(
                        exchange_api, ticker, timeframe, min_time
                    )
                else:
                    df, data_qty = self.sigbot.get_new_data(exchange_api, ticker, timeframe, dt_now)
                # If we previously download this dataframe to the disk -
                # update it with new data
                if data_qty > 1:
                    tmp_ticker = self.sigbot.delete_redundant_symbols_from_ticker(ticker)
                    try:
                        tmp = pd.read_pickle(  # nosec
                            f"../optimizer/ticker_dataframes/" f"{tmp_ticker}_{timeframe}.pkl"
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        if not historical:
                            first_time = df["time"].min()
                            tmp = tmp[tmp["time"] < first_time]
                            df = pd.concat([tmp, df], ignore_index=True)
                    df_path = f"../optimizer/ticker_dataframes/{tmp_ticker}_{timeframe}.pkl"
                    df = df.drop_duplicates().reset_index(drop=True)
                    df.to_pickle(df_path)
                else:
                    break

    def mon_save_opt_statistics(self, ttype: str, opt_limit: int, opt_flag: bool) -> None:
        """
        Save statistics data for every ticker for further indicator/signal optimization

        Parameters
        ----------
        ttype
            Type of trade (buy or sell).
        opt_limit
            Amount of the last data for which we look for the signals and collect
            signal statistics. This parameter is added to speed up finding the signals
            and signal statistic collection.
        opt_flag
            Flag that is used in optimization process.
            If it's True than Pattern indicator doesn't need to be added,
            if it's False - Pattern indicator will be added.
            This flag is True for the first optimization iteration and False
            for any next iteration. This was done because Pattern indicator doesn't have
            parameters and thus doesn't need to be optimized, so we may add it only one
            time during the first iteration.
        """
        tickers = self.exchange_data["tickers"]

        for ticker in tickers:
            # For every timeframe get the data and find the signal
            for timeframe in self.sigbot.timeframes:
                if (
                    ticker not in self.sigbot.database
                    or timeframe not in self.sigbot.database[ticker]
                ):
                    try:
                        tmp_ticker = self.sigbot.delete_redundant_symbols_from_ticker(ticker)
                        df = pd.read_pickle(  # nosec
                            f"../optimizer/ticker_dataframes/" f"{tmp_ticker}_{timeframe}.pkl"
                        )
                    except FileNotFoundError:
                        continue
                else:
                    df = self.sigbot.database[ticker][timeframe]["data"][ttype].copy()
                # Add indicators
                self.mon_add_indicators(df, ttype, ticker, timeframe, 1000, opt_flag)
                # If current timeframe is working timeframe
                if timeframe == self.sigbot.work_timeframe:
                    self.sigbot.add_higher_time(ticker, ttype)
                    # Get the signals
                    if ttype == "buy":
                        sig_points = self.sigbot.find_signal_buy.find_signal(
                            self.sigbot.database,
                            ticker,
                            timeframe,
                            opt_limit,
                            data_qty_higher=2,
                        )
                    else:
                        sig_points = self.sigbot.find_signal_sell.find_signal(
                            self.sigbot.database,
                            ticker,
                            timeframe,
                            opt_limit,
                            data_qty_higher=2,
                        )
                    # Filter repeating signals
                    sig_points = self.sigbot.filter_sig_points(sig_points)
                    # Add the signals to statistics
                    self.mon_add_statistics(sig_points)
        # Save statistics
        self.mon_save_statistics()

    def run_cycle(self) -> None:
        """
        For every exchange, ticker, timeframe and indicator patterns in the
        database find the latest signals and send them to the Telegram module
        Signal point's structure:
            0 - ticker symbol
            1 - timeframe value
            2 - index in dataframe
            3 - type of signal (buy/sell)
            4 - time when signal appeared
            5 - signal pattern, by which signal was searched for
            6 - path to file with candle/indicator plots of the signal
            7 - list of exchanges where ticker with this signal can be found
            8 - statistics for the current pattern
            9 - ML model prediction
        """
        tickers = self.exchange_data["tickers"]
        dt_now = datetime.now()
        # list of processes
        processes = []
        logger.info(f"Exchange: {self.exchange}, number of tickers: {len(tickers)}")
        for ticker in tickers:
            data_qty_higher = 0
            # flag that allows to pass the ticker in case of errors
            pass_the_ticker = False
            # For every timeframe get the data and find the signal
            for timeframe in self.sigbot.timeframes:
                if pass_the_ticker:
                    continue
                df, data_qty = self.sigbot.get_new_data(
                    self.exchange_data["API"], ticker, timeframe, dt_now
                )
                if data_qty > 1:
                    if timeframe == self.sigbot.work_timeframe:
                        logger.info(
                            f"Cycle number {self.sigbot.main.cycle_number}, "
                            f"exchange {self.exchange}, "
                            f"ticker {ticker}",
                        )
                    else:
                        data_qty_higher = data_qty
                    # Get indicators and quantity of data,
                    # if catch any exception - pass the ticker
                    try:
                        data_qty_buy = self.mon_add_indicators(
                            df, "buy", ticker, timeframe, data_qty
                        )
                        data_qty_sell = self.mon_add_indicators(
                            df, "sell", ticker, timeframe, data_qty
                        )
                    except Exception as exc:
                        logger.exception(
                            f"Something bad has happened to ticker {ticker} "
                            f"on timeframe {timeframe} "
                            f"while getting the indicator data. \nException is {exc}"
                        )
                        pass_the_ticker = True
                        continue
                    # If current timeframe is working timeframe
                    if timeframe == self.sigbot.work_timeframe:
                        # Add time from higher timeframe to dataframe
                        # from working timeframe if data_qty_higher > 1:
                        for ttype in ["buy", "sell"]:
                            self.sigbot.add_higher_time(ticker, ttype)
                        # Get the signals
                        try:
                            sig_buy_points = self.sigbot.get_buy_signals(
                                ticker, timeframe, data_qty_buy, data_qty_higher
                            )
                            sig_sell_points = self.sigbot.get_sell_signals(
                                ticker, timeframe, data_qty_sell, data_qty_higher
                            )
                        except Exception as exc:
                            logger.exception(
                                f"Something bad has happened to ticker {ticker} "
                                "on timeframe {timeframe} "
                                f"while getting the signals. \nException is {exc}"
                            )
                            pass_the_ticker = True
                            continue
                        # If similar signal was added to stat dataframe not too long
                        # time ago (<= 3-5 ticks before) - don't add it again
                        sig_buy_points = self.sigbot.filter_sig_points(sig_buy_points)
                        sig_sell_points = self.sigbot.filter_sig_points(sig_sell_points)
                        # Add signals to statistics
                        self.mon_add_statistics(sig_buy_points, data_qty_higher)
                        self.mon_add_statistics(sig_sell_points, data_qty_higher)
                        # If bot cycle isn't first - calculate statistics
                        # and send Telegram notification
                        if self.sigbot.main.cycle_number > self.sigbot.main.first_cycle_qty_miss:
                            # Join buy and sell points into the one list
                            sig_points = sig_buy_points + sig_sell_points
                            # Send signals in Telegram notification
                            # only if they are fresh (<= 1-2 ticks ago)
                            sig_points = self.sigbot.filter_old_signals(sig_points)
                            # Send higher signals in Telegram notification
                            # only if their hour time is in 3, 7, 11, 15, 19, 23
                            # Add list of exchanges where this ticker is available
                            # and has a good liquidity
                            sig_points = self.sigbot.get_exchange_list(ticker, sig_points)
                            # Add pattern and ticker statistics
                            sig_points = self.sigbot.calc_statistics(sig_points)
                            # Send Telegram notification
                            if sig_points:
                                logger.info("Find the signal point(s).")
                                sig_points = self.sigbot.make_prediction(sig_points, self.exchange)
                                # if trade mode is enabled and
                                # model has made prediction - place an order
                                logger.info(
                                    self.exchange,
                                    [
                                        [
                                            sp[0],
                                            sp[1],
                                            sp[2],
                                            sp[3],
                                            sp[4],
                                            sp[5],
                                            sp[9],
                                        ]
                                        for sp in sig_points
                                    ]
                                )
                                # send Telegram notification, create separate process
                                # for each notification to run processes of signal
                                # search and signal notification simultaneously
                                for sig_point in sig_points:
                                    ticker = sig_point[0]
                                    sig_type = sig_point[3].capitalize()
                                    pattern = sig_point[5]
                                    prediction = sig_point[9]
                                    # # debug
                                    # if prediction > 0:
                                    #     time = sig_point[4]
                                    #     month = time.month
                                    #     day = time.day
                                    #     hour = time.hour
                                    #     df.to_csv(
                                    #         f"./bot/ticker_dataframes/\
                                    #         {ticker}_1h_{ttype}_{month}_{day}_{hour}.csv")
                                    #     df_higher = self.sigbot.database[ticker]\
                                    #         [self.sigbot.higher_timeframe]['data'][ttype]
                                    #     df_higher.to_csv(
                                    #         f"./bot/ticker_dataframes/{ticker}_4h_{ttype}_{month}_{day}_{hour}.csv")
                                    if (
                                        self.sigbot.trade_mode[0]
                                        and prediction > self.sigbot.pred_thresh
                                    ):
                                        if pattern == "STOCH_RSI_Volume24":
                                            # for STOCH_RSI pattern buy / sell trades are inverted
                                            sig_type = "Sell" if sig_type == "Buy" else "Buy"
                                        self.sigbot.trade_exchange.api.place_all_conditional_orders(
                                            ticker, sig_type
                                        )
                                    pr = multiprocessing.Process(
                                        target=self.sigbot.telegram_bot.send_notification,
                                        args=(sig_point,),
                                    )
                                    processes.append(pr)
                                    pr.start()
                                    # self.sigbot.telegram_bot.send_notification(sig_point)
                            # Log the signals
                            for sig_point in sig_points:
                                sig_message = (
                                    f"Exchange is {self.exchange}, ticker is "
                                    f"{ticker}, timeframe is {timeframe}, "
                                    f"type is {sig_point[3]}, "
                                    f"pattern is {sig_point[5]}, "
                                    f"time is {sig_point[4]}, "
                                    f"model confidence is {sig_point[9]}"
                                )
                                logger.info(sig_message)
        # wait until all processes finish
        for pr in processes:
            pr.join()
        # save buy and sell statistics to files
        self.mon_save_statistics()
        # find open orders (not TP / SL) that weren't
        # triggered within an hour and cancel them
        if self.exchange == "ByBitPerpetual":
            if self.sigbot.trade_mode[0]:
                self.sigbot.trade_exchange.api.find_open_orders()
                self.sigbot.trade_exchange.api.check_open_positions()
