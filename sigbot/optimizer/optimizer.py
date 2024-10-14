"""
This module provides functional for optimization and
managing configurations of technical indicators.
"""

import copy
import glob
import itertools as it
from datetime import datetime
from os import environ, remove
from typing import Union

import pandas as pd
from bot.bot import SigBot
from config.config import ConfigFactory
from tqdm import tqdm

# Get configs
configs_ = ConfigFactory.factory(environ).configs


# mock class
class Main:
    """
    A mock class to simulate the main application state.

    Attributes
    ----------
    cycle_number : int
        Tracks the number of cycles.
    """

    def __init__(self):
        self.cycle_number = 1


class Optimizer:
    """
    Class responsible for optimization and managing
    configurations of technical indicators.

    Parameters
    ----------
    pattern : list
        List of signal patterns used for optimization.
    optim_dict : dict
        Dictionary containing indicator parameters.
    clean : bool, optional
        Whether to clean previous statistics or not (default is True).
    configs : dict
        Configuration settings.
    """

    def __init__(self, pattern, optim_dict, clean=True, **configs):
        self.statistics = {}
        self.clean = clean
        self.configs = configs
        self.pattern_list = pattern
        self.optim_dict = optim_dict
        self.remove_path = optim_dict

    @staticmethod
    def clean_prev_stat(ttype: str):
        """
        Clean previous statistics files.

        Parameters
        ----------
        ttype : str
            Trade types for which statistics are being cleaned ('buy', 'sell').
        """
        files = glob.glob(f"signal_stat/{ttype}_stat*.pkl")
        for f in files:
            remove(f)

    @staticmethod
    def clean_prev_tickers_dfs():
        """Clean previous ticker market data files"""
        files = glob.glob("../optimizer/ticker_dataframes/*.pkl")
        for f in files:
            remove(f)

    def clean_dict(self, dict1: dict):
        """
        Clean optimization dictionary by removing unwanted parameters
        and keeping necessary parameters.

        Parameters
        ----------
        dict1 : dict
            Dictionary to clean.

        Returns
        -------
        dict
            Cleaned optimization dictionary with only necessary parameters.
        """
        res_dict = {}
        for key, value in dict1.items():
            if key in self.pattern_list:
                res_dict[key] = value["params"]
        for _, value in res_dict.items():
            i, vk = 0, list(value.keys())
            while i < len(vk):
                k, v = vk[i], value[vk[i]]
                if isinstance(v, str):
                    del vk[i]
                    del value[k]
                else:
                    i += 1
        return res_dict

    def merge_dicts(self, dict1: dict, dict2: dict):
        """
        Merge two cleaned dictionaries, combine values for matching keys.

        Parameters
        ----------
        dict1 : dict
            First dictionary to merge.
        dict2 : dict
            Second dictionary to merge.

        Returns
        -------
        dict
            Merged dictionary with combined values.
        """
        res_dict1 = self.clean_dict(dict1)
        res_dict2 = self.clean_dict(dict2)

        for key, value in res_dict2.items():
            if key not in res_dict1:
                res_dict1[key] = value
            else:
                res_dict1[key] = {**res_dict1[key], **res_dict2[key]}
        return res_dict1

    def get_product_dicts(self):
        """
        Generate all possible combinations of signal pattern parameters.

        Returns
        -------
        list
            List of dictionaries representing all combinations of pattern settings.
        """
        res_dict = {k: v for k, v in self.optim_dict.items() if k in self.pattern_list}
        perm_values = []
        for _, value in res_dict.items():
            keys, values = zip(*value.items())
            perm_dicts = [dict(zip(keys, v)) for v in it.product(*values)]
            perm_values.append(perm_dicts)
        product_dict = [dict(zip(res_dict.keys(), v)) for v in it.product(*perm_values)]
        return product_dict

    def set_configs(self, prod_dict: dict, ttype: str):
        """
        Set configuration values based on the provided product dictionary.

        Parameters
        ----------
        prod_dict : dict
            Dictionary containing indicator parameters.
        ttype : str
            Type of the signal or configuration.

        Returns
        -------
        dict
            Updated configuration dictionary.
        """
        confs = self.configs.copy()
        for key in confs:
            if key == "Patterns":
                confs[key] = [self.pattern_list]
            elif key == "Indicator_list":
                confs[key] = self.pattern_list + ["ATR"]
            elif key in ["Indicator", "Indicator_signal"]:
                for indicator in prod_dict.keys():
                    prod_values = prod_dict[indicator]
                    conf_values = confs[key][ttype][indicator]["params"]
                    for k, _ in conf_values.items():
                        if k in prod_values:
                            conf_values[k] = prod_values[k]
                    if indicator != "Trend":
                        if "high_bound" in conf_values:
                            conf_values["high_bound"] = 100 - conf_values["low_bound"]
                        elif "high_price_quantile" in conf_values:
                            conf_values["high_price_quantile"] = (
                                1000 - conf_values["low_price_quantile"]
                            )
        return confs

    def save_configs(self, prod_dict: dict, ttype: str):
        """
        Save updated configuration based on the provided dictionary
        with indicator parameters.

        Parameters
        ----------
        prod_dict : dict
            Dictionary containing parameters for indicators.
        ttype : str
            Type of the trade ('buy', 'sell').

        Returns
        -------
        dict
            Saved configuration dictionary.
        """
        confs = self.configs.copy()
        for key in confs:
            if key in ["Indicator", "Indicator_signal"]:
                for indicator in prod_dict.keys():
                    prod_values = prod_dict[indicator]
                    conf_values = confs[key][ttype][indicator]["params"]
                    for k, _ in conf_values.items():
                        if k in prod_values:
                            conf_values[k] = prod_values[k]
                    if indicator != "Trend":
                        if "high_bound" in conf_values:
                            conf_values["high_bound"] = 100 - conf_values["low_bound"]
                        elif "high_price_quantile" in conf_values:
                            conf_values["high_price_quantile"] = (
                                1000 - conf_values["low_price_quantile"]
                            )
        return confs

    @staticmethod
    def get_headers_from_dict(prod_dict: dict) -> list:
        """
        Retrieve headers from the dictionary with indicator parameters.

        Parameters
        ----------
        prod_dict : dict
            Dictionary containing indicator parameters.

        Returns
        -------
        list
            List of headers extracted from the dictionary.
        """
        headers = []

        def helper(prod_dict_, header):
            """Function for recursive retrieving of headers"""
            for key in prod_dict_:
                if not isinstance(prod_dict_[key], dict):
                    headers.append(header + key)
                else:
                    helper(prod_dict_[key], header + key + "_")

        helper(prod_dict, "")
        return headers

    @staticmethod
    def get_values_from_dict(prod_dict: dict) -> list:
        """
        Retrieve values from the product dictionary.

        Parameters
        ----------
        prod_dict : dict
            Dictionary containing indicator parameters.

        Returns
        -------
        list
            List of values extracted from the dictionary.
        """
        headers = []

        def helper(prod_dict_):
            for key in prod_dict_:
                if not isinstance(prod_dict_[key], dict):
                    headers.append(prod_dict_[key])
                else:
                    helper(prod_dict_[key])

        helper(prod_dict)
        return headers

    def optimize(
        self,
        pattern: str,
        ttype: str,
        opt_limit: int,
        load: bool,
        op_type: Union[str, None],
        historical: bool = False,
        min_time: Union[datetime, None] = None,
    ):
        """
        Perform indicator parameter optimization based on the given pattern
        and trade type.

        Parameters
        ----------
        pattern : list
            List of signal patterns for optimization.
        ttype : str
            Type of the trade ('buy', 'sell').
        opt_limit: int
            Amount of the last data for which we look for the signals and
            collect signal statistics.
            This parameter is added to speed up finding the signals and s
            ignal statistic collection.
        load : bool
            Whether to load new data from exchanges.
        op_type: bool
            Flag that shows if SigBot class is used in optimization mode or not.
            If it's used in optimization mode than some class instances aren't
            need to be initialized.
        historical : bool, optional
            Whether to use historical data (default is False).
        min_time :datetime, optional
            The earliest time from which historical data are retrieved.

        Returns
        -------
        pd.DataFrame
            DataFrame containing optimization results and statistics.
        """
        main = Main()
        if self.clean:
            self.clean_prev_stat(ttype)
        # set pattern string
        pattern = "_".join(pattern)
        # get list of config dicts with all possible combinations of pattern settings
        product_dicts = self.get_product_dicts()
        print(f"Number of combinations is {len(product_dicts)}")
        # get pattern headers
        headers = self.get_headers_from_dict(product_dicts[0])
        result_statistics = None
        # flag that helps to prevent not necessary exchange data and indicator loading
        load_tickers, exchanges, database = True, None, None
        # if load flag set to True - load fresh data from exchanges,
        # else get data from dict
        for prod_dict in tqdm(product_dicts):
            # load data
            confs = self.set_configs(prod_dict, ttype)
            sb = SigBot(main, load_tickers=load_tickers, opt_type=op_type, **confs)
            # save database with indicators at the first time
            if not load_tickers:
                sb.exchanges = copy.deepcopy(exchanges)
                sb.database = copy.deepcopy(database)
            # load candle data from exchanges only at first time
            if load:
                sb.save_opt_dataframes(ttype, historical, min_time)
                load = False
            sb.save_opt_statistics(ttype, opt_limit, not load_tickers)
            # save candle data from exchanges only at second and next times
            if load_tickers:
                exchanges = copy.deepcopy(sb.exchanges)
                database = copy.deepcopy(sb.database)
                database["stat"]["buy"] = pd.DataFrame(
                    columns=["time", "ticker", "timeframe", "pattern"]
                )
                database["stat"]["sell"] = pd.DataFrame(
                    columns=["time", "ticker", "timeframe", "pattern"]
                )
                load_tickers = False
            # calculate statistic
            rs, fn = sb.stat.calculate_total_stat(sb.database, ttype, pattern)
            # create df to store statistics results
            tmp = pd.DataFrame(
                columns=["pattern"]
                + headers
                + [f"e_ratio_{lag + 1}" for lag in range(24)]
                + [f"pct_price_diff_{lag + 1}" for lag in range(24)]
                + ["forecasts_num"]
            )
            tmp["pattern"] = [pattern]
            tmp[headers] = self.get_values_from_dict(prod_dict)
            tmp[[f"e_ratio_{lag + 1}" for lag in range(24)]] = [r[0] for r in rs]
            tmp[[f"pct_price_diff_{lag + 1}" for lag in range(24)]] = [r[1] for r in rs]
            tmp["forecasts_num"] = fn
            # add temp df to the result df
            if result_statistics is None:
                result_statistics = tmp.copy()
            else:
                result_statistics = pd.concat([result_statistics, tmp])
            result_statistics = result_statistics.reset_index(drop=True)

        return result_statistics


if __name__ == "__main__":
    ttype_ = "buy"
    patterns_ = ["Pattern", "Trend"]
    indicator_list = patterns_
    indicator_list_higher = patterns_

    opt_limit_ = 1000
    load_ = False

    optim_dict_ = {
        "Pattern": {
            "use_vol": [0],
            "window_low_bound": [1],
            "window_high_bound": [6],
            "first_candle": [0.8],
            "second_candle": [0.7],
            "third_candle": [0.5],
        },
        "Trend": {"timeperiod": [6, 8, 10], "low_bound": [0]},
    }

    work_timeframe = "15m"
    higher_timeframe = "1h"

    configs_["Indicator_list"] = indicator_list
    configs_["Higher_TF_indicator_list"] = indicator_list_higher
    configs_["Timeframes"]["work_timeframe"] = work_timeframe
    configs_["Timeframes"]["higher_timeframe"] = higher_timeframe

    opt_ = Optimizer(patterns_, optim_dict_, **configs_)
    rs_ = opt_.optimize(patterns_[0], ttype_, opt_limit_, load_, "optimize")
    print("")
