"""Loads OHLCV candle data from exchanges and collects signal statistics for ML training."""

from datetime import datetime
from os import environ
from typing import List

import pandas as pd

from bot.bot import SigBot
from config.config import ConfigFactory

configs_ = ConfigFactory.factory(environ).configs


class _Main:
    def __init__(self):
        self.cycle_number = 1


class TickerLoader:
    """Fetches candle data from exchanges, saves per-ticker .pkl files to data/tickers/,
    and collects signal statistics to data/signal_stat/.

    Parameters
    ----------
    pattern : list
        Signal pattern indicators, e.g. ["STOCH", "RSI", "Volume24"].
    configs : dict
        Config dict from ConfigFactory, with Timeframes/Higher_TF_indicator_list already set.
    """

    def __init__(self, pattern: List[str], test_mode: bool = False, **configs):
        configs["Patterns"] = [pattern]
        configs["Indicator_list"] = pattern + ["ATR"]
        self.configs = configs
        self.test_mode = test_mode
        self._sb = None

    def _get_sigbot(self) -> SigBot:
        if self._sb is None:
            self._sb = SigBot(
                _Main(), load_tickers=True, start_telegram=False, opt_type="ml", **self.configs
            )
            if self.test_mode:
                for exchange_data in self._sb.exchanges.values():
                    exchange_data["tickers"] = dict(list(exchange_data["tickers"].items())[:10])
        return self._sb

    def load(self, min_time: datetime = None) -> None:
        """Download historical candle data for all tickers on all exchanges.

        Parameters
        ----------
        min_time : datetime, optional
            Earliest timestamp to fetch. Defaults to 10 years ago.
        """
        if min_time is None:
            min_time = datetime.now().replace(microsecond=0, second=0, minute=0) - pd.to_timedelta(
                365 * 10, unit="D"
            )
        self._get_sigbot().save_opt_dataframes(load=True, min_time=min_time)

    def collect_statistics(self, ttype: str, opt_limit: int = 100000) -> None:
        """Find signals in saved ticker data and write stats to data/signal_stat/.

        Parameters
        ----------
        ttype : str
            Trade type: "buy" or "sell".
        opt_limit : int, optional
            Max number of recent candles to scan for signals. Default is 100000.
        """
        self._get_sigbot().save_opt_statistics(ttype, opt_limit, opt_flag=False)
