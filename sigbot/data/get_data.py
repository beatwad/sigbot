from typing import Tuple

import os
from os import environ
from time import sleep
import pandas as pd
from api.binance_api import Binance
from api.binance_futures_api import BinanceFutures
from api.okex_api import OKEX
from api.okex_swap_api import OKEXSwap
from api.bybit_api import ByBit
from api.bybit_perpetual_api import ByBitPerpetual
from api.mexc_api import MEXC
from api.mexc_futures_api import MEXCFutures
from datetime import datetime
from log.log import logger
from api.tvdatafeed.main import TvDatafeed, Interval

bybit_key = os.getenv("BYBIT_KEY")
bybit_secret = os.getenv("BYBIT_SECRET")
bybit_test_key = os.getenv("BYBIT_TEST_KEY")
bybit_test_secret = os.getenv("BYBIT_TEST_SECRET")

binance_key = os.getenv("BINANCE_KEY")
binance_secret = os.getenv("BINANCE_SECRET")
binance_perp_key = os.getenv("BINANCE_PERP_KEY")
binance_perp_secret = os.getenv("BINANCE_PERP_SECRET")

tv_username = os.getenv("TV_USERNAME")
tv_password = os.getenv("TV_PASSWORD")
env = environ.get("ENV", "debug")


class DataFactory(object):
    """Factory class to generate instances of different exchange data classes."""

    @staticmethod
    def factory(exchange, **configs):
        """
        Create an instance of the data class for the given exchange.

        Parameters
        ----------
        exchange : str
            Name of the exchange (e.g., 'Binance', 'ByBit').
        **configs
            Configuration parameters for the data classes.

        Returns
        -------
        GetData
            An instance of the corresponding data class.
        """
        if exchange == 'Binance':
            return GetBinanceData(**configs)
        elif exchange == 'BinanceFutures':
            return GetBinanceFuturesData(**configs)
        elif exchange == 'OKEX':
            return GetOKEXData(**configs)
        elif exchange == 'OKEXSwap':
            return GetOKEXSwapData(**configs)
        elif exchange == 'ByBit':
            return GetByBitData(**configs)
        elif exchange == 'ByBitPerpetual':
            return GetByBitPerpetualData(**configs)
        elif exchange == 'MEXC':
            return GetMEXCData(**configs)
        elif exchange == 'MEXCFutures':
            return GetMEXCFuturesData(**configs)


class GetData:
    """Base class for fetching and processing cryptocurrency exchange data."""

    type = 'Data'
    name = 'Basic'

    def __init__(self, **configs):
        """
        Initialize with configuration parameters.

        Parameters
        ----------
        **configs : dict
            Configuration parameters such as limit, min_volume, etc.
        """
        self.configs = configs[self.type][self.name]['params']
        # basic interval (number of candles) to upload at startup
        self.limit = self.configs.get('limit', 0)
        # minimum trading volume (USD) for exchange ticker to be added to watch list
        self.min_volume = self.configs.get('min_volume', 0)
        # parameter to convert seconds to intervals
        self.timeframe_div = self.configs.get('timeframe_div', dict())
        # dict to store timestamp for every timeframe
        self.ticker_dict = dict()
        self.api = None
        # work timeframe
        self.work_timeframe = configs['Timeframes']['work_timeframe']
        # number of tries to get candles
        self.num_retries = 3
        # TradingView class for obtaining BTC dominance
        if self.name == 'Binance':
            self.tv_data = TvDatafeed(username=tv_username, password=tv_password)
        else:
            self.tv_data = None

    @staticmethod
    def load_saved_data(df: pd.DataFrame, ticker: str, timeframe: str) -> pd.DataFrame:
        """
        Load previously saved data for a specific ticker and timeframe if available.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing existing candle data.
        ticker : str
            Ticker symbol (e.g., BTCUSDT, ETHUSDT).
        timeframe : str
            Timeframe for the candles (e.g., '5m', '1h', '4h').

        Returns
        -------
        pd.DataFrame
            Dataframe with the saved candlestick data if available.
        """
        try:
            df = pd.read_pickle(f'optimizer/ticker_dataframes/{ticker}_{timeframe}.pkl')
        except FileNotFoundError:
            pass
        return df

    def get_data(self, df: pd.DataFrame, ticker: str, timeframe: str,
                 dt_now: datetime) -> Tuple[pd.DataFrame, int]:
        """
        Retrieve candlestick data from the exchange for a specific ticker and timeframe.

        Parameters
        ----------
        df : pd.DataFrame
            Existing dataframe with candle data.
        ticker : str
            Ticker symbol (e.g., BTCUSDT).
        timeframe : str
            Timeframe for the candles.
        dt_now : datetime
            Current datetime used to determine the time intervals.

        Returns
        -------
        Tuple[pd.DataFrame, int]
            Updated dataframe and the data retrieval limit.
        """
        limit = self.get_limit(df, ticker, timeframe, dt_now)
        # get data from exchange only when there is at least one interval to get
        if limit < 2:
            return pd.DataFrame(), limit
        # if there are errors in connection, try 3 times and only then log exception
        for i in range(self.num_retries):
            try:
                klines = self.api.get_klines(ticker, timeframe, min(limit + 2, self.limit))
            except:
                if i == self.num_retries - 1:
                    logger.exception(f'Catch an exception while trying to get candles. ' 
                                     f'API is {self.api}, ticker is {ticker}')
                sleep(1)
                continue
            else:
                break
        else:
            return df, 0
        df = self.process_data(klines, df)

        # add funding rate
        if timeframe == self.work_timeframe:
            min_time = klines['time'].min()
            for i in range(self.num_retries):
                try:
                    funding_rates = self.api.get_historical_funding_rate(ticker, min(limit + 2, self.limit // 2),
                                                                         min_time)
                except:
                    if i == self.num_retries - 1:
                        logger.exception(f'Catch an exception while trying to get funding rate. '
                                         f'API is {self.api}, ticker is {ticker}')
                    sleep(1)
                    continue
                else:
                    break
            else:
                return df, 0

            df = self.add_funding_rate(df, funding_rates, timeframe)

        return df, limit

    def get_btc_dom(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieve BTC dominance data from TradingView.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Two dataframes containing BTC dominance data.
        """
        btcd_cols = ['time', 'btcd_open', 'btcd_high', 'btcd_low', 'btcd_close', 'btcd_volume']
        btcdom_cols = ['time', 'btcdom_open', 'btcdom_high', 'btcdom_low', 'btcdom_close', 'btcdom_volume']
        # if there are errors in connection, try 3 times and only then log exception
        for i in range(self.num_retries):
            try:
                btcd = self.tv_data.get_hist('BTC.D', 'CRYPTOCAP', interval=Interval.in_daily, n_bars=50,
                                             extended_session=True).reset_index()
                btcdom = self.tv_data.get_hist('BTCDOMUSDT.P', 'BINANCE', interval=Interval.in_4_hour, n_bars=200,
                                               extended_session=True).reset_index()
            except:
                if i == self.num_retries - 1:
                    logger.exception(f'Catch an exception while trying to get BTC dominance.')
                sleep(1)
                continue
            else:
                break
        else:
            return pd.DataFrame(columns=btcd_cols), pd.DataFrame(columns=btcdom_cols)

        btcd = btcd.drop(columns='symbol')
        btcd.columns = btcd_cols
        btcd['time'] = btcd['time'] + pd.to_timedelta(23, unit='h')

        btcdom = btcdom.drop(columns='symbol')
        btcdom.columns = btcdom_cols
        btcdom['time'] = btcdom['time'] + pd.to_timedelta(3, unit='h')

        return btcd[:-1], btcdom[:-1]

    def get_hist_data(self, df: pd.DataFrame, ticker: str, timeframe: str, 
                      min_time: datetime) -> Tuple[pd.DataFrame, int]:
        """
        Retrieve historical data and funding rate for a given period.

        Parameters
        ----------
        df : pd.DataFrame
            Existing dataframe with candle data.
        ticker : str
            Ticker symbol (e.g., BTCUSDT).
        timeframe : str
            Timeframe for the candles.
        min_time : datetime
            Minimum time for historical data retrieval.

        Returns
        -------
        tuple[pd.DataFrame, int]
            Updated dataframe and the retrieval limit.
        """
        for i in range(self.num_retries):
            try:
                klines = self.api.get_historical_klines(ticker, timeframe, self.limit, min_time)
                funding_rates = self.api.get_historical_funding_rate(ticker, self.limit, min_time)
            except:
                if i == self.num_retries - 1:
                    logger.exception(f'Catch an exception while trying to get candles. ' 
                                     f'API is {self.api}, ticker is {ticker}')
                sleep(1)
                continue
            else:
                break
        else:
            return df, 0
        df = self.process_data(klines, df)
        df = df[df['time'] >= min_time].reset_index(drop=True)

        df = self.add_funding_rate(df, funding_rates, timeframe)

        return df, self.limit

    def add_funding_rate(self, df: pd.DataFrame, funding_rates: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Add funding rate data to the existing dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Existing dataframe with candle data.
        funding_rates : pd.DataFrame
            Dataframe with funding rate data.
        timeframe : str
            Timeframe for the candles.

        Returns
        -------
        pd.DataFrame
            Dataframe with funding rate data merged.
        """
        if timeframe == self.work_timeframe:
            if funding_rates.shape[0] > 0:
                funding_rates = self.process_funding_rate_data(funding_rates)
                if 'funding_rate' in df.columns:
                    df = df.drop(columns='funding_rate')
                df = pd.merge(df, funding_rates, how='left', on='time')
                df['funding_rate'] = df['funding_rate'].ffill()
                if self.name == 'OKEXSwap':  # OKEXSwap funding rate history is capped with 3 months
                    df['funding_rate'] = df['funding_rate'].fillna(0)
            else:
                df['funding_rate'] = 0
        return df
    
    def get_hist_funding_rate_data(self, ticker: str, min_time: datetime) -> tuple[pd.DataFrame, int]:
        """
        Get historical funding rate data from the exchange for a given period.

        Parameters
        ----------
        ticker : str
            Ticker symbol (e.g., BTCUSDT).
        min_time : datetime
            The earliest time from which to retrieve funding rate data.

        Returns
        -------
        tuple[pd.DataFrame, int]
            A tuple containing the dataframe with funding rate data and the retrieval limit.
        """
        for i in range(self.num_retries):
            try:
                funding_rates = self.api.get_historical_funding_rate(ticker, self.limit, min_time)
            except:
                if i == self.num_retries - 1:
                    logger.exception(f'Catch an exception while trying to get candles. ' 
                                     f'API is {self.api}, ticker is {ticker}')
                sleep(1)
                continue
            else:
                break
        else:
            return pd.DataFrame(columns=['time', 'funding_rate']), 0

        return funding_rates, self.limit

    def get_tickers(self) -> list:
        """
        Get a list of available ticker names from the exchange.

        Returns
        -------
        list
            A list of available ticker symbols that meet the minimum volume criteria.
        """
        tickers = self.api.get_ticker_names(self.min_volume)
        return tickers

    def fill_ticker_dict(self, tickers: str) -> None:
        """
        Initialize the ticker dictionary with the current timestamp for each ticker and timeframe.

        Parameters
        ----------
        tickers : str
            List of tickers to initialize in the dictionary.

        Returns
        -------
        None
        """
        # dt = datetime.now()
        for ticker in tickers:
            self.ticker_dict[ticker] = dict()
            for tf in self.timeframe_div.keys():
                self.ticker_dict[ticker][tf] = -1

    @staticmethod
    def add_utc_3(df):
        """
        Adjust the time in the dataframe by adding 3 hours (UTC+3).

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing time data.

        Returns
        -------
        pd.DataFrame
            Updated dataframe with the time shifted by 3 hours.
        """
        df['time'] = df['time'] + pd.to_timedelta(3, unit='h')
        return df

    def process_data(self, klines: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the retrieved candlestick data and merge with existing dataframe.

        Parameters
        ----------
        klines : pd.DataFrame
            Dataframe containing candlestick data from the exchange.
        df : pd.DataFrame
            Existing dataframe with candle data.

        Returns
        -------
        pd.DataFrame
            Updated dataframe with new kline data merged.
        """
        # convert numeric data to float type
        klines[['open', 'high', 'low', 'close', 'volume']] = klines[['open', 'high', 'low',
                                                                     'close', 'volume']].astype(float)
        # convert time to UTC+3
        if self.name == 'MEXCFutures':
            klines['time'] = pd.to_datetime(klines['time'], unit='s')
        else:
            klines['time'] = pd.to_datetime(klines['time'], unit='ms')
        klines = self.add_utc_3(klines)
        # If dataframe is empty - fill it with the new data
        if df.shape[0] == 0:
            df = klines
        else:
            # Update dataframe with new candles if it's not empty
            earliest_time = klines['time'].min()
            df = df[df['time'] < earliest_time]
            df = pd.concat([df, klines])
            # if size of dataframe more than limit - short it
            df = df.iloc[max(df.shape[0]-self.limit, 0):].reset_index(drop=True)
        # remove the last candle because it's not closed yet
        df = df[:-1]
        return df

    def process_funding_rate_data(self, funding_rates) -> pd.DataFrame:
        """
        Process the funding rate data by updating or creating a dataframe.

        This function converts numeric data to float, processes the timestamp,
        and adjusts it to UTC+3 for consistency.

        Parameters
        ----------
        funding_rates : pd.DataFrame
            Dataframe containing funding rate data with columns such as 'funding_rate' and 'time'.

        Returns
        -------
        pd.DataFrame
            Processed dataframe with adjusted time and converted funding rates.
        """
        # convert numeric data to float type
        funding_rates[['funding_rate']] = funding_rates[['funding_rate']].astype(float)
        # convert time to UTC+3
        funding_rates['time'] = pd.to_datetime(funding_rates['time'], unit='ms')
        funding_rates = self.add_utc_3(funding_rates)
        return funding_rates

    @staticmethod
    def add_indicator_data(dfs: dict, df: pd.DataFrame, ttype: str, indicators: list, ticker: str, timeframe: str,
                           data_qty: int, opt_flag: bool = False) -> dict:
        """
        Add indicator data to the cryptocurrency dataframe and update the dataframe dictionary.

        This method processes a list of indicators, applying them to the dataframe, and updates the dictionary
        of dataframes with the new indicator data.

        Parameters
        ----------
        dfs : dict
            Dictionary to store dataframes for different tickers, timeframes and trade types.
        df : pd.DataFrame
            Dataframe to which indicators will be added.
        ttype : str
            Type of trade ('buy' or 'sell').
        indicators : list
            List of indicator objects to be applied to the dataframe.
        ticker : str
            The cryptocurrency ticker symbol (e.g., BTCUSDT).
        timeframe : str
            Time interval for the data (e.g., '5m', '1h').
        data_qty : int
            Quantity of data points to be used for calculations.
        opt_flag : bool, optional
            Optimization flag, in this method it's used to skip certain indicators
            like 'Pattern' if True (default is False).

        Returns
        -------
        dict
            Updated dictionary with the processed dataframe and indicator levels.
        """
        levels = list()
        indicators = [i for i in indicators if i.ttype == ttype]
        # Add indicators
        for indicator in indicators:
            if opt_flag and indicator.name == 'Pattern':
                continue
            df = indicator.get_indicator(df, ticker, timeframe, data_qty)
        # Update dataframe dict
        if ticker not in dfs:
            dfs[ticker] = dict()
        if timeframe not in dfs[ticker]:
            dfs[ticker][timeframe] = dict()
        if 'data' not in dfs[ticker][timeframe]:
            dfs[ticker][timeframe]['data'] = dict()
        dfs[ticker][timeframe]['data'][ttype] = df.copy()
        dfs[ticker][timeframe]['levels'] = levels.copy()
        return dfs

    @staticmethod
    def get_time_label(dt_now: datetime, timeframe: str) -> int:
        """
        Define a time label based on the timeframe and current time.

        This function divides the time into intervals (e.g., 5-minute, 15-minute)
        depending on the timeframe provided.

        Parameters
        ----------
        dt_now : datetime
            Current datetime object.
        timeframe : str
            Time interval (e.g., '5m', '15m', '1h').

        Returns
        -------
        int
            The time label for the given timeframe, representing which part of the time interval we are in.
        """
        if timeframe == '5m':
            return dt_now.minute // 5
        elif timeframe == '15m':
            return dt_now.minute // 15
        elif timeframe == '30m':
            return dt_now.minute // 30
        elif timeframe == '1h':
            return dt_now.hour
        elif timeframe == '4h':
            hour = dt_now.hour - 3
            return hour // 4 if hour >= 0 else -1
        elif timeframe == '12h':
            hour = dt_now.hour - 3
            return hour // 12 if hour >= 0 else -1
        else:
            return dt_now.day

    def get_limit(self, df: pd.DataFrame, ticker: str, timeframe: str, dt_now: datetime) -> int:
        """
        Determine the limit for how much data to retrieve based on the existing data and the current time.

        Parameters
        ----------
        df : pd.DataFrame
            Existing dataframe with candle data.
        ticker : str
            Ticker symbol.
        timeframe : str
            Timeframe for the candles.
        dt_now : datetime
            Current datetime used to determine the time intervals.

        Returns
        -------
        int
            The number of candles to retrieve from the exchange.
        """
        dt_measure = self.get_time_label(dt_now, timeframe)
        if df.shape[0] == 0:
            self.ticker_dict[ticker][timeframe] = dt_measure
            return self.limit
        else:
            # if enough time has passed and time label has changed - increase the limit to update candle data
            if dt_measure != self.ticker_dict[ticker][timeframe] and dt_measure >= 0:
                limit = 2
                self.ticker_dict[ticker][timeframe] = dt_measure
            else:
                limit = 1
            return min(self.limit, limit)


class GetBinanceData(GetData):
    """
    Class to retrieve spot data from Binance exchange.

    Inherits from GetData and sets up Binance-specific API key, secret,
    and initializes the Binance API client.

    Attributes
    ----------
    name : str
        Name of the exchange (Binance).
    key : str
        API key for Binance.
    secret : str
        API secret for Binance.
    api : Binance
        Binance API client for interacting with the Binance exchange.
    """
    name = 'Binance'

    def __init__(self, **configs):
        super(GetBinanceData, self).__init__(**configs)
        self.key = binance_key
        self.secret = binance_secret
        self.api = Binance(self.key, self.secret)


class GetBinanceFuturesData(GetData):
    """
    Class to retrieve futures data from Binance Futures exchange.

    Inherits from GetData and sets up Binance Futures-specific API key, secret,
    and initializes the BinanceFutures API client.

    Attributes
    ----------
    name : str
        Name of the exchange (BinanceFutures).
    key : str
        API key for Binance Futures.
    secret : str
        API secret for Binance Futures.
    api : BinanceFutures
        Binance Futures API client for interacting with Binance Futures exchange.
    """
    name = 'BinanceFutures'

    def __init__(self, **configs):
        super(GetBinanceFuturesData, self).__init__(**configs)
        self.key = binance_perp_key
        self.secret = binance_perp_secret
        self.api = BinanceFutures(self.key, self.secret)


class GetOKEXData(GetData):
    """
    Class to retrieve spot data from OKEX exchange.

    Inherits from GetData and initializes the OKEX API client.

    Attributes
    ----------
    name : str
        Name of the exchange (OKEX).
    api : OKEX
        OKEX API client for interacting with OKEX exchange.
    """
    name = 'OKEX'

    def __init__(self, **configs):
        super(GetOKEXData, self).__init__(**configs)
        self.api = OKEX()


class GetOKEXSwapData(GetData):
    """
    Class to retrieve swap data from OKEX Swap exchange.

    Inherits from GetData and initializes the OKEXSwap API client.

    Attributes
    ----------
    name : str
        Name of the exchange (OKEXSwap).
    api : OKEXSwap
        OKEXSwap API client for interacting with OKEX Swap exchange.
    """
    name = 'OKEXSwap'

    def __init__(self, **configs):
        super(GetOKEXSwapData, self).__init__(**configs)
        self.api = OKEXSwap()


class GetByBitData(GetData):
    """
    Class to retrieve spot data from ByBit exchange.

    Inherits from GetData and initializes the ByBit API client.

    Attributes
    ----------
    name : str
        Name of the exchange (ByBit).
    api : ByBit
        ByBit API client for interacting with ByBit exchange.
    """
    name = 'ByBit'

    def __init__(self, **configs):
        super(GetByBitData, self).__init__(**configs)
        self.api = ByBit()


class GetByBitPerpetualData(GetData):
    """
    Class to retrieve perpetual futures data from ByBit Futures exchange.

    Inherits from GetData and sets up API key, secret based on environment
    and initializes the ByBitPerpetual API client.

    Attributes
    ----------
    name : str
        Name of the exchange (ByBitPerpetual).
    key : str
        API key for ByBit Perpetual, varies between test and production environments.
    secret : str
        API secret for ByBit Perpetual, varies between test and production environments.
    api : ByBitPerpetual
        ByBit Perpetual API client for interacting with ByBit Perpetual exchange.
    """
    name = 'ByBitPerpetual'

    def __init__(self, **configs):
        super(GetByBitPerpetualData, self).__init__(**configs)
        if env == 'debug':
            self.key = bybit_test_key
            self.secret = bybit_test_secret
        else:
            self.key = bybit_key
            self.secret = bybit_secret
        self.api = ByBitPerpetual(api_key=self.key, api_secret=self.secret)


class GetMEXCData(GetData):
    """
    Class to retrieve spot data from MEXC exchange.

    Inherits from GetData and initializes the MEXC API client.

    Attributes
    ----------
    name : str
        Name of the exchange (MEXC).
    api : MEXC
        MEXC API client for interacting with MEXC exchange.
    """
    name = 'MEXC'

    def __init__(self, **configs):
        super(GetMEXCData, self).__init__(**configs)
        self.api = MEXC()


class GetMEXCFuturesData(GetData):
    """
    Class to retrieve futures data from MEXC Futures exchange.

    Inherits from GetData and initializes the MEXCFutures API client.

    Attributes
    ----------
    name : str
        Name of the exchange (MEXCFutures).
    api : MEXCFutures
        MEXCFutures API client for interacting with MEXC Futures exchange.
    """
    name = 'MEXCFutures'

    def __init__(self, **configs):
        super(GetMEXCFuturesData, self).__init__(**configs)
        self.api = MEXCFutures()
