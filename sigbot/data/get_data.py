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
from constants.constants import bybit_key, bybit_secret, bybit_test_key, bybit_test_secret
from constants.constants import binance_key, binance_secret, binance_perp_key, binance_perp_secret
from api.tvdatafeed.main import TvDatafeed, Interval
from constants.constants import tv_username, tv_password

env = environ.get("ENV", "debug")


class DataFactory(object):
    @staticmethod
    def factory(exchange, **configs):
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
    type = 'Data'
    name = 'Basic'

    def __init__(self, **configs):
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
        """ If there is a previously saved dataframe in our base - load it
            Parameters
                ----------
                df
                    Dataframe with candles.
                ticker
                    Name of ticker (e.g. BTCUSDT, ETHUSDT).
                timeframe
                    Timeframe value (e.g. 5m, 1h, 4h, 1d).
                Returns
                -------
                df
                    List of higher timeframe indicators, list of working timeframe indicators
        """
        try:
            df = pd.read_pickle(f'optimizer/ticker_dataframes/{ticker}_{timeframe}.pkl')
        except FileNotFoundError:
            pass
        return df

    def get_data(self, df: pd.DataFrame, ticker: str, timeframe: str,
                 dt_now: datetime) -> tuple[pd.DataFrame, int]:
        """ Get data from exchange """
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

    def get_btc_dom(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ Get two types of BTC dominance indicators """
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

    def get_historical_data(self, df: pd.DataFrame, ticker: str, timeframe: str,
                            min_time: datetime) -> tuple[pd.DataFrame, int]:
        """ Get historical data from exchange for some period and also add funding rate data """
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
        """ Add funding rate data to the kandle dataframe """
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
    
    def get_historical_funding_rate_data(self, ticker: str, min_time: datetime) -> tuple[pd.DataFrame, int]:
        """ Get historical funding rate data from exchange for some period """
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
        """ Get list of available ticker names """
        tickers = self.api.get_ticker_names(self.min_volume)
        return tickers

    def fill_ticker_dict(self, tickers: str) -> None:
        """ For every ticker set timestamp of the current time """
        # dt = datetime.now()
        for ticker in tickers:
            self.ticker_dict[ticker] = dict()
            for tf in self.timeframe_div.keys():
                self.ticker_dict[ticker][tf] = -1

    @staticmethod
    def add_utc_3(df):
        df['time'] = df['time'] + pd.to_timedelta(3, unit='h')
        return df

    def process_data(self, klines: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """ Update dataframe for current ticker or create new dataframe if it's first run """
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
        """ Update dataframe for current ticker or create new dataframe if it's first run """
        # convert numeric data to float type
        funding_rates[['funding_rate']] = funding_rates[['funding_rate']].astype(float)
        # convert time to UTC+3
        funding_rates['time'] = pd.to_datetime(funding_rates['time'], unit='ms')
        funding_rates = self.add_utc_3(funding_rates)
        return funding_rates

    @staticmethod
    def add_indicator_data(dfs: dict, df: pd.DataFrame, ttype: str, indicators: list, ticker: str, timeframe: str,
                           data_qty: int, opt_flag: bool = False) -> dict:
        """ Add indicator data to cryptocurrency dataframe """
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
        """ Define time label according to the timeframe """
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
        """ Get interval needed to download from exchange according to time label """
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
    name = 'Binance'

    def __init__(self, **configs):
        super(GetBinanceData, self).__init__(**configs)
        self.key = binance_key
        self.secret = binance_secret
        self.api = Binance(self.key, self.secret)


class GetBinanceFuturesData(GetData):
    name = 'BinanceFutures'

    def __init__(self, **configs):
        super(GetBinanceFuturesData, self).__init__(**configs)
        self.key = binance_perp_key
        self.secret = binance_perp_secret
        self.api = BinanceFutures(self.key, self.secret)


class GetOKEXData(GetData):
    name = 'OKEX'

    def __init__(self, **configs):
        super(GetOKEXData, self).__init__(**configs)
        self.api = OKEX()


class GetOKEXSwapData(GetData):
    name = 'OKEXSwap'

    def __init__(self, **configs):
        super(GetOKEXSwapData, self).__init__(**configs)
        self.api = OKEXSwap()


class GetByBitData(GetData):
    name = 'ByBit'

    def __init__(self, **configs):
        super(GetByBitData, self).__init__(**configs)
        self.api = ByBit()


class GetByBitPerpetualData(GetData):
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
    name = 'MEXC'

    def __init__(self, **configs):
        super(GetMEXCData, self).__init__(**configs)
        self.api = MEXC()


class GetMEXCFuturesData(GetData):
    name = 'MEXCFutures'

    def __init__(self, **configs):
        super(GetMEXCFuturesData, self).__init__(**configs)
        self.api = MEXCFutures()
