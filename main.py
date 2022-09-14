import logging
import functools
import sys

import pandas as pd
from datetime import datetime
import glob
from os import remove
from os import environ
from data.get_data import GetData
from data.get_data import DataFactory
from config.config import ConfigFactory
from signals.find_signal import FindSignal
from signal_stat.signal_stat import SignalStat
from indicators.indicators import IndicatorFactory
from telegram_api.telegram_api import TelegramBot

from time import sleep


debug = False
# Set environment variable
environ["ENV"] = "development"
# Get configs
configs = ConfigFactory.factory(environ).configs


def create_logger():
    """
    Creates a logging object and returns it
    """
    _logger = logging.getLogger("example_logger")
    _logger.setLevel(logging.INFO)
    # create the logging file handler
    log_path = configs['Log']['params']['log_path']
    fh = logging.FileHandler(log_path)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    # add handler to logger object
    _logger.addHandler(fh)
    return _logger


# create logger
logger = create_logger()


def exception(function):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            # log the exception
            err = "There was an exception in  "
            err += function.__name__
            logger.exception(err)
            # re-raise the exception
            raise
    return wrapper


class MainClass:
    """ Class for running main program cycle """
    # Create statistics class
    stat = SignalStat(**configs)
    # Create find signal class
    find_signal = FindSignal(configs)
    buy_stat, sell_stat = stat.load_statistics()
    database = {'stat': {'buy': buy_stat,
                         'sell': sell_stat}}
    # List that is used to avoid processing of ticker that was already processed before
    processed_tickers = list()

    def __init__(self, **configs):
        # Flag of first candles read from exchanges
        self.first = True
        # Get list of working and higher timeframes
        self.work_timeframe = configs['Timeframes']['work_timeframe']
        self.higher_timeframe = configs['Timeframes']['higher_timeframe']
        self.futures_exchanges = configs['Exchanges']['futures_exchanges']
        self.timeframes = [self.higher_timeframe, self.work_timeframe]
        # Create indicators
        self.higher_tf_indicators, self.work_tf_indicators = self.create_indicators()
        # Set list of available exchanges, cryptocurrencies and tickers
        self.exchanges = {'Binance': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []},
                          'BinanceFutures': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []},
                          'OKEX': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []},
                          'OKEXSwap': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []}}
        self.max_prev_candle_limit = configs['Signal_params']['params']['max_prev_candle_limit']
        # Get API and ticker list for every exchange in list
        for ex in list(self.exchanges.keys()):
            # get exchange API
            exchange_api = DataFactory.factory(ex, **configs)
            self.exchanges[ex]['API'] = exchange_api
            # get ticker list
            tickers, all_tickers = self.exchanges[ex]['API'].get_tickers()
            self.exchanges[ex]['tickers'] = tickers
            self.exchanges[ex]['all_tickers'] = all_tickers
            # fill ticker dict of exchange API with tickers to store current time
            # for periodic updates of ticker information
            exchange_api.fill_ticker_dict(tickers)
        # Start Telegram bot
        self.telegram_bot = TelegramBot(token='5770186369:AAFrHs_te6bfjlHeD6mZDVgwvxGQ5TatiZA', **configs)
        self.telegram_bot.start()
        # Set candle range in which signal stat update can happen
        self.stat_update_range = configs['SignalStat']['params']['stat_range'] * 2

    def check_ticker(self, ticker: str) -> bool:
        # Check if ticker was already processesed before
        ticker = ticker.replace('-', '').replace('/', '').replace('SWAP', '')
        if ticker[:-4] not in self.processed_tickers:
            self.processed_tickers.append(ticker[:-4])
            return True
        return False

    def get_data(self, exchange_api, ticker: str, timeframe: str) -> (pd.DataFrame, int, bool):
        """ Check if new data appeared. If it is - return dataframe with the new data and amount of data """
        df = self.database.get(ticker, dict()).get(timeframe, dict()).get('data', pd.DataFrame())
        # Write data to the dataframe
        df, data_qty = exchange_api.get_data(df, ticker, timeframe)
        return df, data_qty

    @staticmethod
    def create_indicators() -> (list, list):
        """ Create indicators list for higher and working timeframe """
        higher_tf_indicators = list()
        working_tf_indicators = list()
        higher_tf_indicator_list = configs['Higher_TF_indicator_list']
        indicator_list = configs['Indicator_list']
        # get indicators for higher timeframe
        for indicator in higher_tf_indicator_list:
            ind_factory = IndicatorFactory.factory(indicator, configs)
            if ind_factory:
                higher_tf_indicators.append(ind_factory)
        # get indicators for working timeframe
        for indicator in indicator_list:
            ind_factory = IndicatorFactory.factory(indicator, configs)
            if ind_factory:
                working_tf_indicators.append(ind_factory)
        return higher_tf_indicators, working_tf_indicators

    def get_indicators(self, df: pd.DataFrame, ticker: str, timeframe: str,
                       exchange_api, data_qty: int) -> (pd.DataFrame, int):
        """ Create indicator list from search signal patterns list, if it has the new data and
            data is not from higher timeframe, else get only levels """
        if timeframe == self.work_timeframe:
            indicators = self.work_tf_indicators
        else:
            indicators = self.higher_tf_indicators
        # Write indicators to the dataframe, update dataframe dict
        self.database, df = exchange_api.add_indicator_data(self.database, df, indicators, ticker, timeframe,
                                                            data_qty, configs)
        # If enough time has passed - update statistics
        if data_qty > 1 and self.first is False and timeframe == self.work_timeframe:
            data_qty = self.stat_update_range
        return df, data_qty

    def get_signals(self, ticker: str, timeframe: str, data_qty: int) -> list:
        """ Try to find the signals and if succeed - return them and support/resistance levels """
        sig_points = self.find_signal.find_signal(self.database, ticker, timeframe, data_qty)
        return sig_points

    @staticmethod
    def filter_sig_points(sig_points: list) -> list:
        """ Remove signals if earlier signal is already exists in the signal list """
        filtered_points = list()
        signal_combination = list()
        for point in sig_points:
            ticker, timeframe, point_index, pattern = point[0], point[1], point[2], point[5]
            # pattern is PriceChange - we need only its name without settings
            if str(pattern[0][0]).startswith('PriceChange'):
                pattern = str(pattern[0][0])
            else:
                pattern = str(pattern)

            # if earlier signal is already exists in the signal list - don't add one more
            if (ticker, timeframe, point_index-1, pattern) in signal_combination or \
                    (ticker, timeframe, point_index-2, pattern) in signal_combination or \
                    (ticker, timeframe, point_index-3, pattern) in signal_combination or \
                    (ticker, timeframe, point_index-4, pattern) in signal_combination or \
                    (ticker, timeframe, point_index-5, pattern) in signal_combination:
                continue
            else:
                signal_combination.append((ticker, timeframe, point_index, pattern))
                filtered_points.append(point)
        return filtered_points

    def filter_early_sig_points(self, sig_points: list, df: pd.DataFrame) -> list:
        """ Remove signals that were sent too long time ago (more than 10-15 minutes) """
        filtered_points = list()
        for point in sig_points:
            point_index = point[2]
            # if too much time has passed after signal was found - skip it
            if point_index >= df.shape[0] - self.max_prev_candle_limit:
                filtered_points.append(point)
        return filtered_points

    def add_statistics(self, sig_points: list) -> None:
        """ Calculate statistics and write it to the database """
        self.database = self.stat.write_stat(self.database, sig_points)

    def calc_statistics(self, sig_points: list) -> list:
        """ Calculate statistics and write it for every signal """
        for sig_point in sig_points:
            sig_type = sig_point[3]
            pattern = sig_point[5]
            result_statistics = self.stat.calculate_total_stat(self.database, sig_type, pattern)
            sig_point[8].append(result_statistics)
        return sig_points

    def clean_statistics(self) -> None:
        # delete trades for the same tickers that are too close to each other
        self.database['stat']['buy'] = self.stat.delete_close_trades(self.database['stat']['buy'])
        self.database['stat']['sell'] = self.stat.delete_close_trades(self.database['stat']['sell'])

    def get_exchange_list(self, ticker: str, sig_points: list) -> list:
        """ Add list of exchanges on which this ticker can be traded """
        for sig_point in sig_points:
            for exchange, exchange_data in self.exchanges.items():
                ticker = ticker.replace('SWAP', '')
                if ticker in exchange_data['all_tickers'] or ticker.replace('-', '') in exchange_data['all_tickers'] \
                        or ticker.replace('/', '') in exchange_data['all_tickers']:
                    sig_point[7].append(exchange)
        return sig_points

    def save_dataframe(self, df: pd.DataFrame, ticker: str, timeframe: str) -> None:
        if timeframe == self.work_timeframe:
            """ Save dataframe to the disk """
            try:
                open(f'data/{ticker}_{timeframe}.pkl', 'w').close()
            except FileNotFoundError:
                pass
            df.to_pickle(f'data/{ticker}_{timeframe}.pkl')

    @exception
    def main_cycle(self) -> None:
        """ For every exchange, ticker, timeframe and indicator patterns in the database find the latest signals and
            send them to the Telegram module
            Signal point's structure:
                0 - ticker symbol
                1 - timeframe value
                2 - index in dataframe
                3 - type of signal (buy/sell)
                4 - time when signal appeared
                5 - signal pattern, by which signal was searched for
                6 - path to file with candle/indicator plots of the signal
                7 - list of exchanges where ticker with this signal can be found
                8 - statistics for the current pattern """
        self.processed_tickers = list()
        for exchange, exchange_data in self.exchanges.items():
            exchange_api = exchange_data['API']
            tickers = exchange_data['tickers']
            for ticker in tickers:
                if not self.check_ticker(ticker):
                    continue
                # For every timeframe get the data and find the signal
                for timeframe in self.timeframes:
                    print(f'Cycle number {i}, exchange {exchange}, ticker {ticker}, timeframe {timeframe}')
                    df, data_qty = self.get_data(exchange_api, ticker, timeframe)
                    # If we get new data - create indicator list from search signal patterns list, if it has
                    # the new data and data is not from higher timeframe, else get only levels
                    if data_qty > 1:
                        # Get indicators and quantity of data
                        df, data_qty = self.get_indicators(df, ticker, timeframe, exchange_api, data_qty)
                        # If current timeframe is working timeframe
                        if timeframe == self.work_timeframe:
                            # Get the signals
                            sig_points = self.get_signals(ticker, timeframe, data_qty)
                            # Filter repeating signals
                            sig_points = self.filter_sig_points(sig_points)
                            # Add the signals to statistics
                            self.add_statistics(sig_points)
                            # Get signals only if they are fresh (not earlier than 10-15 min ago)
                            sig_points = self.filter_early_sig_points(sig_points, df)
                            if sig_points:
                                # Clean statistics dataframes from close signal points
                                self.clean_statistics()
                                # Add list of exchanges where this ticker is available and has a good liquidity
                                sig_points = self.get_exchange_list(ticker, sig_points)
                                # Add pattern and ticker statistics
                                sig_points = self.calc_statistics(sig_points)
                                # Send Telegram notification
                                print([[sp[0], sp[1], sp[2], sp[3], sp[4], sp[5]] for sp in sig_points])
                                if not self.first:
                                    self.telegram_bot.database = self.database
                                    self.telegram_bot.notification_list += sig_points
                                    self.telegram_bot.update_bot.set()
                                    # Log the signals
                                    sig_message = f'Find the signal points. Exchange is {exchange}, ticker is ' \
                                                  f'{ticker}, timeframe is {timeframe}, time is {sig_points[0][4]}'
                                    logger.info(sig_message)
                        # Save dataframe for further analysis
                        # self.save_dataframe(df, ticker, timeframe)

        self.first = False


if __name__ == "__main__":
    # Counter
    i = 1
    main = MainClass(**configs)

    while True:
        try:
            dt1 = datetime.now()
            main.main_cycle()
            dt2 = datetime.now()
            dtm, dts = divmod((dt2 - dt1).total_seconds(), 60)
            print(f'Cycle is {i}, time for the cycle (min:sec) - {int(dtm)}:{round(dts, 2)}')
            i += 1
            sleep(60)
        except (KeyboardInterrupt, SystemExit):
            # on interruption or exit stop Telegram module thread
            main.telegram_bot.stopped.set()
            # delete everything in image directory on exit
            files = glob.glob('visualizer/images/*')
            for f in files:
                remove(f)
            # exit program
            sys.exit()
