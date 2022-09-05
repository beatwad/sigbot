import logging
import functools
import pandas as pd
from datetime import datetime
from os import environ
from data.get_data import GetData
from data.get_data import DataFactory
from config.config import ConfigFactory
from signals.find_signal import FindSignal
from signal_stat.signal_stat import SignalStat
from visualizer.visualizer import Visualizer
from indicators.indicators import IndicatorFactory

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
    logger = logging.getLogger("example_logger")
    logger.setLevel(logging.INFO)
    # create the logging file handler
    log_path = configs['Log']['params']['log_path']
    fh = logging.FileHandler(log_path)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    # add handler to logger object
    logger.addHandler(fh)
    return logger


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
    # Set dataframe dict
    database = {'stat': {'buy': pd.DataFrame(columns=['time', 'ticker', 'timeframe']),
                         'sell': pd.DataFrame(columns=['time', 'ticker', 'timeframe'])}}
    # List that is used to avoid processing of ticker that was already processed before
    processed_tickers = list()

    def __init__(self, **configs):
        # Get list of working and higher timeframes
        self.work_timeframe = configs['Timeframes']['work_timeframe']
        self.higher_timeframe = configs['Timeframes']['higher_timeframe']
        self.timeframes = [self.higher_timeframe, self.work_timeframe]
        # Set list of available exchanges, cryptocurrencies and tickers
        self.exchanges = {'Binance': {'API': GetData(**configs), 'tickers': []},
                          'OKEX': {'API': GetData(**configs), 'tickers': []}}
        # Get API and ticker list for every exchange in list
        for ex in list(self.exchanges.keys()):
            # get exchange API
            exchange_api = DataFactory.factory(ex, **configs)
            self.exchanges[ex]['API'] = exchange_api
            # get ticker list
            tickers = self.exchanges[ex]['API'].get_tickers()
            self.exchanges[ex]['tickers'] = tickers
            # fill ticker dict of exchange API with tickers to store current time
            # for periodic updates of ticker information
            exchange_api.fill_ticker_dict(tickers)

    def check_ticker(self, ticker: str) -> bool:
        # Check if ticker was already processesed before
        if ticker not in self.processed_tickers and ticker.replace('-', '') not in self.processed_tickers and \
                ticker.replace('/', '') not in self.processed_tickers:
            self.processed_tickers.append(ticker)
            return True
        return False

    def get_data(self, exchange_api, ticker: str, timeframe: str) -> (pd.DataFrame, int, bool):
        """ Check if new data appeared. If it is - return dataframe with the new data and amount of data """
        df = self.database.get(ticker, dict()).get(timeframe, dict()).get('data', pd.DataFrame())
        # Write data to the dataframe
        df, data_qty = exchange_api.get_data(df, ticker, timeframe)
        return df, data_qty

    def get_indicators(self, df: pd.DataFrame, ticker: str, timeframe: str, exchange_api) -> pd.DataFrame:
        """ Create indicator list from search signal patterns list, if it has the new data and
            data is not from higher timeframe, else get only levels """
        indicators = list()
        if timeframe == self.work_timeframe:
            indicator_list = configs['Indicator_list']
        else:
            indicator_list = ['SUP_RES']
        for indicator in indicator_list:
            ind_factory = IndicatorFactory.factory(indicator, configs)
            if ind_factory:
                indicators.append(ind_factory)
        # Write indicators to the dataframe, update dataframe dict
        self.database, df = exchange_api.add_indicator_data(self.database, df, indicators, ticker, timeframe,
                                                            configs)
        return df

    def get_signals(self, df: pd.DataFrame, ticker: str, timeframe: str, data_qty: int) -> (list, list):
        """ Try to find the signals and if succeed - return them and support/resistance levels """
        fs = FindSignal(configs)
        levels = self.database[ticker][timeframe]['levels']
        sig_points = fs.find_signal(df, ticker, timeframe, levels, data_qty)
        return sig_points, levels

    def get_statistics(self, sig_points: list) -> None:
        """ Calculate statistics and write it to the database """
        ss = SignalStat(**configs)
        self.database = ss.write_stat(self.database, sig_points)
        # Calculate statistics
        # print(ss.calculate_total_stat(database, 'buy'))
        # print(ss.calculate_total_stat(database, 'sell'))
        # print(ss.calculate_ticker_stat(database, 'buy', ticker, timeframe))
        # print(ss.calculate_ticker_stat(database, 'sell', ticker, timeframe))

    def add_plot(self, sig_points: list, levels: list) -> list:
        """ Generate signal plot, save it to file and add this filepath to the signal point data """
        v = Visualizer(**configs)
        for sig_point in sig_points:
            filename = v.create_plot(self.database, sig_point, levels)
            sig_point[-2].append(filename)
        return sig_points

    def get_exchange_list(self, ticker: str, sig_points: list) -> list:
        """ Add list of exchanges on which this ticker can be traded """
        for sig_point in sig_points:
            for exchange, exchange_data in self.exchanges.items():
                if ticker in exchange_data['tickers']:
                    sig_point[-1].append(exchange)
        return sig_points

    @staticmethod
    def save_dataframe(df: pd.DataFrame, ticker: str, timeframe: str) -> None:
        """ Save dataframe to the disk """
        try:
            open(f'{ticker}_{timeframe}.pkl', 'w').close()
        except FileNotFoundError:
            pass
        df.to_pickle(f'{ticker}_{timeframe}.pkl')

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
                5 - list of signal patterns, by which signal was searched for
                6 - path to file with candle/indicator plots of the signal
                7 - list of exchanges where ticker with this signal can be found """
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
                    # if debug:
                    #     df = pd.read_pickle('tests/test_ETHUSDT_5m.pkl')
                    #     if i > 1:
                    #         data_qty = 2
                    #     else:
                    #         data_qty = 1000
                    # else:
                    df, data_qty = self.get_data(exchange_api, ticker, timeframe)
                    # If we get new data - create indicator list from search signal patterns list, if it has
                    # the new data and data is not from higher timeframe, else get only levels
                    if data_qty > 1:
                        # Get indicators
                        df = self.get_indicators(df, ticker, timeframe, exchange_api)
                        # If current timeframe is working timeframe
                        if timeframe == self.work_timeframe:
                            # Get the signals
                            sig_points, levels = self.get_signals(df, ticker, timeframe, data_qty)
                            # Get the statistics
                            self.get_statistics(sig_points)
                            # For every signal create its plot and add path to it
                            sig_points = self.add_plot(sig_points, levels)
                            # Add list of exchanges where this ticker is available and has a good liquidity
                            sig_points = self.get_exchange_list(ticker, sig_points)
                            # Send signal to the Telegram bot
                            telegram_send_queue.append(sig_points)
                            # If signals are found - log them
                            if sig_points and data_qty < 1000:
                                sig_message = f'Find the signal points. Exchange is {exchange}, ticker is {ticker}, ' \
                                              f'timeframe is {timeframe}, time is {sig_points[0][4]}'
                                logger.info(sig_message)


if __name__ == "__main__":
    # Counter
    i = 1
    telegram_send_queue = list()
    main = MainClass(**configs)

    while True:
        dt1 = datetime.now()
        main.main_cycle()
        dt2 = datetime.now()
        dtm, dts = divmod((dt2 - dt1).total_seconds(), 60)
        print(f'Cycle is {i}, time for the cycle (min:sec) - {int(dtm)}:{round(dts, 2)}')
        i += 1
        sleep(300)
