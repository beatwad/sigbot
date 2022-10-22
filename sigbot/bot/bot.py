import sys
import functools
import threading
from threading import Thread
from threading import Event
import pandas as pd
from os import environ
from data.get_data import GetData
from data.get_data import DataFactory
from config.config import ConfigFactory
from signals.find_signal import FindSignal
from signal_stat.signal_stat import SignalStat
from indicators.indicators import IndicatorFactory
from telegram_api.telegram_api import TelegramBot
from log.log import exception, logger

sys.path.insert(0, '..')


# Get configs
configs = ConfigFactory.factory(environ).configs
# variable for thread locking
global_lock = threading.Lock()


def thread_lock(function):
    """ Threading lock decorator """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # wait until global lock released
        while global_lock.locked():
            continue
        # acquire lock
        global_lock.acquire()
        # execute function code
        f = function(*args, **kwargs)
        # after all operations are done - release the lock
        global_lock.release()
        return f
    return wrapper


@thread_lock
def t_print(*args):
    """ Thread safe print """
    print(*args, flush=True)


class SigBot:
    """ Class for running main the entire Signal Bot """

    @exception
    def __init__(self, main_class, load_tickers=True, **configs):
        # Get main bot class
        self.main = main_class
        # Create statistics class
        self.stat = SignalStat(**configs)
        # Create find signal class
        self.find_signal_buy = FindSignal('buy', configs)
        self.find_signal_sell = FindSignal('sell', configs)
        # List that is used to avoid processing of ticker that was already processed before
        self.used_tickers = list()
        # Get working and higher timeframes
        self.work_timeframe = configs['Timeframes']['work_timeframe']
        self.higher_timeframe = configs['Timeframes']['higher_timeframe']
        self.futures_exchanges = configs['Exchanges']['futures_exchanges']
        self.timeframes = [self.higher_timeframe, self.work_timeframe]
        # Create indicators
        self.higher_tf_indicators, self.work_tf_indicators = self.create_indicators(configs)
        # Set list of available exchanges, cryptocurrencies and tickers
        self.exchanges = {'Binance': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []},
                          # 'OKEX': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []},
                          'BinanceFutures': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []}, }
                          # 'OKEXSwap': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []}}
        self.max_prev_candle_limit = configs['Signal_params']['params']['max_prev_candle_limit']
        # Get API and ticker list for every exchange in list
        if load_tickers:
            # Load statistics
            buy_stat, sell_stat = self.stat.load_statistics()
            self.database = {'stat': {'buy': buy_stat, 'sell': sell_stat}}
            # Load tickers
            self.get_api_and_tickers()
            # Start Telegram bot
            self.telegram_bot = TelegramBot(token='5770186369:AAFrHs_te6bfjlHeD6mZDVgwvxGQ5TatiZA',
                                            database=self.database, **configs)
            self.telegram_bot.start()
        else:
            buy_stat = pd.DataFrame(columns=['time', 'ticker', 'timeframe', 'pattern'])
            sell_stat = pd.DataFrame(columns=['time', 'ticker', 'timeframe', 'pattern'])
            self.database = {'stat': {'buy': buy_stat, 'sell': sell_stat}}
        # Set candle range in which signal stat update can happen
        self.stat_update_range = configs['SignalStat']['params']['stat_range'] * 2
        # Lists for storing exchange monitor threads (Spot and Futures)
        self.spot_ex_monitor_list = list()
        self.fut_ex_monitor_list = list()
        # dataframe storage for optimization
        self.opt_dfs = dict()

    def get_api_and_tickers(self) -> None:
        """ Get API and ticker list for every exchange in list """
        exchange_list = list(self.exchanges.keys())
        for i, ex in enumerate(exchange_list):
            # get exchange API
            exchange_api = DataFactory.factory(ex, **configs)
            self.exchanges[ex]['API'] = exchange_api
            # get ticker list
            tickers, all_tickers = self.exchanges[ex]['API'].get_tickers()
            # check if ticker wasn't used by previous exchange
            # if i > 0:
            #     prev_tickers = self.exchanges[exchange_list[i - 1]]['tickers']
            #     tickers, prev_tickers = self.filter_used_tickers(tickers, prev_tickers)
            #     #self.exchanges[exchange_list[i - 1]]['tickers'] = prev_tickers
            # else:
            prev_tickers = list()
            tickers, prev_tickers = self.filter_used_tickers(tickers, prev_tickers)

            self.exchanges[ex]['tickers'] = tickers
            self.exchanges[ex]['all_tickers'] = all_tickers
            # fill ticker dict of exchange API with tickers to store current time
            # for periodic updates of ticker information
            self.exchanges[ex]['API'].fill_ticker_dict(tickers)

    def filter_used_tickers(self, tickers: list, prev_tickers: list, len_diff=50) -> (list, list):
        """ Check if ticker was already used by previous exchange and balance number of tickers in current
            and previous exchanges if current exchange also has these tickers and their number is lesser
            than number of tickers in previous exchange """
        # create list of cleaned tickers from previous exchange
        not_used_tickers = list()
        # prev_cleaned_tickers = [t.replace('-', '').replace('/', '').replace('SWAP', '')[:-4] for t in prev_tickers]
        # prev_tickers_len = len(prev_tickers)
        # prev_tickers_indexes = list()
        for ticker in tickers:
            orig_ticker = ticker
            ticker = ticker.replace('-', '').replace('/', '').replace('SWAP', '')[:-4]
            # if tickers is not used - add it to current exchange list
            if ticker not in self.used_tickers:
                self.used_tickers.append(ticker)
                not_used_tickers.append(orig_ticker)
            # if tickers is used by previous exchange, but number of tickers in previous exchange is significantly
            # bigger than number of tickers in current exchange - add it to current exchange list
            # and remove from previous exchange list
            # elif ticker in prev_cleaned_tickers and len(not_used_tickers) < prev_tickers_len - len_diff:
            #     prev_tickers_len -= 1
            #     idx = prev_cleaned_tickers.index(ticker)
            #     prev_tickers_indexes.append(idx)
            #     not_used_tickers.append(orig_ticker)
        # prev_tickers = self.clean_prev_exchange_tickers(prev_tickers, prev_tickers_indexes)
        return not_used_tickers, prev_tickers

    @staticmethod
    def clean_prev_exchange_tickers(prev_tickers: list, prev_tickers_indexes: list) -> list:
        """ Delete tickers from previous to balance load on to exchanges """
        cleaned_prev_tickers = list()
        for idx, ticker in enumerate(prev_tickers):
            if idx not in prev_tickers_indexes:
                cleaned_prev_tickers.append(ticker)
        return cleaned_prev_tickers

    def get_data(self, exchange_api, ticker: str, timeframe: str) -> (pd.DataFrame, int):
        """ Check if new data appeared. If it is - return dataframe with the new data and amount of data """
        df = self.database.get(ticker, dict()).get(timeframe,
                                                   dict()).get('data', pd.DataFrame()).get('buy', pd.DataFrame())
        # Write data to the dataframe
        df, data_qty = exchange_api.get_data(df, ticker, timeframe)
        return df, data_qty

    @staticmethod
    def create_indicators(configs) -> (list, list):
        """ Create indicators list for higher and working timeframe """
        higher_tf_indicators = list()
        working_tf_indicators = list()
        higher_tf_indicator_list = configs['Higher_TF_indicator_list']
        indicator_list = configs['Indicator_list']
        # get indicators for higher timeframe
        for ttype in ['buy', 'sell']:
            for indicator in higher_tf_indicator_list:
                ind_factory = IndicatorFactory.factory(indicator, ttype, configs)
                if ind_factory:
                    higher_tf_indicators.append(ind_factory)
            # get indicators for working timeframe
            for indicator in indicator_list:
                ind_factory = IndicatorFactory.factory(indicator, ttype, configs)
                if ind_factory and indicator not in higher_tf_indicator_list:
                    working_tf_indicators.append(ind_factory)
        return higher_tf_indicators, working_tf_indicators

    def get_indicators(self, df: pd.DataFrame, ttype: str, ticker: str, timeframe: str,
                       exchange_api, data_qty: int) -> (dict, int):
        """ Create indicator list from search signal patterns list, if it has the new data and
            data is not from higher timeframe, else get only levels """
        if timeframe == self.work_timeframe:
            indicators = self.work_tf_indicators
        else:
            indicators = self.higher_tf_indicators
        # Write indicators to the dataframe, update dataframe dict
        database = exchange_api.add_indicator_data(self.database, df, ttype, indicators, ticker, timeframe,
                                                   data_qty, configs)
        # If enough time has passed - update statistics
        if data_qty > 1 and self.main.cycle_number > 1 and timeframe == self.work_timeframe:
            data_qty = self.stat_update_range
        return database, data_qty

    def get_buy_signals(self, ticker: str, timeframe: str, data_qty: int) -> list:
        """ Try to find the signals and if succeed - return them and support/resistance levels """
        sig_points_buy = self.find_signal_buy.find_signal(self.database, ticker, timeframe, data_qty)
        return sig_points_buy

    def get_sell_signals(self, ticker: str, timeframe: str, data_qty: int) -> list:
        """ Try to find the signals and if succeed - return them and support/resistance levels """
        sig_points_sell = self.find_signal_sell.find_signal(self.database, ticker, timeframe, data_qty)
        return sig_points_sell

    @staticmethod
    def filter_sig_points(sig_points: list) -> list:
        """ Remove signals if earlier signal is already exists in the signal list """
        filtered_points = list()
        signal_combination = list()
        for point in sig_points:
            ticker, timeframe, point_index, pattern = point[0], point[1], point[2], point[5]
            # pattern is PriceChange - we need only its name without settings
            if str(pattern[0][0]).startswith('PriceChange'):
                pattern = str([pattern[0][0]] + pattern[1:])
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

    def add_statistics(self, sig_points: list, ttype: str) -> dict:
        """ Get statistics and write it to the database """
        database = self.stat.write_stat(self.database, sig_points, ttype)
        return database

    def calc_statistics(self, sig_points: list) -> list:
        """ Calculate statistics and write it for every signal """
        for sig_point in sig_points:
            sig_type = sig_point[3]
            pattern = sig_point[5]
            result_statistics, _ = self.stat.calculate_total_stat(self.database, sig_type, pattern)
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

    def create_exchange_monitors(self) -> (list, list):
        """ Create list of instances for ticker monitoring for every exchange """
        spot_ex_monitor_list = list()
        fut_ex_monitor_list = list()
        for exchange, exchange_data in self.exchanges.items():
            monitor = MonitorExchange(self, exchange, exchange_data)
            if exchange.endswith('Futures') or exchange.endswith('Swap'):
                fut_ex_monitor_list.append(monitor)
            else:
                spot_ex_monitor_list.append(monitor)
        return spot_ex_monitor_list, fut_ex_monitor_list

    def save_opt_dataframes(self, load=False) -> None:
        """ Save all ticker dataframes for further indicator/signal optimization """
        self.spot_ex_monitor_list, self.fut_ex_monitor_list = self.create_exchange_monitors()
        if load:
            print('\nLoad the datasets...')
            # start all spot exchange monitors
            for monitor in self.spot_ex_monitor_list:
                monitor.save_opt_dataframes()
            # start all futures exchange monitors
            for monitor in self.fut_ex_monitor_list:
                monitor.save_opt_dataframes()

    def save_opt_statistics(self, ttype: str, opt_limit: int) -> None:
        """ Save statistics in program memory for further indicator/signal optimization """
        self.spot_ex_monitor_list, self.fut_ex_monitor_list = self.create_exchange_monitors()
        # start all spot exchange monitors
        for monitor in self.spot_ex_monitor_list:
            monitor.save_opt_statistics(ttype, opt_limit)
        # start all futures exchange monitors
        for monitor in self.fut_ex_monitor_list:
            monitor.save_opt_statistics(ttype, opt_limit)

    @exception
    def main_cycle(self):
        """ Create and run exchange monitors """
        self.spot_ex_monitor_list, self.fut_ex_monitor_list = self.create_exchange_monitors()
        # start all spot exchange monitors
        for monitor in self.spot_ex_monitor_list:
            monitor.start()
        # wait until spot monitor finish its work
        for monitor in self.spot_ex_monitor_list:
            monitor.join()
        # start all futures exchange monitors
        for monitor in self.fut_ex_monitor_list:
            monitor.start()
        # wait until futures monitor finish its work
        for monitor in self.fut_ex_monitor_list:
            monitor.join()

    def stop_monitors(self):
        """ Stop all exchange monitors """
        for monitor in self.spot_ex_monitor_list:
            monitor.stopped.set()
        for monitor in self.fut_ex_monitor_list:
            monitor.stopped.set()


class MonitorExchange(Thread):
    # constructor
    def __init__(self, sigbot, exchange, exchange_data):
        # initialize separate thread the Telegram bot, so it can work independently
        Thread.__init__(self)
        # instance of main class
        self.sigbot = sigbot
        # event for stopping bot thread
        self.stopped = Event()
        # exchange name
        self.exchange = exchange
        # exchange data
        self.exchange_data = exchange_data
        # limit of candles for use in optimization statistics
        self.opt_limit = 1000

    @thread_lock
    def get_indicators(self, df, ttype, ticker, timeframe, exchange_api, data_qty) -> (pd.DataFrame, pd.DataFrame, int):
        # Get indicators and quantity of data
        self.sigbot.database, data_qty = self.sigbot.get_indicators(df, ttype, ticker, timeframe, exchange_api,
                                                                    data_qty)
        return data_qty

    @thread_lock
    def add_statistics(self, sig_points, ttype) -> None:
        self.sigbot.database = self.sigbot.add_statistics(sig_points, ttype)

    @thread_lock
    def clean_statistics(self) -> None:
        self.sigbot.clean_statistics()

    def save_opt_dataframes(self) -> None:
        """ Save dataframe for every ticker for further indicator/signal optimization """
        exchange_api = self.exchange_data['API']
        tickers = self.exchange_data['tickers']
        print(f'{self.exchange}')
        for ticker in tickers:
            # For every timeframe get the data and find the signal
            for timeframe in self.sigbot.timeframes:
                df, data_qty = self.sigbot.get_data(exchange_api, ticker, timeframe)
                # Save dataframe to the disk
                try:
                    open(f'ticker_dataframes/{ticker}_{timeframe}.pkl', 'w').close()
                except FileNotFoundError:
                    pass
                df_path = f'ticker_dataframes/{ticker}_{timeframe}.pkl'
                df.to_pickle(df_path)

    def save_opt_statistics(self, ttype: str, opt_limit: int):
        """ Save statistics data for every ticker for further indicator/signal optimization """
        exchange_api = self.exchange_data['API']
        tickers = self.exchange_data['tickers']
        for ticker in tickers:
            # For every timeframe get the data and find the signal
            for timeframe in self.sigbot.timeframes:
                if f'{ticker}_{timeframe}' not in self.sigbot.opt_dfs:
                    try:
                        df = pd.read_pickle(f'ticker_dataframes/{ticker}_{timeframe}.pkl')
                        self.sigbot.opt_dfs[f'{ticker}_{timeframe}'] = df.copy()
                    except FileNotFoundError:
                        continue
                else:
                    df = self.sigbot.opt_dfs[f'{ticker}_{timeframe}'].copy()
                self.get_indicators(df, ttype, ticker, timeframe, exchange_api, 1000)
                # If current timeframe is working timeframe
                if timeframe == self.sigbot.work_timeframe:
                    # Get the signals
                    if ttype == 'buy':
                        sig_points = self.sigbot.find_signal_buy.find_signal(self.sigbot.database, ticker, timeframe,
                                                                             opt_limit)
                    else:
                        sig_points = self.sigbot.find_signal_sell.find_signal(self.sigbot.database, ticker, timeframe,
                                                                              opt_limit)
                    # Filter repeating signals
                    sig_points = self.sigbot.filter_sig_points(sig_points)
                    # Add the signals to statistics
                    self.add_statistics(sig_points, ttype)

    @exception
    def run(self) -> None:
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
        exchange_api = self.exchange_data['API']
        tickers = self.exchange_data['tickers']
        for ticker in tickers:
            # stop thread if stop flag is set
            if self.stopped.is_set():
                break
            # For every timeframe get the data and find the signal
            for timeframe in self.sigbot.timeframes:
                df, data_qty = self.sigbot.get_data(exchange_api, ticker, timeframe)
                if timeframe == self.sigbot.work_timeframe:
                    t_print(f'Cycle number {self.sigbot.main.cycle_number}, exchange {self.exchange}, ticker {ticker}')
                # If we get new data - create indicator list from search signal patterns list, if it has
                # the new data and data is not from higher timeframe, else get only levels
                if data_qty > 1:
                    # Get indicators and quantity of data
                    data_qty_buy = self.get_indicators(df, 'buy', ticker, timeframe, exchange_api, data_qty)
                    data_qty_sell = self.get_indicators(df, 'sell', ticker, timeframe, exchange_api, data_qty)
                    # If current timeframe is working timeframe
                    if timeframe == self.sigbot.work_timeframe:
                        # Get the signals
                        sig_buy_points = self.sigbot.get_buy_signals(ticker, timeframe, data_qty_buy)
                        sig_sell_points = self.sigbot.get_sell_signals(ticker, timeframe, data_qty_sell)
                        # Filter repeating signals
                        sig_buy_points = self.sigbot.filter_sig_points(sig_buy_points)
                        sig_sell_points = self.sigbot.filter_sig_points(sig_sell_points)
                        # Add the signals to statistics
                        self.add_statistics(sig_buy_points, 'buy')
                        self.add_statistics(sig_sell_points, 'sell')
                        # Get signals only if they are fresh (not earlier than 10-15 min ago)
                        df_buy = self.sigbot.database[ticker][timeframe]['data']['buy']
                        sig_buy_points = self.sigbot.filter_early_sig_points(sig_buy_points, df_buy)
                        df_sell = self.sigbot.database[ticker][timeframe]['data']['sell']
                        sig_sell_points = self.sigbot.filter_early_sig_points(sig_sell_points, df_sell)
                        sig_points = sig_buy_points # + sig_sell_points
                        if sig_points:
                            # Clean statistics dataframes from close signal points
                            self.clean_statistics()
                            # Add list of exchanges where this ticker is available and has a good liquidity
                            sig_points = self.sigbot.get_exchange_list(ticker, sig_points)
                            # Add pattern and ticker statistics
                            sig_points = self.sigbot.calc_statistics(sig_points)
                            # Send Telegram notification
                            t_print(self.exchange, [[sp[0], sp[1], sp[2], sp[3], sp[4], sp[5]] for sp in sig_points])
                            if self.sigbot.main.cycle_number > 1:
                                self.sigbot.telegram_bot.notification_list += sig_points
                                self.sigbot.telegram_bot.update_bot.set()
                                # Log the signals
                                for sig_point in sig_points:
                                    sig_message = f'Find the signal point. Exchange is {self.exchange}, ticker is ' \
                                                  f'{ticker}, timeframe is {timeframe}, type is {sig_point[3]}, ' \
                                                  f'pattern is {sig_point[5]}, time is {sig_point[4]}'
                                    logger.info(sig_message)
                    # Save dataframe for further analysis
                    # self.save_dataframe(df, ticker, timeframe)


# if __name__ == "__main__":
#     # Counter
#     cycle_number = 1
#     main = SigBot(**configs)
#
#     while True:
#         try:
#             dt1 = datetime.now()
#             main.main_cycle()
#             dt2 = datetime.now()
#             dtm, dts = divmod((dt2 - dt1).total_seconds(), 60)
#             print(f'Cycle is {cycle_number}, time for the cycle (min:sec) - {int(dtm)}:{round(dts, 2)}')
#             cycle_number += 1
#             sleep(30)
#         except (KeyboardInterrupt, SystemExit):
#             # stop all exchange monitors
#             main.stop_monitors()
#             # on interruption or exit stop Telegram module thread
#             main.telegram_bot.stopped.set()
#             # delete everything in image directory on exit
#             files = glob.glob('visualizer/images/*')
#             for f in files:
#                 remove(f)
#             # exit program
#             sys.exit()
