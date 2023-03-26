import functools
import threading
from threading import Thread, Event
import pandas as pd
from os import environ
from datetime import datetime
from data.get_data import GetData
from data.get_data import DataFactory
from config.config import ConfigFactory
from signals.find_signal import FindSignal
from signal_stat.signal_stat import SignalStat
from indicators.indicators import IndicatorFactory
from telegram_api.telegram_api import TelegramBot
from log.log import exception, logger
from constants.constants import telegram_token

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
        self.timeframes = [self.higher_timeframe, self.work_timeframe]
        self.higher_tf_patterns = configs['Higher_TF_indicator_list']
        # List of Futures Exchanges
        self.futures_exchanges = configs['Exchanges']['futures_exchanges']
        # Create indicators
        self.higher_tf_indicators, self.work_tf_indicators = self.create_indicators(configs)
        # Set list of available exchanges, cryptocurrencies and tickers
        self.exchanges = {
                          'Binance': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []},
                          'ByBit': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []},
                          'OKEX': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []},
                          'BinanceFutures': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []},
                          'ByBitPerpetual': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []},
                          'OKEXSwap': {'API': GetData(**configs), 'tickers': [], 'all_tickers': []}
                          }
        self.max_prev_candle_limit = configs['Signal_params']['params']['max_prev_candle_limit']
        # Get API and ticker list for every exchange in list
        if load_tickers:
            # Load statistics
            buy_stat, sell_stat = self.stat.load_statistics()
            self.database = {'stat': {'buy': buy_stat, 'sell': sell_stat}}
            # Load tickers
            self.get_api_and_tickers()
            # Start Telegram bot
            self.telegram_bot = TelegramBot(token=telegram_token,
                                            database=self.database, **configs)
            # self.telegram_bot.run()
        else:
            buy_stat = pd.DataFrame(columns=['time', 'ticker', 'timeframe', 'pattern'])
            sell_stat = pd.DataFrame(columns=['time', 'ticker', 'timeframe', 'pattern'])
            self.database = {'stat': {'buy': buy_stat, 'sell': sell_stat}}
        # Set candle range in which signal stat update can happen
        self.stat_update_range = configs['SignalStat']['params']['stat_range'] + 1
        # Lists for storing exchange monitor threads (Spot and Futures)
        self.spot_ex_monitor_list = list()
        self.fut_ex_monitor_list = list()
        # dictionary that is used to determine too late signals according to current work_timeframe
        self.timeframe_div = configs['Data']['Basic']['params']['timeframe_div']

    def get_api_and_tickers(self) -> None:
        """ Get API and ticker list for every exchange in list """
        exchange_list = list(self.exchanges.keys())
        for i, ex in enumerate(exchange_list):
            # get exchange API
            exchange_api = DataFactory.factory(ex, **configs)
            self.exchanges[ex]['API'] = exchange_api
            # get ticker list
            try:
                tickers, ticker_vols, all_tickers = self.exchanges[ex]['API'].get_tickers()
            except:
                del self.exchanges[ex]
                logger.exception(f'Catch an exception while accessing to exchange {ex}')
                continue

            tickers, ticker_vols = self.filter_used_tickers(tickers, ticker_vols)

            # create dictionary to store info for each ticker (volume, funding, etc.)
            self.exchanges[ex]['tickers'] = {tickers[i]: [float(ticker_vols[i])] for i in range(len(tickers))}
            # create list of all available tickers
            self.exchanges[ex]['all_tickers'] = all_tickers
            # fill ticker dict of exchange API with tickers to store current time
            # for periodic updates of ticker information
            self.exchanges[ex]['API'].fill_ticker_dict(tickers)

    def filter_used_tickers(self, tickers: list, ticker_vols: list) -> (list, list):
        """ Check if ticker was already used by previous exchange and balance number of tickers in current
            and previous exchanges if current exchange also has these tickers and their number is lesser
            than number of tickers in previous exchange """
        # create list of cleaned tickers from previous exchange
        not_used_tickers = list()
        not_used_ticker_vols = list()
        for ticker, ticker_vol in zip(tickers, ticker_vols):
            orig_ticker = ticker
            ticker = ticker.replace('-', '').replace('/', '').replace('SWAP', '')[:-4]
            # if tickers is not used - add it to current exchange list
            if ticker not in self.used_tickers:
                self.used_tickers.append(ticker)
                not_used_tickers.append(orig_ticker)
                not_used_ticker_vols.append(ticker_vol)
        return not_used_tickers, not_used_ticker_vols

    def get_data(self, exchange_api, ticker: str, timeframe: str, dt_now: datetime) -> (pd.DataFrame, int):
        """ Check if new data appeared. If it is - return dataframe with the new data and amount of data """
        df = self.database.get(ticker, dict()).get(timeframe, dict()).get('data', pd.DataFrame()).get('buy',
                                                                                                      pd.DataFrame())
        # Write data to the dataframe
        try:
            df, data_qty = exchange_api.get_data(df, ticker, timeframe, dt_now)
        except KeyError:
            logger.exception(f'Catch an exception while trying to get data')
            return df, 0
        return df, data_qty

    @staticmethod
    def create_indicators(configs) -> (list, list):
        """ Create indicators list for higher and working timeframes """
        higher_tf_indicators = list()
        working_tf_indicators = list()
        higher_tf_indicator_list = configs['Higher_TF_indicator_list']
        indicator_list = configs['Indicator_list']
        # get indicators for higher timeframe
        for ttype in ['buy', 'sell']:
            ind_factory = IndicatorFactory.factory('ATR', ttype, configs)
            higher_tf_indicators.append(ind_factory)
            working_tf_indicators.append(ind_factory)

            for indicator in higher_tf_indicator_list:
                ind_factory = IndicatorFactory.factory(indicator, ttype, configs)
                if ind_factory:
                    higher_tf_indicators.append(ind_factory)
            # get indicators for working timeframe
            for indicator in indicator_list:
                if ttype == 'sell' and indicator == 'HighVolume':
                    continue
                ind_factory = IndicatorFactory.factory(indicator, ttype, configs)
                if ind_factory and indicator not in higher_tf_indicator_list:
                    working_tf_indicators.append(ind_factory)
        return higher_tf_indicators, working_tf_indicators

    def get_indicators(self, df: pd.DataFrame, ttype: str, ticker: str, timeframe: str,
                       exchange_data: dict, data_qty: int, opt_flag: bool = False) -> (dict, int):
        """ Create indicator list from search signal patterns list, if it has the new data and
            data is not from higher timeframe, else get only levels """
        if timeframe == self.work_timeframe:
            indicators = self.work_tf_indicators
        else:
            indicators = self.higher_tf_indicators
        # Write indicators to the dataframe, update dataframe dict
        exchange_api = exchange_data['API']
        database = exchange_api.add_indicator_data(self.database, df, ttype, indicators, ticker, timeframe, data_qty,
                                                   opt_flag)
        # If enough time has passed - update statistics
        if data_qty > 1 and self.main.cycle_number > 1 and timeframe == self.work_timeframe:
            data_qty = self.stat_update_range * 1.5
        return database, data_qty

    def get_buy_signals(self, ticker: str, timeframe: str, data_qty: int, data_qty_higher: int) -> list:
        """ Try to find the signals and if succeed - return them and support/resistance levels """
        sig_points_buy = self.find_signal_buy.find_signal(self.database, ticker, timeframe, data_qty, data_qty_higher)
        return sig_points_buy

    def get_sell_signals(self, ticker: str, timeframe: str, data_qty: int, data_qty_higher: int) -> list:
        """ Try to find the signals and if succeed - return them and support/resistance levels """
        sig_points_sell = self.find_signal_sell.find_signal(self.database, ticker, timeframe, data_qty, data_qty_higher)
        return sig_points_sell

    def filter_sig_points(self, sig_points: list) -> list:
        """ Don't add signal if relatively fresh similar signal was already added to the statistics dataframe before """
        filtered_points = list()
        prev_point = (None, None, None)
        for point in sig_points:
            ticker, timeframe, index, ttype, timestamp, pattern = point[0], point[1], point[2], point[3], point[4], \
                point[5]
            # pattern is PumpDump - we need only its name without settings
            if str(pattern[0][0]).startswith('PumpDump'):
                pattern = str([pattern[0][0]] + pattern[1:])
            else:
                pattern = str(pattern)
            # if earlier signal is already exists in the signal list - don't add one more
            stat = self.database['stat'][ttype]
            df_len = self.database[ticker][timeframe]['data'][ttype].shape[0]
            if self.stat.check_close_trades(stat, df_len, ticker, index, timestamp, pattern, prev_point):
                filtered_points.append(point)
                prev_point = (ticker, timestamp, pattern)
        return filtered_points

    def filter_old_signals(self, sig_points: list) -> list:
        """ Don't send Telegram notification for the old signals (older than 1-2 candles ago) """
        filtered_points = list()
        dt_now = datetime.now()
        for point in sig_points:
            point_time = point[4]
            if (dt_now - point_time).total_seconds() <= self.timeframe_div[self.work_timeframe] * \
                    self.max_prev_candle_limit:
                filtered_points.append(point)
        return filtered_points

    def add_statistics(self, sig_points: list) -> dict:
        """ Get statistics and write it to the database """
        database = self.stat.write_stat(self.database, sig_points)
        return database

    def save_statistics(self) -> None:
        self.stat.save_statistics(self.database)

    def calc_statistics(self, sig_points: list) -> list:
        """ Calculate statistics and write it for every signal """
        for sig_point in sig_points:
            sig_type = sig_point[3]
            pattern = sig_point[5]
            result_statistics, _ = self.stat.calculate_total_stat(self.database, sig_type, pattern)
            sig_point[8].append(result_statistics)
        return sig_points

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
            if exchange.endswith('Futures') or exchange.endswith('Swap') or exchange.endswith('Perpetual'):
                fut_ex_monitor_list.append(monitor)
            else:
                spot_ex_monitor_list.append(monitor)
        return spot_ex_monitor_list, fut_ex_monitor_list

    def save_opt_dataframes(self, load=False) -> None:
        """ Save all ticker dataframes for further indicator/signal optimization """
        self.spot_ex_monitor_list, self.fut_ex_monitor_list = self.create_exchange_monitors()
        dt_now = datetime.now()
        if load:
            print('\nLoad the datasets...')
            # start all spot exchange monitors
            for monitor in self.spot_ex_monitor_list:
                monitor.save_opt_dataframes(dt_now)
            # start all futures exchange monitors
            for monitor in self.fut_ex_monitor_list:
                monitor.save_opt_dataframes(dt_now)

    def save_opt_statistics(self, ttype: str, opt_limit: int, opt_flag: bool) -> None:
        """ Save statistics in program memory for further indicator/signal optimization """
        self.spot_ex_monitor_list, self.fut_ex_monitor_list = self.create_exchange_monitors()
        # start all spot exchange monitors
        for monitor in self.spot_ex_monitor_list:
            monitor.save_opt_statistics(ttype, opt_limit, opt_flag)
        # start all futures exchange monitors
        for monitor in self.fut_ex_monitor_list:
            monitor.save_opt_statistics(ttype, opt_limit, opt_flag)

    @exception
    def main_cycle(self):
        """ Create and run exchange monitors """
        self.spot_ex_monitor_list, self.fut_ex_monitor_list = self.create_exchange_monitors()
        # start all spot exchange monitors
        for monitor in self.spot_ex_monitor_list:
            monitor.run_cycle()
        # start all futures exchange monitors
        for monitor in self.fut_ex_monitor_list:
            monitor.run_cycle()

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

    @thread_lock
    def get_indicators(self, df: pd.DataFrame, ttype: str, ticker: str, timeframe: str, data_qty: int,
                       opt_flag: bool = False) -> int:
        """ Get indicators and quantity of data """
        self.sigbot.database, data_qty = self.sigbot.get_indicators(df, ttype, ticker, timeframe, self.exchange_data,
                                                                    data_qty, opt_flag)
        return data_qty

    @thread_lock
    def add_statistics(self, sig_points: list) -> None:
        self.sigbot.database = self.sigbot.add_statistics(sig_points)

    @thread_lock
    def save_statistics(self) -> None:
        self.sigbot.save_statistics()

    def save_opt_dataframes(self, dt_now: datetime) -> None:
        """ Save dataframe for every ticker for further indicator/signal optimization """
        exchange_api = self.exchange_data['API']
        tickers = self.exchange_data['tickers']
        print(f'{self.exchange}')
        for ticker in tickers:
            # For every timeframe get the data and find the signal
            for timeframe in self.sigbot.timeframes:
                df, data_qty = self.sigbot.get_data(exchange_api, ticker, timeframe, dt_now)
                # If we previously download this dataframe to the disk - update it with new data
                try:
                    tmp = pd.read_pickle(f'ticker_dataframes/{ticker}_{timeframe}.pkl')
                except FileNotFoundError:
                    pass
                else:
                    last_time = tmp['time'].max()
                    df = df[df['time'] > last_time]
                    df = pd.concat([tmp, df], ignore_index=True)
                df_path = f'ticker_dataframes/{ticker}_{timeframe}.pkl'
                df.to_pickle(df_path)

    def save_opt_statistics(self, ttype: str, opt_limit: int, opt_flag: bool):
        """ Save statistics data for every ticker for further indicator/signal optimization """
        tickers = self.exchange_data['tickers']
        for ticker in tickers:
            # For every timeframe get the data and find the signal
            for timeframe in self.sigbot.timeframes:
                if ticker not in self.sigbot.database or timeframe not in self.sigbot.database[ticker]:
                    try:
                        df = pd.read_pickle(f'ticker_dataframes/{ticker}_{timeframe}.pkl')
                    except FileNotFoundError:
                        continue
                else:
                    df = self.sigbot.database[ticker][timeframe]['data'][ttype].copy()
                # Add indicators
                self.get_indicators(df, ttype, ticker, timeframe, 1000, opt_flag)
                # If current timeframe is working timeframe
                if timeframe == self.sigbot.work_timeframe:
                    # Get the signals
                    if ttype == 'buy':
                        sig_points = self.sigbot.find_signal_buy.find_signal(self.sigbot.database, ticker, timeframe,
                                                                             opt_limit, data_qty_higher=2)
                    else:
                        sig_points = self.sigbot.find_signal_sell.find_signal(self.sigbot.database, ticker, timeframe,
                                                                              opt_limit, data_qty_higher=2)
                    # Filter repeating signals
                    sig_points = self.sigbot.filter_sig_points(sig_points)
                    # Add the signals to statistics
                    self.add_statistics(sig_points)

    @exception
    def run_cycle(self) -> None:
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
        tickers = self.exchange_data['tickers']
        dt_now = datetime.now()
        for ticker in tickers:
            data_qty_higher = 0
            # flag that allows to pass the ticker in case of errors
            pass_the_ticker = False
            # stop thread if stop flag is set
            if self.stopped.is_set():
                break
            # For every timeframe get the data and find the signal
            for timeframe in self.sigbot.timeframes:
                if pass_the_ticker:
                    continue
                df, data_qty = self.sigbot.get_data(self.exchange_data['API'], ticker, timeframe, dt_now)
                if timeframe == self.sigbot.work_timeframe:
                    t_print(f'Cycle number {self.sigbot.main.cycle_number}, exchange {self.exchange}, ticker {ticker}')
                else:
                    data_qty_higher = data_qty
                # If we get new data - create indicator list from search signal patterns list, if it has
                # the new data and data is not from higher timeframe, else get only levels
                if data_qty > 1:
                    # Get indicators and quantity of data, if catch any exception - pass the ticker
                    try:
                        data_qty_buy = self.get_indicators(df, 'buy', ticker, timeframe, data_qty)
                        data_qty_sell = self.get_indicators(df, 'sell', ticker, timeframe, data_qty)
                    except:
                        logger.exception(f'Something bad has happened to ticker {ticker} on timeframe {timeframe}')
                        pass_the_ticker = True
                        continue
                    # If current timeframe is working timeframe
                    if timeframe == self.sigbot.work_timeframe:
                        # Get the signals
                        sig_buy_points = self.sigbot.get_buy_signals(ticker, timeframe, data_qty_buy, data_qty_higher)
                        sig_sell_points = self.sigbot.get_sell_signals(ticker, timeframe, data_qty_sell,
                                                                       data_qty_higher)
                        # If similar signal was added to stat dataframe not too long time ago (<= 3-5 ticks before) -
                        # don't add it again
                        sig_buy_points = self.sigbot.filter_sig_points(sig_buy_points)
                        sig_sell_points = self.sigbot.filter_sig_points(sig_sell_points)
                        # Add signals to statistics
                        self.add_statistics(sig_buy_points)
                        self.add_statistics(sig_sell_points)
                        # If bot cycle isn't first - calculate statistics and send Telegram notification
                        if self.sigbot.main.cycle_number > self.sigbot.main.first_cycle_qty_miss:
                            # Send signals in Telegram notification only if they are fresh (<= 1-2 ticks ago)
                            sig_buy_points = self.sigbot.filter_old_signals(sig_buy_points)
                            sig_sell_points = self.sigbot.filter_old_signals(sig_sell_points)
                            # Join buy and sell points into the one list
                            sig_points = sig_buy_points + sig_sell_points
                            # Add list of exchanges where this ticker is available and has a good liquidity
                            sig_points = self.sigbot.get_exchange_list(ticker, sig_points)
                            # Add pattern and ticker statistics
                            sig_points = self.sigbot.calc_statistics(sig_points)
                            # Send Telegram notification
                            if sig_points:
                                t_print(self.exchange,
                                        [[sp[0], sp[1], sp[2], sp[3], sp[4], sp[5]] for sp in sig_points])
                                self.sigbot.telegram_bot.notification_list += sig_points
                                # self.sigbot.telegram_bot.update_bot.set()
                                self.sigbot.telegram_bot.check_notifications()
                            # Log the signals
                            for sig_point in sig_points:
                                sig_message = f'Find the signal point. Exchange is {self.exchange}, ticker is ' \
                                              f'{ticker}, timeframe is {timeframe}, type is {sig_point[3]}, ' \
                                              f'pattern is {sig_point[5]}, time is {sig_point[4]}'
                                logger.info(sig_message)
                # Save dataframe for further analysis
                # self.save_dataframe(df, ticker, timeframe)
        # save buy and sell statistics to files
        self.save_statistics()


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
