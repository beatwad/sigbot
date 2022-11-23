import pandas as pd


class SignalStat:
    """ Class for acquiring signal statistics """
    type = 'SignalStat'

    def __init__(self, **configs):
        self.configs = configs[self.type]['params']
        self.take_profit_multiplier = self.configs.get('take_profit_multiplier', 2)
        self.stop_loss_multiplier = self.configs.get('stop_loss_multiplier', 2)
        # the number of last candles for which statistics are calculated
        self.stat_range = self.configs.get('stat_range', 24)
        self.test = self.configs.get('test', False)
        self.stat_limit_hours = self.configs.get('stat_limit_hours', 72)
        # Minimal number of candles that should pass from previous signal to add the new signal to statistics
        self.min_prev_candle_limit = self.configs.get('min_prev_candle_limit', 3)
        # dictionary that is used to determine too late signals according to current work_timeframe
        self.timeframe_div = configs['Data']['Basic']['params']['timeframe_div']
        # Get working and higher timeframes
        self.work_timeframe = configs['Timeframes']['work_timeframe']
        self.buy_stat_path = f'signal_stat/buy_stat_{self.work_timeframe}.pkl'
        self.sell_stat_path = f'signal_stat/sell_stat_{self.work_timeframe}.pkl'

    def write_stat(self, dfs: dict, signal_points: list) -> dict:
        """ Write signal statistics for every signal point for current ticker on current timeframe.
            Statistics for buy and sell trades is written separately """
        for point in signal_points:
            ticker, timeframe, index, ttype, time, pattern, plot_path, exchange_list, total_stat, ticker_stat = point
            df = dfs[ticker][timeframe]['data'][ttype]
            # array of prices after signal
            signal_price = df['close'].iloc[index]
            # Try to get information about price movement after signal, if can't - continue
            try:
                _ = df['high'].iloc[index + int(self.stat_range)]
            except IndexError:
                continue
            # If index of point was found too early - we shouldn't use it
            if index < 50:
                continue
            # Get statistics, process it and write into the database
            result_prices = self.get_result_price_after_period(df, index, ttype)
            dfs = self.process_statistics(dfs, point, signal_price, result_prices)

        # Save trade statistics on disk
        if signal_points:
            self.save_statistics(dfs)
        return dfs

    def get_result_price_after_period(self, df: pd.DataFrame, index: int, ttype: str) -> list:
        """ Get result prices after every 5 minutes """
        result_prices = list()
        for t in range(1, self.stat_range + 1):
            if ttype == 'buy':
                result_prices.append(df['high'].iloc[index + t])
            else:
                result_prices.append(df['low'].iloc[index + t])
        return result_prices

    @staticmethod
    def process_statistics(dfs: dict, point: list, signal_price: float, result_prices: list) -> dict:
        """ Calculate statistics and write it to the stat dataframe if it's not presented in it """
        # get data
        ticker, timeframe, index, ttype, time, pattern, plot_path, exchange_list, total_stat, ticker_stat = point
        ticker = ticker.replace('-', '').replace('/', '').replace('SWAP', '')
        tmp = pd.DataFrame()
        tmp['time'] = [time]
        tmp['ticker'] = [ticker]
        tmp['timeframe'] = [timeframe]

        tmp['pattern'] = [pattern]
        # If current statistics is not in stat dataframe - write it
        if ttype == 'buy':
            stat = dfs['stat']['buy']
        else:
            stat = dfs['stat']['sell']
        if stat[(stat['time'] == time) & (stat['ticker'] == ticker) &
                (stat['timeframe'] == timeframe) & (stat['pattern'] == pattern)].shape[0] == 0:
            tmp['signal_price'] = [signal_price]
            # write statistics after certain period of time
            for t, v in enumerate(result_prices):
                tmp[f'result_price_{t+1}'] = [v]
                tmp[f'price_diff_{t+1}'] = tmp[f'result_price_{t+1}'] - tmp['signal_price']
                tmp[f'pct_price_diff_{t+1}'] = tmp[f'price_diff_{t+1}'] / tmp['signal_price'] * 100
                tmp = tmp.drop(f'result_price_{t+1}', axis=1)
            stat = pd.concat([stat, tmp], ignore_index=True)
            # updata database with new stat data
            if ttype == 'buy':
                dfs['stat']['buy'] = stat
            else:
                dfs['stat']['sell'] = stat
        return dfs

    def save_statistics(self, dfs: dict) -> None:
        """ Save statistics to the disk """
        if not self.test:
            # Write statistics to the dataframe dict
            dfs['stat']['buy'].to_pickle(self.buy_stat_path)
            dfs['stat']['sell'].to_pickle(self.sell_stat_path)

    def load_statistics(self) -> (pd.DataFrame, pd.DataFrame):
        """ Load statistics from the disk """
        try:
            buy_stat = pd.read_pickle(self.buy_stat_path)
            sell_stat = pd.read_pickle(self.sell_stat_path)
        except (FileNotFoundError, EOFError):
            buy_stat = pd.DataFrame(columns=['time', 'ticker', 'timeframe', 'pattern'])
            sell_stat = pd.DataFrame(columns=['time', 'ticker', 'timeframe', 'pattern'])
        return buy_stat, sell_stat

    def cut_stat_df(self, stat, pattern):
        """ Get only last signals that have been created not earlier than N hours ago (depends on pattern) """
        latest_time = stat['time'].max()
        stat = stat[latest_time - stat['time'] < pd.Timedelta(self.stat_limit_hours[pattern], "h")]
        return stat

    def calculate_total_stat(self, dfs: dict, ttype: str, pattern: str) -> (list, int):
        """ Calculate signal statistics for all found signals and all tickers  """
        stat = dfs['stat'][ttype]

        # get statistics by pattern
        stat = stat[stat['pattern'] == pattern]
        # get only last signals that has been created not earlier than N hours ago (depends on pattern)
        stat = self.cut_stat_df(stat, pattern)
        if stat.shape[0] == 0:
            return [(0, 0, 0) for _ in range(1, self.stat_range + 1)], stat.shape[0]
        result_statistics = list()
        # calculate percent of right forecast
        for t in range(1, self.stat_range + 1):
            if ttype == 'buy':
                pct_price_right_forecast = round(stat[stat[f'price_diff_{t}'] > 0].shape[0] / stat.shape[0] * 100, 2)
            else:
                pct_price_right_forecast = round(stat[stat[f'price_diff_{t}'] < 0].shape[0] / stat.shape[0] * 100, 2)
            pct_price_diff_mean = round(stat[f'pct_price_diff_{t}'].median(), 2)
            pct_price_diff_std = round(stat[f'pct_price_diff_{t}'].std(), 2)
            result_statistics.append((pct_price_right_forecast, pct_price_diff_mean, pct_price_diff_std))
        return result_statistics, stat.shape[0]

    def check_close_trades(self, df: pd.DataFrame, ticker: str, timeframe: str,
                           point_time: pd.Timestamp, pattern: str) -> bool:
        """ Check if signal point wasn't appeared not long time ago """
        same_signal = df[(df['ticker'] == ticker) & (df['timeframe'] == timeframe) &
                         (df['pattern'] == pattern) & (df['time'] == point_time)]
        if same_signal.shape[0] > 0:
            return False
        same_signal_timestamps = df.loc[(df['ticker'] == ticker) & (df['timeframe'] == timeframe) &
                                        (df['pattern'] == pattern), 'time']
        # if can't find similar signals at all - add this signal to statistics
        if same_signal_timestamps.shape[0] == 0:
            return True
        last_signal_timestamp = same_signal_timestamps.max()
        if (point_time - last_signal_timestamp).total_seconds() > self.timeframe_div[self.work_timeframe] * \
                self.min_prev_candle_limit:
            return True
        return False
