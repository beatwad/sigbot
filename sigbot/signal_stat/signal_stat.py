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
        self.higher_timeframe = configs['Timeframes']['higher_timeframe']
        self.higher_tf_patterns = configs['Higher_TF_indicator_list']
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
            high_result_prices, low_result_prices, atr = self.get_result_price_after_period(df, index)
            dfs = self.process_statistics(dfs, point, signal_price, high_result_prices, low_result_prices, atr)
        # Save trade statistics on disk
        if signal_points:
            self.save_statistics(dfs)
        return dfs

    def get_result_price_after_period(self, df: pd.DataFrame, index: int) -> (list, list, float):
        """ Get result prices after every T minutes """
        high_result_prices = list()
        low_result_prices = list()
        atr = df['atr'].iloc[index]
        for t in range(1, self.stat_range + 1):
            high_result_prices.append(df['high'].iloc[index + t])
            low_result_prices.append(df['low'].iloc[index + t])
        return high_result_prices, low_result_prices, atr

    @staticmethod
    def process_statistics(dfs: dict, point: list, signal_price: float, high_result_prices: list,
                           low_result_prices: list, atr: float) -> dict:
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
            # write result price, MFE and MAE, if MAE is too low - replace it with MFE/1000 to prevent zero division
            for i in range(len(high_result_prices)):
                if ttype == 'buy':
                    tmp['result_price'] = [high_result_prices[i]]
                    mfe = max(max(high_result_prices[:i+1]) - signal_price, 0) / atr
                    tmp[f'mfe_{i+1}'] = [mfe]
                    tmp[f'mae_{i+1}'] = max(signal_price - min(low_result_prices[:i+1]), 0) / atr
                    if max(signal_price - min(low_result_prices[:i+1]), 0) / atr > 1000:
                        pass
                else:
                    tmp['result_price'] = [low_result_prices[i]]
                    mfe = max(signal_price - min(low_result_prices[:i+1]), 0) / atr
                    tmp[f'mfe_{i+1}'] = [mfe]
                    tmp[f'mae_{i+1}'] = max(max(high_result_prices[:i+1]) - signal_price, 0) / atr
                    if max(max(high_result_prices[:i+1]) - signal_price, 0) / atr > 1000:
                        pass
                tmp[f'price_diff_{i+1}'] = tmp[f'result_price'] - tmp['signal_price']
                tmp[f'pct_price_diff_{i+1}'] = tmp[f'price_diff_{i+1}'] / tmp['signal_price'] * 100
                tmp = tmp.drop(['result_price', f'price_diff_{i+1}'], axis=1)

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
        # calculate E-ratio (MFE/MAE), median and standard deviation of price movement
        # for each time interval after signal
        for t in range(1, self.stat_range + 1):
            try:
                e_ratio = round(sum(stat[f'mfe_{t}']) / sum(stat[f'mae_{t}']), 4)
                e_ratio = min(e_ratio, 10)
            except ZeroDivisionError:
                e_ratio = 10
            pct_price_diff_mean = round(stat[f'pct_price_diff_{t}'].median(), 2)
            pct_price_diff_std = round(stat[f'pct_price_diff_{t}'].std(), 2)
            result_statistics.append((e_ratio, pct_price_diff_mean, pct_price_diff_std))
        return result_statistics, stat.shape[0]

    def check_close_trades(self, df: pd.DataFrame, ticker: str, timeframe: str,
                           point_time: pd.Timestamp, pattern: str, prev_point: tuple) -> bool:
        """ Check if signal point wasn't appeared not long time ago """
        same_signal = df[(df['ticker'] == ticker) & (df['timeframe'] == timeframe) &
                         (df['pattern'] == pattern) & (df['time'] == point_time)]
        if same_signal.shape[0] > 0:
            return False
        same_signal_timestamps = df.loc[(df['ticker'] == ticker) & (df['timeframe'] == timeframe) &
                                        (df['pattern'] == pattern), 'time']
        # if can't find similar signals at all - check previous added point
        prev_ticker, prev_time, prev_pattern = prev_point
        if same_signal_timestamps.shape[0] == 0:
            if prev_time is None or ticker != prev_ticker or pattern != prev_pattern:
                return True
            last_signal_timestamp = prev_time
        else:
            last_signal_timestamp = same_signal_timestamps.max()
        # check if point appeared too early after previous point
        if pattern in self.higher_tf_patterns:
            if (point_time - last_signal_timestamp).total_seconds() > self.timeframe_div[self.higher_timeframe] * \
                    self.min_prev_candle_limit:
                return True
        else:
            if (point_time - last_signal_timestamp).total_seconds() > self.timeframe_div[self.work_timeframe] * \
                    self.min_prev_candle_limit:
                return True
        return False
