import pandas as pd


class SignalStat:
    """ Class for acquiring signal statistics """
    type = 'SignalStat'

    def __init__(self, **params):
        self.params = params[self.type]['params']
        self.take_profit_multiplier = self.params.get('take_profit_multiplier', 2)
        self.stop_loss_multiplier = self.params.get('stop_loss_multiplier', 2)
        self.stat_range = self.params.get('stat_range', 12)
        self.test = self.params.get('test', False)
        self.stat_day_limit = self.params.get('stat_day_limit', 7)
        self.prev_sig_limit = self.params.get('prev_sig_limit', 1500)

    def write_stat(self, dfs: dict, signal_points: list) -> dict:
        """ Write signal statistics for every signal point for current ticker on current timeframe.
            Statistics for buy and sell trades is written separately """
        for point in signal_points:
            ticker, timeframe, index, ttype, time, pattern, plot_path, exchange_list, total_stat, ticker_stat = point
            df = dfs[ticker][timeframe]['data']
            # array of prices after signal
            signal_price = df['close'].iloc[index]
            # Try to get information about price movement after signal, if can't - continue
            try:
                _ = df['high'].iloc[index + int(3 * self.stat_range / 2)]
            except IndexError:
                continue
            # Get statistics, process it and write into the database
            result_prices = self.get_result_price_after_period(df, index, ttype, signal_price)
            dfs = self.process_statistics(dfs, point, signal_price, result_prices)

        # Save trade statistics on disk
        dfs = self.save_statistics(dfs)
        return dfs

    def get_result_price_after_period(self, df, index, ttype, signal_price):
        """ Get result prices after 15/30/45/60/75/90 minutes """
        if ttype == 'buy':
            result_price_15 = df['high'].iloc[index + int(self.stat_range / 4)]
            result_price_30 = df['high'].iloc[index + int(self.stat_range / 2)]
            result_price_45 = df['high'].iloc[index + int(3 * self.stat_range / 4)]
            result_price_60 = df['high'].iloc[index + self.stat_range]
            result_price_75 = df['high'].iloc[index + int(5 * self.stat_range / 4)]
            result_price_90 = df['high'].iloc[index + int(3 * self.stat_range / 2)]
        else:
            result_price_15 = df['low'].iloc[index + int(self.stat_range / 4)]
            result_price_30 = df['low'].iloc[index + int(self.stat_range / 2)]
            result_price_45 = df['low'].iloc[index + int(3 * self.stat_range / 4)]
            result_price_60 = df['low'].iloc[index + self.stat_range]
            result_price_75 = df['low'].iloc[index + int(5 * self.stat_range / 4)]
            result_price_90 = df['low'].iloc[index + int(3 * self.stat_range / 2)]
        return result_price_15, result_price_30, result_price_45, result_price_60, result_price_75, result_price_90

    def process_statistics(self, dfs: dict, point: list, signal_price: float, result_prices: tuple):
        """ Calculate statistics and write it to the stat dataframe if it's not presented in it """
        ticker, timeframe, index, ttype, time, pattern, plot_path, exchange_list, total_stat, ticker_stat = point
        df = dfs[ticker][timeframe]['data']
        tmp = pd.DataFrame()
        time = df['time'].iloc[index]
        tmp['time'] = [time]
        tmp['ticker'] = [ticker]
        tmp['timeframe'] = [timeframe]
        # Convert pattern to string
        pattern = str(pattern)
        tmp['pattern'] = [pattern]
        # If current statistics is not in stat dataframe - write it
        if ttype == 'buy':
            stat = dfs.get('stat').get('buy')
        else:
            stat = dfs.get('stat').get('sell')
        if stat[(stat['time'] == time) & (stat['ticker'] == ticker) &
                (stat['timeframe'] == timeframe) & (stat['pattern'] == pattern)].shape[0] == 0:
            tmp['signal_price'] = [signal_price]
            # write statistics after certain period of time
            for i, t in enumerate(range(15, 105, 15)):
                tmp[f'result_price_{t}'] = [result_prices[i]]
                tmp[f'price_diff_{t}'] = (tmp[f'result_price_{t}'] - tmp['signal_price'])
                tmp[f'pct_price_diff_{t}'] = tmp[f'price_diff_{t}'] / tmp['signal_price'] * 100
                stat = pd.concat([stat, tmp], ignore_index=True)
            # delete trades for the same tickers that are too close to each other
            stat = self.delete_close_trades(stat)
            # updata database with new stat data
            if ttype == 'buy':
                dfs['stat']['buy'] = stat
            else:
                dfs['stat']['sell'] = stat
        return dfs

    # def get_result_price_tp_sl(self, df, index, ttype, signal_price):
    #     """ Depending on deal type, try to find if price after signal hit stop loss and take profit levels.
    #         If price hit stop loss but hit take profit before - save take profit as the result price.
    #         If price hit stop loss and didn't hit take profit before - save stop loss as the result price.
    #         If price didn't hit stop loss but hit take profit  - depending on deal type save max or min price value.
    #         If price hit neither stop loss nor take profit  - depending on deal type save max or min price value. """
    #     # Get high and low price arrays
    #     high_price_points = np.zeros(self.stat_range)
    #     low_price_points = np.zeros(self.stat_range)
    #     for i in range(self.stat_range):
    #         high_price_points[i] = df['high'].iloc[index + i]
    #         low_price_points[i] = df['low'].iloc[index + i]
    #     # Calculate the result price
    #     take_profit = np.mean(df['high'] - df['low']) * self.take_profit_multiplier
    #     stop_loss = np.mean(df['high'] - df['low']) * self.stop_loss_multiplier
    #     if ttype == 'buy':
    #         try:
    #             low_price_index = np.where(np.abs(signal_price - low_price_points) > stop_loss)[0][0]
    #             stop_loss_price = low_price_points[low_price_index]
    #         except IndexError:
    #             low_price_index = self.stat_range
    #             stop_loss_price = None
    #         try:
    #             high_price_index = np.where(np.abs(high_price_points[:low_price_index] - signal_price) >
    #                                         take_profit)[0][0]
    #             take_profit_price = high_price_points[high_price_index]
    #         except IndexError:
    #             take_profit_price = None
    #     else:
    #         try:
    #             high_price_index = np.where(np.abs(high_price_points - signal_price) >
    #                                         take_profit)[0][0]
    #             stop_loss_price = high_price_points[high_price_index]
    #         except IndexError:
    #             high_price_index = self.stat_range
    #             stop_loss_price = None
    #         try:
    #             low_price_index = np.where(np.abs(signal_price - low_price_points[:high_price_index]) >
    #                                        stop_loss)[0][0]
    #             take_profit_price = low_price_points[low_price_index]
    #         except IndexError:
    #             take_profit_price = None
    #
    #     if stop_loss_price is None:
    #         if ttype == 'buy':
    #             result_price = np.max(high_price_points)
    #         else:
    #             result_price = np.min(low_price_points)
    #     elif stop_loss_price is not None and take_profit_price is None:
    #         result_price = stop_loss_price
    #     else:
    #         result_price = take_profit_price
    #     return result_price

    def save_statistics(self, dfs: dict) -> dict:
        """ Save statistics to the disk """
        if not self.test:
            try:
                open('signal_stat/buy_stat.pkl', 'r').close()
                open('signal_stat/sell_stat.pkl', 'r').close()
            except FileNotFoundError:
                open('signal_stat/buy_stat.pkl', 'w+').close()
                open('signal_stat/sell_stat.pkl', 'w+').close()
            # Write statistics to the dataframe dict
            dfs['stat']['buy'].to_pickle('signal_stat/buy_stat.pkl')
            dfs['stat']['sell'].to_pickle('signal_stat/sell_stat.pkl')
        return dfs

    @staticmethod
    def load_statistics() -> (pd.DataFrame, pd.DataFrame):
        """ Load statistics from the disk """
        try:
            buy_stat = pd.read_pickle('signal_stat/buy_stat.pkl')
            sell_stat = pd.read_pickle('signal_stat/sell_stat.pkl')
        except FileNotFoundError:
            buy_stat = pd.DataFrame(columns=['time', 'ticker', 'timeframe', 'pattern'])
            sell_stat = pd.DataFrame(columns=['time', 'ticker', 'timeframe', 'pattern'])
        return buy_stat, sell_stat

    def cut_stat_df(self, stat):
        """ Cut statistics and get only data earlier than 'stat_day_limit' days ago """
        latest_time = stat['time'].max()
        stat = stat[latest_time - stat['time'] < pd.Timedelta(self.stat_day_limit, "d")]
        return stat

    def calculate_total_stat(self, dfs: dict, ttype: str, pattern: str) -> list:
        """ Calculate signal statistics for all found signals and all tickers  """
        stat = dfs['stat'][ttype]
        stat = self.cut_stat_df(stat)
        stat = stat[(stat['pattern'].astype(str) == pattern)]
        if stat.shape[0] == 0:
            return [None for _ in range(15, 105, 15)]
        result_statistics = list()
        # calculate percent of right prognosis
        if ttype == 'buy':
            for t in range(15, 105, 15):
                pct_price_right_prognosis = round(stat[stat[f'price_diff_{t}'] > 0].shape[0] / stat.shape[0] * 100, 2)
                pct_price_diff_mean = round(stat[f'pct_price_diff_{t}'].mean(), 2)
                result_statistics.append((pct_price_right_prognosis, pct_price_diff_mean))
        else:
            for t in range(15, 105, 15):
                pct_price_right_prognosis = round(stat[stat[f'price_diff_{t}'] < 0].shape[0] / stat.shape[0] * 100, 2)
                pct_price_diff_mean = round(stat[f'pct_price_diff_{t}'].mean(), 2)
                result_statistics.append((pct_price_right_prognosis, pct_price_diff_mean))

        return result_statistics

    def calculate_ticker_stat(self, dfs: dict, ttype, ticker: str, timeframe: str, pattern: str) -> tuple:
        """ Calculate signal statistics for current ticker and current timeframe """
        stat = dfs['stat'][ttype]
        stat = self.cut_stat_df(stat)
        stat = stat[(stat['ticker'] == ticker) & (stat['timeframe'] == timeframe) & (stat['pattern'] == pattern)]
        if stat.shape[0] == 0:
            return None, None, None, None
        # calculate percent of right prognosis
        if ttype == 'buy':
            pct_price_right_prognosis = (stat[stat['price_diff'] > 0].shape[0] / stat.shape[0]) * 100
        else:
            pct_price_right_prognosis = (stat[stat['price_diff'] < 0].shape[0] / stat.shape[0]) * 100
        # calculate mean absolute and mean percent price difference after the signal
        price_diff_mean = stat['price_diff'].mean()
        pct_price_diff_mean = stat['pct_price_diff'].mean()
        return pct_price_right_prognosis, price_diff_mean, pct_price_diff_mean, stat.shape[0]

    def delete_close_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Find adjacent in time trades for the same tickers and delete them """
        patterns = df['pattern'].unique().tolist()
        df = df.sort_values(['ticker', 'time'])
        df['to_drop'] = False
        for pattern in patterns:
            tmp = df[df['pattern'] == pattern].copy()
            tmp['time_diff'] = tmp['time'].shift(-1) - tmp['time']
            tmp['ticker_shift'] = tmp['ticker'].shift(-1)
            drop_index = tmp[(tmp['ticker'] == tmp['ticker_shift']) & (tmp['time_diff'] >= pd.Timedelta(0, 's')) & \
                             (tmp['time_diff'] < pd.Timedelta(self.prev_sig_limit, 's'))].index
            df.loc[drop_index, 'to_drop'] = True
        df = df[df['to_drop'] == False]
        df = df.drop(['to_drop'], axis=1).reset_index(drop=True)
        return df
