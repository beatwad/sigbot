import numpy as np
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

    def write_stat(self, dfs: dict, ticker: str, timeframe: str, signal_points: list) -> dict:
        """ Calculate signal statistics for every signal point for current ticker on current timeframe.
            Statistics for buy and sell trades is written separately """
        df = dfs[ticker][timeframe]['data']
        for point in signal_points:
            index, ttype, time, pattern, plot_path, exchange_list = point
            # array of prices after signal
            high_price_points = np.zeros(self.stat_range)
            low_price_points = np.zeros(self.stat_range)
            signal_price = df['close'].iloc[index]
            # Try to get information about price movement after signal, if can't - continue
            index_error = False
            for i in range(self.stat_range):
                try:
                    high_price = df['high'].iloc[index + i]
                    low_price = df['low'].iloc[index + i]
                except IndexError:
                    index_error = True
                    break
                high_price_points[i] = high_price
                low_price_points[i] = low_price
            if index_error:
                continue
            # Depending on deal type, try to find if price after signal hit stop loss and take profit levels.
            # If price hit stop loss but hit take profit before - save take profit as the result price.
            # If price hit stop loss and didn't hit take profit before - save stop loss as the result price.
            # If price didn't hit stop loss but hit take profit  - depending on deal type save max or min price value.
            # If price hit neither stop loss nor take profit  - depending on deal type save max or min price value.
            take_profit = np.mean(df['high'] - df['low']) * self.take_profit_multiplier
            stop_loss = np.mean(df['high'] - df['low']) * self.stop_loss_multiplier
            if ttype == 'buy':
                try:
                    low_price_index = np.where(np.abs(signal_price-low_price_points) > stop_loss)[0][0]
                    stop_loss_price = low_price_points[low_price_index]
                except IndexError:
                    low_price_index = self.stat_range
                    stop_loss_price = None
                try:
                    high_price_index = np.where(np.abs(high_price_points[:low_price_index]-signal_price) >
                                                take_profit)[0][0]
                    take_profit_price = high_price_points[high_price_index]
                except IndexError:
                    take_profit_price = None
            else:
                try:
                    high_price_index = np.where(np.abs(high_price_points-signal_price) >
                                                take_profit)[0][0]
                    stop_loss_price = high_price_points[high_price_index]
                except IndexError:
                    high_price_index = self.stat_range
                    stop_loss_price = None
                try:
                    low_price_index = np.where(np.abs(signal_price-low_price_points[:high_price_index]) >
                                               stop_loss)[0][0]
                    take_profit_price = low_price_points[low_price_index]
                except IndexError:
                    take_profit_price = None

            if stop_loss_price is None:
                if ttype == 'buy':
                    result_price = np.max(high_price_points)
                else:
                    result_price = np.min(low_price_points)
            elif stop_loss_price is not None and take_profit_price is None:
                result_price = stop_loss_price
            else:
                result_price = take_profit_price
            # Calculate statistics and write it to the stat dataframe if it's not presented in it
            tmp = pd.DataFrame()
            time = df['time'].iloc[index]
            tmp['time'] = [time]
            tmp['ticker'] = [ticker]
            tmp['timeframe'] = [timeframe]
            tmp['pattern'] = [pattern]
            # If current statistics is not in stat dataframe - write it
            if ttype == 'buy':
                stat = dfs.get('stat').get('buy')
            else:
                stat = dfs.get('stat').get('sell')
            if stat[(stat['time'] == time) & (stat['ticker'] == ticker) &
                    (stat['timeframe'] == timeframe)].shape[0] == 0:
                tmp['signal_price'] = [signal_price]
                tmp['result_price'] = [result_price]
                if ttype == 'buy':
                    tmp['price_diff'] = (tmp['result_price'] - tmp['signal_price'])
                else:
                    tmp['price_diff'] = (tmp['signal_price'] - tmp['result_price'])
                tmp['pct_price_diff'] = tmp['price_diff'] / tmp['signal_price'] * 100
                stat = pd.concat([stat, tmp], ignore_index=True)
                if ttype == 'buy':
                    dfs['stat']['buy'] = stat
                else:
                    dfs['stat']['sell'] = stat
        # Save statistics to the disk
        if not self.test:
            try:
                open('signal_stat/buy_stat.pkl', 'w').close()
                open('signal_stat/sell_stat.pkl', 'w').close()
            except FileNotFoundError:
                pass
            # Write statistics to the dataframe dict
            dfs['stat']['buy'].to_pickle('signal_stat/buy_stat.pkl')
            dfs['stat']['sell'].to_pickle('signal_stat/sell_stat.pkl')
        return dfs

    @staticmethod
    def calculate_total_stat(dfs: dict, ttype) -> tuple:
        """ Calculate statistics for all found signals for all tickers on all timeframes """
        stat = dfs['stat'][ttype]
        if stat.shape[0] == 0:
            return ()
        pct_price_diff_mean = stat['pct_price_diff'].mean()
        return pct_price_diff_mean, stat.shape[0]

    @staticmethod
    def calculate_ticker_stat(dfs: dict, ttype, ticker: str, timeframe: str) -> tuple:
        """ Calculate statistics for signals for current ticker on current timeframe """
        stat = dfs['stat'][ttype]
        stat = stat[(stat['ticker'] == ticker) & (stat['timeframe'] == timeframe)]
        if stat.shape[0] == 0:
            return ()
        price_diff_mean = stat['price_diff'].mean()
        pct_price_diff_mean = stat['pct_price_diff'].mean()
        return price_diff_mean, pct_price_diff_mean, stat.shape[0]
