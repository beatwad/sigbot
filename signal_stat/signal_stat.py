import pandas as pd


class SignalStat:
    """ Class for acquiring signal statistics """
    type = 'SignalStat'

    # def __init__(self, params):
    #     self.params = params[self.type]['params']

    @staticmethod
    def write_stat(dfs: dict, ticker: str, timeframe: str, signal_points: list) -> dict:
        """ Calculate signal statistics for every signal point for current ticker on current timeframe.
            Statistics for buy and sell trades is written separately """
        stat_buy = dfs.get('stat').get('buy')
        stat_sell = dfs.get('stat').get('sell')
        df = dfs[ticker][timeframe]
        # print(dfs)
        for point in signal_points:
            index, ttype = point
            # Try to get information about price movement after signal if can't - continue
            try:
                signal_price = df['close'].iloc[index]
                price_lag_5 = df['high'].iloc[index + 1]
                price_lag_10 = df['high'].iloc[index + 2]
            except IndexError:
                continue
            # Calculate statistics and write it to the stat dataframe if it's not presented in it
            tmp = pd.DataFrame()
            time = df['time'].iloc[index]
            tmp['time'] = [time]
            tmp['ticker'] = [ticker]
            tmp['timeframe'] = [timeframe]
            # If current statistics is not in stat dataframe - write it
            if ttype == 'buy':
                stat = stat_buy
            else:
                stat = stat_sell
            if stat[(stat['time'] == time) & (stat['ticker'] == ticker) &
                    (stat['timeframe'] == timeframe)].shape[0] == 0:
                tmp['signal_price'] = [signal_price]
                tmp['price_lag_5'] = [price_lag_5]
                tmp['price_lag_10'] = [price_lag_10]
                tmp['diff_5'] = tmp['price_lag_5'] - tmp['signal_price']
                tmp['pct_diff_5'] = ((tmp['price_lag_5'] - tmp['signal_price']) / tmp['signal_price']) * 100
                tmp['diff_10'] = tmp['price_lag_10'] - tmp['signal_price']
                tmp['pct_diff_10'] = ((tmp['price_lag_10'] - tmp['signal_price']) / tmp['signal_price']) * 100
                stat = pd.concat([stat, tmp])
                if ttype == 'buy':
                    dfs['stat']['buy'] = stat
                else:
                    dfs['stat']['sell'] = stat
        return dfs

    @staticmethod
    def calculate_total_stat(dfs: dict, type) -> tuple:
        """ Calculate statistics for all found signals for all tickers on all timeframes """
        stat = dfs['stat'][type]
        if stat.shape[0] == 0:
            return ()
        avg_5_mean = stat['diff_5'].mean()
        avg_5_median = stat['diff_5'].median()
        pct_avg_5_mean = stat['pct_diff_5'].mean()
        pct_avg_5_median = stat['pct_diff_5'].median()
        avg_10_mean = stat['diff_5'].mean()
        avg_10_median = stat['diff_5'].median()
        pct_avg_10_mean = stat['pct_diff_5'].mean()
        pct_avg_10_median = stat['pct_diff_5'].median()
        return avg_5_mean, avg_5_median, pct_avg_5_mean, pct_avg_5_median, \
               avg_10_mean, avg_10_median, pct_avg_10_mean, pct_avg_10_median

    @staticmethod
    def calculate_ticker_stat(dfs: dict, ticker: str, timeframe: str) -> tuple:
        """ Calculate statistics for signals for current ticker on current timeframe """
        stat = dfs['stat']
        stat = stat[(stat['ticker'] == ticker) & (stat['timeframe'] == timeframe)]
        if stat.shape[0] == 0:
            return ()
        avg_5_mean = stat['diff_5'].mean()
        avg_5_median = stat['diff_5'].median()
        pct_avg_5_mean = stat['pct_diff_5'].mean()
        pct_avg_5_median = stat['pct_diff_5'].median()
        avg_10_mean = stat['diff_5'].mean()
        avg_10_median = stat['diff_5'].median()
        pct_avg_10_mean = stat['pct_diff_5'].mean()
        pct_avg_10_median = stat['pct_diff_5'].median()
        return avg_5_mean, avg_5_median, pct_avg_5_mean, pct_avg_5_median, \
               avg_10_mean, avg_10_median, pct_avg_10_mean, pct_avg_10_median
