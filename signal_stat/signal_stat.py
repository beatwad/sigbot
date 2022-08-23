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
        df = dfs[ticker][timeframe]['data']
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
                stat = dfs.get('stat').get('buy')
            else:
                stat = dfs.get('stat').get('sell')
            if stat[(stat['time'] == time) & (stat['ticker'] == ticker) &
                    (stat['timeframe'] == timeframe)].shape[0] == 0:
                tmp['signal_price'] = [signal_price]
                tmp['price_lag_5'] = [price_lag_5]
                tmp['price_lag_10'] = [price_lag_10]
                tmp['pct_diff_5'] = ((tmp['price_lag_5'] - tmp['signal_price']) / tmp['signal_price']) * 100
                tmp['pct_diff_10'] = ((tmp['price_lag_10'] - tmp['signal_price']) / tmp['signal_price']) * 100
                stat = pd.concat([stat, tmp], ignore_index=True)
                if ttype == 'buy':
                    dfs['stat']['buy'] = stat
                else:
                    dfs['stat']['sell'] = stat
        # Save statistics to the disk
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
    def calculate_total_stat(dfs: dict, type) -> tuple:
        """ Calculate statistics for all found signals for all tickers on all timeframes """
        stat = dfs['stat'][type]
        if stat.shape[0] == 0:
            return ()
        pct_avg_5_mean = stat['pct_diff_5'].mean()
        pct_avg_5_median = stat['pct_diff_5'].median()
        pct_avg_10_mean = stat['pct_diff_10'].mean()
        pct_avg_10_median = stat['pct_diff_10'].median()
        return pct_avg_5_mean, pct_avg_5_median, pct_avg_10_mean, pct_avg_10_median

    @staticmethod
    def calculate_ticker_stat(dfs: dict, type, ticker: str, timeframe: str) -> tuple:
        """ Calculate statistics for signals for current ticker on current timeframe """
        stat = dfs['stat'][type]
        stat = stat[(stat['ticker'] == ticker) & (stat['timeframe'] == timeframe)]
        if stat.shape[0] == 0:
            return ()
        pct_avg_5_mean = stat['pct_diff_5'].mean()
        pct_avg_5_median = stat['pct_diff_5'].median()
        pct_avg_10_mean = stat['pct_diff_10'].mean()
        pct_avg_10_median = stat['pct_diff_10'].median()
        return pct_avg_5_mean, pct_avg_5_median, pct_avg_10_mean, pct_avg_10_median
