import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.style as style

matplotlib.use('Agg')
style.use('fivethirtyeight')


class Visualizer:
    type = 'Visualizer'

    def __init__(self, **params):
        self.params = params[self.type]['params']
        # Path to save plot files
        self.save_path = self.params['save_path']
        self.indicator_params = params['Indicator_signal']
        self.plot_width = self.params.get('plot_width', 10)
        self.indicator_dict = self.params.get('indicator_dict', dict())
        self.level_indicators = self.params.get('level_indicators', list())
        self.boundary_indicators = self.params.get('boundary_indicators', list())
        # Max number of previous candles for which signal can be searched for
        self.max_prev_candle_limit = self.params.get('max_prev_candle_limit', 0)

    def plot_indicator_parameters(self, point, index, indicator, axs, indicator_params):
        """ Plot parameters of indicator (like low or high boundary, etc.)"""
        point_type = point[1]
        indicator_param = indicator_params[index]
        if indicator_param:
            if indicator in self.boundary_indicators:
                if point_type == 'buy':
                    axs[index + 1].axhline(y=indicator_param[0], color='g', linestyle='--', linewidth=1.5)
                else:
                    axs[index + 1].axhline(y=indicator_param[1], color='r', linestyle='--', linewidth=1.5)

    def plot_point(self, point, data, ax):
        """ Plot trade point """
        point_type = point[1]
        if point_type == 'buy':
            ax.scatter(self.plot_width, data['close'].iloc[-1], s=50, color='blue')
        else:
            ax.scatter(self.plot_width, data['close'].iloc[-1], s=50, color='blue')

    @staticmethod
    def plot_levels(data, levels, axs):
        """ Plot support and resistance levels"""
        for level in levels:
            if data['low'].min() <= level[0] <= data['high'].max():  # and level[1] == 3:
                axs[0].axhline(y=level[0], color='b', linestyle='dotted', linewidth=1.5)

    def save_plot(self, ticker, timeframe, data):
        filename = f"{self.save_path}/{ticker}_{timeframe}_{data['time'].iloc[-1]}.png"
        plt.savefig(filename, bbox_inches='tight')
        return filename

    @staticmethod
    def process_ticker(ticker):
        """ Bring ticker to more convenient view """
        if '-' in ticker:
            return ticker
        if '/' in ticker:
            ticker = ticker.replace('/', '-')
            return ticker
        ticker = ticker[:-4] + '-' + ticker[-4:]
        return ticker

    def create_plot(self, dfs, ticker, timeframe, point, levels):
        # get necessary info
        point_index = point[0]
        df = dfs[ticker][timeframe]['data']
        data = df.loc[point_index - self.plot_width:point_index]
        ohlc = data[['time', 'open', 'high', 'low', 'close', 'volume']]
        # if too much time has passed after signal was found - skip it
        if point_index < df.shape[0] - self.max_prev_candle_limit:
            return ''
        # get indicator list
        indicator_list = [p[0] for p in point[3] if p[0] not in self.level_indicators]
        indicator_params = [p[1] for p in point[3] if p not in self.level_indicators]
        plot_num = len(indicator_list) + 1

        # make subplots
        fig, axs = plt.subplots(plot_num, 1, figsize=(6, 3 * plot_num), sharex=True)
        ap = list()

        # plot candles
        ohlc = ohlc.set_index('time')

        for index, indicator in enumerate(indicator_list):
            # plot indicator
            indicator_columns = self.indicator_dict[indicator]
            for i_c in indicator_columns:
                m = mpf.make_addplot(data[i_c], panel=index + 1, title=indicator, ax=axs[index + 1], width=2)
                ap.append(m)
            # plot indicator parameters
            self.plot_indicator_parameters(point, index, indicator, axs, indicator_params)
            # plot y-labels from right side
            axs[index + 1].yaxis.set_label_position("right")
            axs[index + 1].yaxis.tick_right()

        # set xlabel
        axs[-1].set_xlabel(f"\n{data['time'].iloc[-1].date()}")

        # plot all subplots
        title = self.process_ticker(ticker)
        mpf.plot(ohlc, type='candle', ax=axs[0], addplot=ap, warn_too_much_data=1001, style='yahoo',
                 axtitle=f'{title}-{timeframe}', ylabel='')

        # plot point of trade
        self.plot_point(point, data, axs[0])

        # plot levels
        self.plot_levels(data, levels, axs)

        # save plot to file
        filename = self.save_plot(ticker, timeframe, data)

        return filename
