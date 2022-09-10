import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.style as style
from matplotlib import rcParams

matplotlib.use('Agg')
style.use('dark_background')
rcParams['font.family'] = 'cursive'


class Visualizer:
    type = 'Visualizer'

    def __init__(self, **params):
        self.params = params[self.type]['params']
        # Path to save plot files
        self.image_path = self.params['image_path']
        self.indicator_params = params['Indicator_signal']
        self.plot_width = self.params.get('plot_width', 10)
        self.indicator_dict = self.params.get('indicator_dict', dict())
        self.level_indicators = self.params.get('level_indicators', list())
        self.boundary_indicators = self.params.get('boundary_indicators', list())
        # Max number of previous candles for which signal can be searched for
        self.max_prev_candle_limit = self.params.get('max_prev_candle_limit', 0)

    def plot_indicator_parameters(self, point_type: str, index: int, indicator: str,
                                  axs: plt.axis, indicator_params: list) -> None:
        """ Plot parameters of indicator (like low or high boundary, etc.)"""
        indicator_param = indicator_params[index]
        if indicator_param:
            if indicator in self.boundary_indicators:
                if point_type == 'buy':
                    axs[index + 1].axhline(y=indicator_param[0], color='g', linestyle='--', linewidth=1.5)
                else:
                    axs[index + 1].axhline(y=indicator_param[1], color='r', linestyle='--', linewidth=1.5)

    def plot_point(self, point_type: str, data: pd.DataFrame, ax: plt.axis) -> None:
        """ Plot trade point """
        if point_type == 'buy':
            ax.scatter(self.plot_width, data['close'].iloc[-1], s=50, color='blue')
        else:
            ax.scatter(self.plot_width, data['close'].iloc[-1], s=50, color='blue')

    @staticmethod
    def plot_levels(data: pd.DataFrame, levels: list, axs: plt.axis) -> None:
        """ Plot support and resistance levels"""
        for level in levels:
            if data['low'].min() <= level[0] <= data['high'].max():  # and level[1] == 3:
                axs[0].axhline(y=level[0], color='b', linestyle='dotted', linewidth=1.5)

    def save_plot(self, ticker, timeframe, data):
        filename = f"{self.image_path}/{ticker}_{timeframe}_{data['time'].iloc[-1]}.png"
        plt.savefig(filename, bbox_inches='tight')
        return filename

    def save_stat_plot(self, ticker, timeframe):
        filename = f"{self.image_path}/{ticker}_{timeframe}_statistics.png"
        plt.savefig(filename, bbox_inches='tight')
        return filename

    @staticmethod
    def process_ticker(ticker: str) -> str:
        """ Bring ticker to more convenient view """
        if '-' in ticker:
            return ticker
        if '/' in ticker:
            ticker = ticker.replace('/', '-')
            return ticker
        ticker = ticker[:-4] + '-' + ticker[-4:]
        return ticker

    def create_plot(self, dfs, point, levels):
        # get necessary info
        ticker, timeframe, point_index, point_type, time, pattern, plot_path, exchange_list, statistics, y = point
        df = dfs[ticker][timeframe]['data']
        data = df.loc[point_index - self.plot_width:point_index]
        ohlc = data[['time', 'open', 'high', 'low', 'close', 'volume']]
        # get indicator list
        indicator_list = [p[0] for p in pattern if p[0] not in self.level_indicators]
        indicator_params = [p[1] for p in pattern if p not in self.level_indicators]
        plot_num = len(indicator_list) + 1

        # Plot signal
        # make subfigs
        fig = plt.figure(constrained_layout=True, figsize=(6, 3 * (plot_num + 1)))
        subfigs = fig.subfigures(2, 1, wspace=0.07, height_ratios=[3, 2])

        # make subplots
        axs1 = subfigs[0].subplots(plot_num, 1, sharex=True)
        ap = list()

        # plot candles
        ohlc = ohlc.set_index('time')

        for index, indicator in enumerate(indicator_list):
            # plot indicator
            indicator_columns = self.indicator_dict[indicator]
            for i_c in indicator_columns:
                m = mpf.make_addplot(data[i_c], panel=index + 1, title=indicator, ax=axs1[index + 1], width=2)
                ap.append(m)
            # plot indicator parameters
            self.plot_indicator_parameters(point_type, index, indicator, axs1, indicator_params)
            # plot y-labels from right side
            axs1[index + 1].yaxis.set_label_position("right")
            axs1[index + 1].yaxis.tick_right()
            # plot grid
            axs1[index + 1].grid(which='both', linestyle='--', linewidth=0.3)

        axs1[0].grid(which='both', linestyle='--', linewidth=0.3)

        # set x-labels
        axs1[-1].set_xlabel(f"{data['time'].iloc[-1].date()}\n", fontsize=14)
        plt.xticks(rotation=30)

        # plot all subplots
        mpf.plot(ohlc, type='candle', ax=axs1[0], addplot=ap, warn_too_much_data=1001, style='yahoo',
                 axtitle=f'{ticker}-{timeframe}', ylabel='')

        # plot point of trade
        self.plot_point(point_type, data, axs1[0])

        # plot levels
        # self.plot_levels(data, levels, axs1)

        # Plot signal statistics
        pct_right_prognosis = [s[0] for s in statistics[0]]
        pct_price_diff_mean = [s[1] for s in statistics[0]]
        pct_price_diff_std = [s[2] for s in statistics[0]]
        pct_price_diff_mean_plus_std = [a + b for a, b in zip(pct_price_diff_mean, pct_price_diff_std)]
        pct_price_diff_mean_minus_std = [a - b for a, b in zip(pct_price_diff_mean, pct_price_diff_std)]

        # make subplots
        axs2 = subfigs[1].subplots(2, 1, sharex=True)

        # make plots
        axs2[0].plot(pct_right_prognosis, linewidth=2, color='green')
        axs2[0].yaxis.set_label_position("right")
        axs2[0].yaxis.tick_right()
        axs2[1].plot(pct_price_diff_mean_plus_std, linewidth=1.5, linestyle='--')
        axs2[1].plot(pct_price_diff_mean_minus_std, linewidth=1.5, linestyle='--')
        axs2[1].plot(pct_price_diff_mean, linewidth=2)
        axs2[1].yaxis.set_label_position("right")
        axs2[1].yaxis.tick_right()
        # plot grid
        axs2[0].grid(which='both', linestyle='--', linewidth=0.3)
        axs2[1].grid(which='both', linestyle='--', linewidth=0.3)

        # set title
        if point_type == 'buy':
            title = 'СTATИСТИКА СИГНАЛА НА ПОКУПКУ'
        else:
            title = 'СTATИСТИКА СИГНАЛА НА ПРОДАЖУ'
        axs2[0].set_title(f'{title}\n\nВероятность правильного\nдвижения цены после сигнала\n'
                          f'(в среднем - {round(sum(pct_right_prognosis)/len(pct_right_prognosis), 2)})%', fontsize=15)
        axs2[1].set_title('Средняя разница между текущей ценой актива\nи его ценой во время сигнала', fontsize=15)

        # set x-ticks
        xticklabels = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120']

        axs2[1].set_xticks(np.arange(1, 25, 2))
        axs2[1].set_xticklabels(xticklabels)
        plt.xticks(rotation=30)

        # set x-labels
        axs2[1].set_xlabel(f"Время после сигнала, в минутах", fontsize=14)

        # set y-labels
        axs2[0].set_ylabel("Вероятность, %", fontsize=10.5)
        axs2[1].set_ylabel("Разница цен %", fontsize=10.5)

        # save plot to file
        filename = self.save_plot(ticker, timeframe, data)

        # close figure
        plt.close()

        return filename