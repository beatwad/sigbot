import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib import rcParams

matplotlib.use('Agg')
# style.use('dark_background')
rcParams['font.family'] = 'Nimbus Sans'


class Visualizer:
    type = 'Visualizer'
    ticker_color = 'white'
    border_color = 'white'
    background_color = '#010113'
    stat_color_1 = '#0ED6F1'
    stat_std_color_1 = '#19E729'
    stat_std_color_2 = '#E73B19'
    stat_color_2 = '#EE4B1A'

    def __init__(self, **params):
        # Get working and higher timeframes
        self.working_timeframe = params['Timeframes']['work_timeframe']
        self.higher_timeframe = params['Timeframes']['higher_timeframe']
        # Get Visualizer parameters
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
        # dict for storing previous statistics values
        self.prev_stat_dict = dict()

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

    def plot_point(self, point_type: str, data: pd.DataFrame, ax: plt.axis, index=0) -> None:
        """ Plot trade point """
        if index > 0:
            color = 'blue'
        elif point_type == 'buy':
            color = 'green'
        else:
            color = 'red'
        if point_type == 'buy':
            ax.scatter(self.plot_width-index, data['close'].iloc[-1-index], s=50, color=color)
        else:
            ax.scatter(self.plot_width-index, data['close'].iloc[-1-index], s=50, color=color)

    @staticmethod
    def plot_levels(data: pd.DataFrame, levels: list, axs: plt.axis) -> None:
        """ Plot support and resistance levels"""
        for level in levels:
            if data['low'].min() <= level[0] <= data['high'].max():  # and level[1] == 3:
                axs[0].axhline(y=level[0], color='b', linestyle='dotted', linewidth=1.5)

    def save_plot(self, ticker, timeframe, pattern, data):
        filename = f"{self.image_path}/{ticker}_{timeframe}_{pattern}_{data['time'].iloc[-1]}.png"
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
        ticker = ticker[:-4] + '/' + ticker[-4:]
        return ticker

    @staticmethod
    def statistics_change(prev_mean_right_prognosis, mean_right_prognosis):
        """ Measure statistics difference between previous signal and current signal """
        stat_diff = round(mean_right_prognosis - prev_mean_right_prognosis, 2)
        if stat_diff < 0:
            return f'= уменьшилась на {abs(stat_diff)}%'
        if stat_diff > 0:
            return f'= выросла на {stat_diff}%'
        return '= без изменений'

    def create_plot(self, dfs, point, levels):
        # get necessary info
        ticker, timeframe, point_index, point_type, sig_time, pattern, plot_path, exchange_list, statistics, y = point
        df_working = dfs[ticker][self.working_timeframe]['data']
        df_working = df_working.loc[point_index - self.plot_width:point_index]
        ohlc = df_working[['time', 'open', 'high', 'low', 'close', 'volume']].set_index('time')
        # get indicator list
        indicator_list = [p[0] for p in pattern if p[0] not in self.level_indicators]
        indicator_params = [p[1] for p in pattern if p not in self.level_indicators]

        # check if PriceChange indicator is in indicator list to make a special plot
        if 'PriceChange' in indicator_list:
            plot_num = len(indicator_list)
            point_index = indicator_list.index('PriceChange')
            price_index = indicator_params[point_index]
            del indicator_params[point_index]
            indicator_list.remove('PriceChange')
            has_price_change_flag = True
            candles_height = 1.5
            plot_height_mult = 2.5
        else:
            price_index = 0
            plot_num = len(indicator_list) + 1
            has_price_change_flag = False
            candles_height = 3
            plot_height_mult = 1.7

        # Plot signals
        # make subfigs
        fig = plt.figure(constrained_layout=True, figsize=(plot_height_mult * (plot_num + 1), 3 * (plot_num + 1)))
        fig.patch.set_facecolor(self.background_color)
        # If linear regression is in indicator list - remove it from list and plot one more plot with higher timeframe
        # candles and linear regression indicator
        if 'LinearReg' in indicator_list:
            indicator_list.remove('LinearReg')
            plot_num -= 1
            subfigs_num = 3
            subfigs = fig.subfigures(subfigs_num, 1, wspace=0, height_ratios=[candles_height, 1.5, 2.5])
            # plot higher timeframe with linear regression
            subfigs[1].patch.set_facecolor(self.background_color)
            axs_higher = subfigs[1].subplots(1, 1, sharex=True)
            df_higher = dfs[ticker][self.higher_timeframe]['data']
            # get corresponding to signal_time index of dataframe with higher timeframe candles
            try:
                oh_index = df_higher[(df_higher['time'].dt.year == sig_time.year) &
                                     (df_higher['time'].dt.month == sig_time.month) &
                                     (df_higher['time'].dt.day == sig_time.day) &
                                     (df_higher['time'].dt.hour == sig_time.hour)].index[0]
            except IndexError:
                oh_index = df_higher.idxmax()
            df_higher = df_higher.loc[max(oh_index - self.plot_width * 2, 0):oh_index + 1].reset_index(drop=True)
            ohlc_higher = df_higher[['time', 'open', 'high', 'low', 'close', 'volume']].set_index('time')
            # plot candles
            mpf.plot(ohlc_higher, type='candle', ax=axs_higher, warn_too_much_data=1001, style='yahoo',
                     ylabel='', returnfig=True)
            # plot linear regression indicator
            if point_type == 'buy':
                axs_higher.plot(df_higher['linear_reg'], linewidth=2, color='green')
            else:
                axs_higher.plot(df_higher['linear_reg'], linewidth=2, color='red')
            # plot grid
            axs_higher.grid(which='both', linestyle='--', linewidth=0.3)
            # set ticker color
            axs_higher.tick_params(axis='x', colors=self.ticker_color)
            axs_higher.tick_params(axis='y', colors=self.ticker_color)
            # set background color
            axs_higher.patch.set_facecolor(self.background_color)
            # set border color
            axs_higher.spines['bottom'].set_color(self.border_color)
            axs_higher.spines['top'].set_color(self.border_color)
            axs_higher.spines['right'].set_color(self.border_color)
            axs_higher.spines['left'].set_color(self.border_color)
            # plot title
            axs_higher.set_title(f'{self.process_ticker(ticker)} - {self.higher_timeframe} - Тренд', fontsize=14,
                                    color=self.ticker_color)
        else:
            subfigs_num = 2
            subfigs = fig.subfigures(subfigs_num, 1, wspace=0, height_ratios=[candles_height, 2.5])

        # make subplots
        axs1 = subfigs[0].subplots(plot_num, 1, sharex=True)

        try:
            axs1_0 = axs1[0]
        except TypeError:
            axs1_0 = axs1
        subfigs[0].patch.set_facecolor(self.background_color)
        subfigs[-1].patch.set_facecolor(self.background_color)
        ap = list()

        # plot candles
        for index, indicator in enumerate(indicator_list):
            # plot indicator
            indicator_columns = self.indicator_dict[indicator]
            for i_c in indicator_columns:
                m = mpf.make_addplot(df_working[i_c], panel=index + 1, title=indicator, ax=axs1[index + 1], width=2)
                ap.append(m)
            # plot indicator parameters
            self.plot_indicator_parameters(point_type, index, indicator, axs1, indicator_params)
            # plot y-labels from right side
            axs1[index + 1].yaxis.set_label_position("right")
            axs1[index + 1].yaxis.tick_right()
            # plot grid
            axs1[index + 1].grid(which='both', linestyle='--', linewidth=0.3)
            # set title
            axs1[index + 1].set_title(indicator, fontsize=14, color=self.ticker_color)
            # set ticker color
            axs1[index + 1].tick_params(axis='x', colors=self.ticker_color)
            axs1[index + 1].tick_params(axis='y', colors=self.ticker_color)
            # set background color
            axs1[index + 1].patch.set_facecolor(self.background_color)
            # set border color
            axs1[index + 1].spines['bottom'].set_color(self.border_color)
            axs1[index + 1].spines['top'].set_color(self.border_color)
            axs1[index + 1].spines['right'].set_color(self.border_color)
            axs1[index + 1].spines['left'].set_color(self.border_color)

        # plot candles
        axs1_0.grid(which='both', linestyle='--', linewidth=0.3)
        # set ticker color
        axs1_0.tick_params(axis='x', colors=self.ticker_color)
        axs1_0.tick_params(axis='y', colors=self.ticker_color)
        # set background color
        axs1_0.patch.set_facecolor(self.background_color)
        # set border color
        axs1_0.spines['bottom'].set_color(self.border_color)
        axs1_0.spines['top'].set_color(self.border_color)
        axs1_0.spines['right'].set_color(self.border_color)
        axs1_0.spines['left'].set_color(self.border_color)

        # set x-labels
        # axs1[-1].set_xlabel(f"\n{data['time'].iloc[-1].date()}\n", fontsize=14)
        plt.xticks(rotation=30)

        # plot all subplots
        mpf.plot(ohlc, type='candle', ax=axs1_0, addplot=ap, warn_too_much_data=1001, style='yahoo',
                 ylabel='', returnfig=True)

        # plot titles
        axs1_0.set_title(f'{self.process_ticker(ticker)} - {timeframe} - '
                         f'{df_working["time"].iloc[-1].date().strftime("%d.%m.%Y")}', fontsize=14,
                         color=self.ticker_color)
        for index, indicator in enumerate(indicator_list):
            axs1[index + 1].set_title(indicator, fontsize=14, color=self.ticker_color)

        # plot point of trade
        self.plot_point(point_type, df_working, axs1_0)
        # plot
        if has_price_change_flag:
            self.plot_point(point_type, df_working, axs1_0, price_index)

        # plot levels
        # self.plot_levels(data, levels, axs1)

        # Plot signal statistics
        pct_right_prognosis = [s[0] for s in statistics[0]]
        pct_price_diff_mean = [s[1] for s in statistics[0]]
        pct_price_diff_std = [s[2] for s in statistics[0]]
        pct_price_diff_mean_plus_std = [a + b for a, b in zip(pct_price_diff_mean, pct_price_diff_std)]
        pct_price_diff_mean_minus_std = [a - b for a, b in zip(pct_price_diff_mean, pct_price_diff_std)]

        # get previous percent of right prognosis and save current percent to statistics dictionary
        mean_right_prognosis = round(sum(pct_right_prognosis)/len(pct_right_prognosis), 2)
        if 'PriceChange' in str(pattern[0][0]):
            key = str([pattern[0][0]] + pattern[1:])
        else:
            key = str(pattern)
        # check if pattern and trade type are in statistics dictionary
        if key in self.prev_stat_dict and self.prev_stat_dict[key] and point_type in self.prev_stat_dict[key]:
            prev_mean_right_prognosis = self.prev_stat_dict[key][point_type]
        else:
            prev_mean_right_prognosis = mean_right_prognosis
            self.prev_stat_dict[key] = dict()

        self.prev_stat_dict[key][point_type] = mean_right_prognosis

        # get change of statistics
        stat_change = self.statistics_change(prev_mean_right_prognosis, mean_right_prognosis)

        # make subplots
        axs2 = subfigs[-1].subplots(2, 1, sharex=True)

        # make plots
        axs2[0].plot(pct_right_prognosis, linewidth=2, color=self.stat_color_1)
        axs2[0].yaxis.set_label_position("right")
        axs2[0].yaxis.tick_right()
        axs2[1].plot(pct_price_diff_mean_plus_std, linewidth=1.5, linestyle='--', color=self.stat_std_color_1)
        axs2[1].plot(pct_price_diff_mean_minus_std, linewidth=1.5, linestyle='--', color=self.stat_std_color_2)
        axs2[1].plot(pct_price_diff_mean, linewidth=2, color=self.stat_color_2)
        axs2[1].yaxis.set_label_position("right")
        axs2[1].yaxis.tick_right()
        # plot grid
        axs2[0].grid(which='both', linestyle='--', linewidth=0.3)
        axs2[1].grid(which='both', linestyle='--', linewidth=0.3)

        # set title
        if point_type == 'buy':
            title = '\nСTATИСТИКА СИГНАЛА НА ПОКУПКУ'
        else:
            title = '\nСTATИСТИКА СИГНАЛА НА ПРОДАЖУ'
        axs2[0].set_title(f'{title}\n\nВероятность правильного движения цены после сигнала\n'
                          f'(в среднем - {mean_right_prognosis}% {stat_change})',
                          fontsize=13, color=self.ticker_color)
        axs2[1].set_title('Средняя разница между текущей ценой актива\nи его ценой во время сигнала + '
                          'среднее отклонение цены', fontsize=13, color=self.ticker_color)

        # set x-ticks
        xticklabels = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120']
        # set ticker color
        axs2[0].tick_params(axis='x', colors=self.ticker_color)
        axs2[0].tick_params(axis='y', colors=self.ticker_color)
        # set background color
        axs2[0].patch.set_facecolor(self.background_color)
        # set border color
        axs2[0].spines['bottom'].set_color(self.border_color)
        axs2[0].spines['top'].set_color(self.border_color)
        axs2[0].spines['right'].set_color(self.border_color)
        axs2[0].spines['left'].set_color(self.border_color)

        axs2[1].set_xticks(np.arange(1, 25, 2))
        axs2[1].set_xticklabels(xticklabels)
        plt.xticks(rotation=30)

        # set x-labels
        axs2[1].set_xlabel(f"время после сигнала, в минутах", fontsize=12, color=self.ticker_color)

        # set y-labels
        axs2[0].set_ylabel("вероятность, %", fontsize=9.5, color=self.ticker_color)
        axs2[1].set_ylabel("разница цен %", fontsize=9.5, color=self.ticker_color)

        # set ticker color
        axs2[1].tick_params(axis='x', colors=self.ticker_color)
        axs2[1].tick_params(axis='y', colors=self.ticker_color)

        # set background color
        axs2[1].patch.set_facecolor(self.background_color)

        # set border color
        axs2[1].spines['bottom'].set_color(self.border_color)
        axs2[1].spines['top'].set_color(self.border_color)
        axs2[1].spines['right'].set_color(self.border_color)
        axs2[1].spines['left'].set_color(self.border_color)

        # save plot to file
        filename = self.save_plot(ticker, timeframe, pattern, df_working)

        # close figure
        plt.close()

        return filename


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    from matplotlib import rcParams

    matplotlib.use('Agg')
    style.use('dark_background')
    rcParams['font.family'] = 'URW Gothic'
    print(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))
