import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib import rcParams

matplotlib.use('Agg')
rcParams['font.family'] = 'DejaVu Sans'


class Visualizer:
    type = 'Visualizer'
    ticker_color = 'white'
    border_color = 'white'
    background_color = '#010113'
    stat_color_1 = '#0ED6F1'
    stat_std_color_1 = '#19E729'
    stat_std_color_2 = '#E73B19'
    stat_color_2 = '#EE4B1A'

    def __init__(self, **configs):
        # Get working and higher timeframes
        self.working_timeframe = configs['Timeframes']['work_timeframe']
        self.higher_timeframe = configs['Timeframes']['higher_timeframe']
        # Get Visualizer parameters
        self.configs = configs[self.type]['params']
        # Path to save plot files
        self.image_path = self.configs['image_path']
        self.indicator_configs = configs['Indicator_signal']
        self.plot_width = self.configs.get('plot_width', 10)
        self.indicator_dict = self.configs.get('indicator_dict', dict())
        self.level_indicators = self.configs.get('level_indicators', list())
        self.boundary_indicators = self.configs.get('boundary_indicators', list())
        # Max number of previous candles for which signal can be searched for
        self.max_prev_candle_limit = self.configs.get('max_prev_candle_limit', 0)
        # dict for storing previous statistics values
        self.prev_stat_dict = dict()
        # list of indicator parameters that can be plotted
        self.indicator_params = configs['Indicator_signal']
        self.indicators_to_plot = ['RSI', 'STOCH']

    def plot_indicator_parameters(self, point_type: str, index: int, indicator: str,
                                  axs: plt.axis, indicator_list: list) -> None:
        """ Plot parameters of indicator (like low or high boundary, etc.)"""
        indicator = indicator_list[index]
        if indicator in self.indicators_to_plot:
            indicator_params = list(self.indicator_params[point_type][indicator]['params'].values())
            if indicator_params:
                if indicator in self.boundary_indicators:
                    if point_type == 'buy':
                        axs[index + 1].axhline(y=indicator_params[0], color='g', linestyle='--', linewidth=1.5)
                    else:
                        axs[index + 1].axhline(y=indicator_params[1], color='r', linestyle='--', linewidth=1.5)

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
    def get_statistics_dict_key(pattern: list) -> str:
        """ Get previous avg E-ratio coefficient and save current E-ratio to statistics dictionary """
        if 'PriceChange' in str(pattern[0][0]):
            key = str([pattern[0][0]] + pattern[1:])
        else:
            key = str(pattern)
        return key
    
    def get_prev_avg_e_ratio_coef(self, key: str, point_type: str, avg_e_ratio_coef: float) -> float:
        """ Get previous E-ratio coefficient and fill the statistic dict by current value of E-ratio """
        if key in self.prev_stat_dict:
            prev_avg_e_ratio_coef = self.prev_stat_dict[key][point_type]
        else:
            prev_avg_e_ratio_coef = avg_e_ratio_coef
            self.prev_stat_dict[key] = {'sell': None, 'buy': None}

        self.prev_stat_dict[key][point_type] = avg_e_ratio_coef
        return prev_avg_e_ratio_coef

    @staticmethod
    def statistics_change(prev_avg_e_ratio_coef, avg_e_ratio_coef):
        """ Measure statistics difference between previous signal and current signal """
        if prev_avg_e_ratio_coef is None:
            return '= без изменений'
        stat_diff = round(avg_e_ratio_coef - prev_avg_e_ratio_coef, 4)
        if stat_diff < 0:
            return f'= уменьшилcя на {abs(stat_diff)}'
        if stat_diff > 0:
            return f'= вырос на {stat_diff}'
        return '= без изменений'

    def create_plot(self, dfs, point, levels):
        # get necessary info
        ticker, timeframe, point_index, point_type, sig_time, pattern, plot_path, exchange_list, statistics, y = point
        df_working = dfs[ticker][self.working_timeframe]['data'][point_type]
        df_working = df_working.loc[point_index - self.plot_width:point_index]
        ohlc = df_working[['time', 'open', 'high', 'low', 'close', 'volume']].set_index('time')
        # get indicator list
        indicator_list = [p for p in pattern.split('_') if p[0] not in self.level_indicators]
        indicator_list_tmp = indicator_list.copy()

        # check if PriceChange indicator is in indicator list to make a special plot
        if 'PriceChange' in indicator_list_tmp:
            plot_num = len(indicator_list)
            indicator_list.remove('PriceChange')
            main_candleplot_ratio = 1.5
            plot_width_mult = 2.5
        elif 'HighVolume' in indicator_list_tmp:
            plot_num = len(indicator_list) + 1
            main_candleplot_ratio = 3
            plot_width_mult = 5
        elif 'MACD' in indicator_list_tmp:
            plot_num = len(indicator_list) + 1
            main_candleplot_ratio = 1.5
            plot_width_mult = 2.5
        else:
            plot_num = len(indicator_list) + 1
            main_candleplot_ratio = 3
            plot_width_mult = 1.8

        # Plot signals
        # make subfigs
        fig = plt.figure(constrained_layout=True, figsize=(plot_width_mult * (plot_num + 1), 3 * (plot_num + 1)))
        fig.patch.set_facecolor(self.background_color)
        # If linear regression is in indicator list - remove it from list and plot one more plot with higher timeframe
        # candles and linear regression indicator
        if 'LinearReg' in indicator_list_tmp or 'MACD' in indicator_list_tmp:
            plot_num -= 1
            subfigs_num = 3
            if 'LinearReg' in indicator_list_tmp:
                indicator_tmp = 'LinearReg'
            else:
                indicator_tmp = 'MACD'
            indicator_list.remove(indicator_tmp)

            subfigs = fig.subfigures(subfigs_num, 1, wspace=0, height_ratios=[main_candleplot_ratio, 2.5, 2.5])
            # plot higher timeframe with linear regression
            subfigs[1].patch.set_facecolor(self.background_color)
            axs_higher = subfigs[1].subplots(2, 1, sharex=True)
            df_higher = dfs[ticker][self.higher_timeframe]['data'][point_type]
            # get corresponding to signal_time index of dataframe with higher timeframe candles
            df_higher = df_higher.loc[max(df_higher.shape[0] - self.plot_width * 2, 0):].reset_index(drop=True)
            ohlc_higher = df_higher[['time', 'open', 'high', 'low', 'close', 'volume']].set_index('time')
            # plot higher timeframe indicator
            indicator_columns = self.indicator_dict[indicator_tmp]
            for i_c in indicator_columns:
                if i_c == 'macdhist':
                    df_higher[i_c].plot(kind='bar', ax=axs_higher[1])
                    plt.xticks(np.arange(min(df_higher.index), max(df_higher.index) + 1, 10))
                else:
                    axs_higher[1].plot(df_higher[i_c], linewidth=2)
                axs_higher[1].yaxis.set_label_position("right")
                axs_higher[1].yaxis.tick_right()
            # plot grid for candles
            axs_higher[0].grid(which='both', linestyle='--', linewidth=0.3)
            # set ticker color
            axs_higher[0].tick_params(axis='x', colors=self.ticker_color)
            axs_higher[0].tick_params(axis='y', colors=self.ticker_color)
            # set background color
            axs_higher[0].patch.set_facecolor(self.background_color)
            # set border color
            axs_higher[0].spines['bottom'].set_color(self.border_color)
            axs_higher[0].spines['top'].set_color(self.border_color)
            axs_higher[0].spines['right'].set_color(self.border_color)
            axs_higher[0].spines['left'].set_color(self.border_color)
            # plot grid for the treand force
            axs_higher[1].grid(which='both', linestyle='--', linewidth=0.3)
            # set ticker color
            axs_higher[1].tick_params(axis='x', colors=self.ticker_color)
            axs_higher[1].tick_params(axis='y', colors=self.ticker_color)
            # set background color
            axs_higher[1].patch.set_facecolor(self.background_color)
            # set border color
            axs_higher[1].spines['bottom'].set_color(self.border_color)
            axs_higher[1].spines['top'].set_color(self.border_color)
            axs_higher[1].spines['right'].set_color(self.border_color)
            axs_higher[1].spines['left'].set_color(self.border_color)
            # plot titles
            if indicator_tmp == 'LinearReg':
                axs_higher[0].set_title(f'{self.process_ticker(ticker)} - {self.higher_timeframe} - Тренд', fontsize=14,
                                        color=self.ticker_color)
                axs_higher[1].set_title('Сила тренда', fontsize=14, color=self.ticker_color)
            elif indicator_tmp == 'MACD':
                axs_higher[0].set_title(f'{self.process_ticker(ticker)} - {self.higher_timeframe} - Тренд', fontsize=14,
                                        color=self.ticker_color)
                axs_higher[1].set_title('MACD', fontsize=14, color=self.ticker_color)
            # plot candles
            mpf.plot(ohlc_higher, type='candle', ax=axs_higher[0], warn_too_much_data=1001, style='yahoo',
                     ylabel='', returnfig=True)

            # workaround to show the last xtick for the higher timeframe candle plot
            format = '%m-%d-%H:%M'
            newxticks = []
            newlabels = []

            # copy and format the existing xticks:
            for xt in axs_higher[0].get_xticks():
                p = int(xt)
                if 0 <= p < len(ohlc_higher):
                    ts = ohlc_higher.index[p]
                    newxticks.append(p)
                    newlabels.append(ts.strftime(format))

            # Here we create the final tick and tick label:
            newxticks.append(len(ohlc_higher) - 1)
            newlabels.append(ohlc_higher.index[len(ohlc_higher) - 1].strftime(format))

            # set the xticks and labels with the new ticks and labels:
            axs_higher[1].set_xticks(newxticks)
            axs_higher[1].set_xticklabels(newlabels, rotation=30)
        elif 'HighVolume' in indicator_list_tmp:
            subfigs_num = 2
            subfigs = fig.subfigures(subfigs_num, 1, wspace=0, height_ratios=[main_candleplot_ratio/3, 0.01])
        else:
            subfigs_num = 2
            subfigs = fig.subfigures(subfigs_num, 1, wspace=0, height_ratios=[main_candleplot_ratio, 2.5])

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
                if i_c == 'volume':
                    m = mpf.make_addplot(df_working[i_c], panel=index + 1, title=indicator, ax=axs1[index + 1],
                                         width=0.5, type='bar')
                else:
                    m = mpf.make_addplot(df_working[i_c], panel=index + 1, title=indicator, ax=axs1[index + 1], width=2)
                ap.append(m)
            # plot indicator parameters
            self.plot_indicator_parameters(point_type, index, indicator, axs1, indicator_list)
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

        # plot all subplots
        mpf.plot(ohlc, type='candle', ax=axs1_0, addplot=ap, warn_too_much_data=1001, style='yahoo',
                 ylabel='', returnfig=True, tz_localize=True)

        # workaround to show the last xtick for the higher timeframe candle plot
        format = '%H:%M'
        newxticks = []
        newlabels = []

        # copy and format the existing xticks:
        for xt in axs1_0.get_xticks():
            p = int(xt)
            if 0 <= p < len(ohlc):
                ts = ohlc.index[p]
                newxticks.append(p)
                newlabels.append(ts.strftime(format))

        # here we create the final tick and tick label:
        newxticks.append(len(ohlc) - 1)
        newlabels.append(ohlc.index[len(ohlc) - 1].strftime(format))

        # set the xticks and labels with the new ticks and labels:
        axs1_0.set_xticks(newxticks)
        axs1_0.set_xticklabels(newlabels, rotation=0)

        # plot titles
        price = df_working["close"].iloc[-1]
        if price > 1:
            price = round(price, 3)
        else:
            price = round(price, 7)
        axs1_0.set_title(f'{self.process_ticker(ticker)} - {timeframe} - {price} $ - '
                         f'{df_working["time"].iloc[-1].date().strftime("%d.%m.%Y")}', fontsize=14,
                         color=self.ticker_color)
        for index, indicator in enumerate(indicator_list):
            axs1[index + 1].set_title(indicator, fontsize=14, color=self.ticker_color)

        # plot point of trade
        self.plot_point(point_type, df_working, axs1_0)
        # plot levels
        # self.plot_levels(data, levels, axs1)

        # Plot signal statistics
        if 'HighVolume' not in indicator_list_tmp:
            pct_right_forecast = [s[0] for s in statistics[0]]
            pct_price_diff_mean = [s[1] for s in statistics[0]]
            pct_price_diff_std = [s[2] for s in statistics[0]]
            pct_price_diff_mean_plus_std = [a + b for a, b in zip(pct_price_diff_mean, pct_price_diff_std)]
            pct_price_diff_mean_minus_std = [a - b for a, b in zip(pct_price_diff_mean, pct_price_diff_std)]

            # get previous percent of right forecast and save current percent to statistics dictionary
            avg_e_ratio_coef = round(sum(pct_right_forecast)/len(pct_right_forecast), 2)

            # get key for statistics dict
            key = self.get_statistics_dict_key(pattern)

            # get previous mean percent of right forecasts and
            # check if pattern and trade type are in statistics dictionary
            prev_avg_e_ratio_coef = self.get_prev_avg_e_ratio_coef(key, point_type, avg_e_ratio_coef)

            # get change of statistics
            stat_change = self.statistics_change(prev_avg_e_ratio_coef, avg_e_ratio_coef)

            # make subplots
            axs2 = subfigs[-1].subplots(2, 1, sharex=True)

            # make plots
            axs2[0].plot(pct_right_forecast, linewidth=2, color=self.stat_color_1)
            axs2[0].axhline(y=1, color='white', linestyle='--', linewidth=1.5)
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
                pro_trade = 'покупка'
                counter_trade = 'продажа'
            else:
                title = '\nСTATИСТИКА СИГНАЛА НА ПРОДАЖУ'
                pro_trade = 'продажа'
                counter_trade = 'покупка'
            axs2[0].set_title(f'{title}\n\nЦеновой перевес (E-ratio) после сигнала \n'
                              f'(E-ratio > 1 - {pro_trade}, E-ratio < 1 - {counter_trade})\n '
                              f'(в среднем - {avg_e_ratio_coef} {stat_change})',
                              fontsize=13, color=self.ticker_color)
            axs2[1].set_title('Средняя разница между текущей ценой актива\nи его ценой во время сигнала + '
                              'среднее отклонение цены', fontsize=13, color=self.ticker_color)

            # set x-ticks
            xticklabels = ['30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '330', '360']
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

            # set x-labels
            axs2[1].set_xlabel(f"время после сигнала, в минутах", fontsize=12, color=self.ticker_color)

            # set y-labels
            axs2[0].set_ylabel("E-ratio", fontsize=9.5, color=self.ticker_color)
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
