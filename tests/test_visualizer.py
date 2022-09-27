import sys
import pytest
from os import environ

import pandas as pd
from config.config import ConfigFactory
from visualizer.visualizer import Visualizer

environ["ENV"] = "test"
# Get configs
configs = ConfigFactory.factory(environ).configs

point1 = ['BNBUSDT', '5m', 816, 'sell', pd.to_datetime('2022-09-27 07:05:00'),
          [('STOCH', (15, 85)), ('RSI', (25, 75))], [],
          ['Binance', 'BinanceFutures'],
          [[(100.0, -0.21, 0.1), (100.0, -0.19, 0.12), (50.0, -0.0, 0.1),
            (0.0, 0.07, 0.1), (0.0, 0.21, 0.3), (50.0, 0.11, 0.2),
            (0.0, 0.12, 0.17), (0.0, 0.16, 0.12), (50.0, 0.12, 0.22),
            (50.0, 0.03, 0.3), (50.0, 0.03, 0.3), (50.0, -0.0, 0.2),
            (50.0, 0.03, 0.15), (50.0, -0.04, 0.1), (0.0, 0.09, 0.08),
            (0.0, 0.11, 0.1), (0.0, 0.3, 0.07), (0.0, 0.37, 0.02),
            (0.0, 0.43, 0.05), (0.0, 0.53, 0.05), (0.0, 0.46, 0.05),
            (0.0, 0.51, 0.03), (0.0, 0.64, 0.1), (0.0, 0.73, 0.18)]], []]

point2 = ['BNBUSDT', '5m', 966, 'buy', pd.to_datetime('2022-09-27 19:35:00'),
          [('STOCH', (15, 85)), ('RSI', (25, 75)), ('LinearReg', ())], [],
          ['Binance', 'BinanceFutures'],
          [[(82.76, 0.21, 0.25), (72.41, 0.23, 0.47), (70.69, 0.24, 0.68),
            (70.69, 0.25, 1.13), (63.79, 0.21, 1.06), (60.34, 0.32, 0.85),
            (60.34, 0.29, 1.01), (60.34, 0.21, 0.88), (58.62, 0.26, 1.08),
            (53.45, 0.13, 1.27), (51.72, 0.08, 1.25), (55.17, 0.09, 1.23),
            (56.9, 0.19, 1.24), (55.17, 0.18, 1.24), (56.9, 0.17, 1.37),
            (56.9, 0.31, 1.43), (55.17, 0.15, 1.48), (50.0, 0.09, 1.45),
            (51.72, 0.23, 1.59), (53.45, 0.11, 1.58), (53.45, 0.13, 1.57),
            (51.72, 0.08, 1.56), (58.62, 0.19, 1.53), (53.45, 0.08, 1.53)]], []]

point3 = ['BNBUSDT', '5m', 980, 'buy', pd.to_datetime('2022-09-27 20:45:00'),
          [('PriceChange', ()), ('LinearReg', ())], [],
          ['Binance', 'BinanceFutures'],
          [[(82.76, 0.21, 0.25), (72.41, 0.23, 0.47), (70.69, 0.24, 0.68),
            (70.69, 0.25, 1.13), (63.79, 0.21, 1.06), (60.34, 0.32, 0.85),
            (60.34, 0.29, 1.01), (60.34, 0.21, 0.88), (58.62, 0.26, 1.08),
            (53.45, 0.13, 1.27), (51.72, 0.08, 1.25), (55.17, 0.09, 1.23),
            (56.9, 0.19, 1.24), (55.17, 0.18, 1.24), (56.9, 0.17, 1.37),
            (56.9, 0.31, 1.43), (55.17, 0.15, 1.48), (50.0, 0.09, 1.45),
            (51.72, 0.23, 1.59), (53.45, 0.11, 1.58), (53.45, 0.13, 1.57),
            (51.72, 0.08, 1.56), (58.62, 0.19, 1.53), (53.45, 0.08, 1.53)]], []]

points = [point1, point2, point3]
keys = ["[('STOCH', (15, 85)), ('RSI', (25, 75))]",
        "[('STOCH', (15, 85)), ('RSI', (25, 75)), ('LinearReg', ())]",
        "['PriceChange', ('LinearReg', ())]"]


@pytest.mark.parametrize('point, expected',
                         [
                          (points[0], keys[0]),
                          (points[1], keys[1]),
                          (points[2], keys[2])
                          ], ids=repr)
def test_get_statistics_dict_key(point, expected):
    vis = Visualizer(**configs)
    pattern = point[5]
    assert vis.get_statistics_dict_key(pattern) == expected


prev_stat_dicts = [dict(),
                   {"[('STOCH', (15, 85)), ('RSI', (25, 75))]": {'sell': 10.0, 'buy': 15},
                    "['PriceChange', ('LinearReg', ())]": {'sell': 50.0, 'buy': 40}},
                   {"[('STOCH', (15, 85)), ('RSI', (25, 75)), ('LinearReg', ())]": {'sell': 10.0, 'buy': None}},
                   {"['PriceChange', ('LinearReg', ())]": {'sell': 10.0, 'buy': 15}},
                   dict()]
prev_mean_pct_right_forecasts = [25, 10, 58.91, None, 15, 40, 15]


@pytest.mark.parametrize('point, prev_stat_dict, expected',
                         [
                          (points[0], prev_stat_dicts[0], prev_mean_pct_right_forecasts[0]),
                          (points[0], prev_stat_dicts[1], prev_mean_pct_right_forecasts[1]),
                          (points[1], prev_stat_dicts[0], prev_mean_pct_right_forecasts[2]),
                          (points[1], prev_stat_dicts[2], prev_mean_pct_right_forecasts[3]),
                          (points[2], prev_stat_dicts[1], prev_mean_pct_right_forecasts[5]),
                          (points[2], prev_stat_dicts[3], prev_mean_pct_right_forecasts[6])
                          ], ids=repr)
def test_get_prev_mean_pct_right_forecast(point, prev_stat_dict, expected):
    vis = Visualizer(**configs)
    vis.prev_stat_dict = prev_stat_dict

    point_type = point[3]
    pattern = point[5]
    statistics = point[8]

    pct_right_forecast = [s[0] for s in statistics[0]]
    mean_pct_right_forecast = round(sum(pct_right_forecast) / len(pct_right_forecast), 2)

    key = vis.get_statistics_dict_key(pattern)
    assert vis.get_prev_mean_pct_right_forecast(key, point_type, mean_pct_right_forecast) == expected


prev_mean_pct_right_forecasts = [None, 58.01, 34.20]
mean_pct_right_forecast = [20, 158.01, 24.20]
stat_results = ['= без изменений', '= выросла на 100.0%', '= уменьшилась на 10.0%']


@pytest.mark.parametrize('prev_mean_pct_right_forecast, mean_pct_right_forecast, expected',
                         [
                          (prev_mean_pct_right_forecasts[0], mean_pct_right_forecast[0], stat_results[0]),
                          (prev_mean_pct_right_forecasts[1], mean_pct_right_forecast[1], stat_results[1]),
                          (prev_mean_pct_right_forecasts[2], mean_pct_right_forecast[2], stat_results[2]),
                          ], ids=repr)
def test_statistics_change(prev_mean_pct_right_forecast, mean_pct_right_forecast, expected):
    vis = Visualizer(**configs)
    assert vis.statistics_change(prev_mean_pct_right_forecast, mean_pct_right_forecast) == expected

