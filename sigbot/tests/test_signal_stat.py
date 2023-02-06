import pytest
from os import environ
import pandas as pd
from config.config import ConfigFactory
from signal_stat.signal_stat import SignalStat

# Set environment variable
environ["ENV"] = "test"

configs = ConfigFactory.factory(environ).configs

df_btc = pd.read_pickle('test_BTCUSDT_5m.pkl')
df_eth = pd.read_pickle('test_ETHUSDT_5m.pkl')
points_btc = [('BTCUSDT', '5m', 0, 'buy', pd.to_datetime('2022-08-21 3:45:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 20, 'sell', pd.to_datetime('2022-08-21 8:25:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 98, 'sell', pd.to_datetime('2022-08-21 11:55:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 513, 'buy', pd.to_datetime('2022-08-22 22:30:00'),
               'STOCH_RSI_LinearReg', [], [], [], []
               ),
              ('BTCUSDT', '5m', 576, 'sell', pd.to_datetime('2022-08-23 03:45:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 673, 'sell', pd.to_datetime('2022-08-23 11:50:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 745, 'buy', pd.to_datetime('2022-08-23 17:50:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 977, 'sell', pd.to_datetime('2022-08-24 12:05:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 995, 'buy', pd.to_datetime('2022-08-24 14:40:00'),
               'STOCH_RSI_LinearReg', [], [], [], []
               )]

points_eth = [('ETHUSDT', '5m', 45, 'sell', pd.to_datetime('2022-08-21 09:00:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('ETHUSDT', '5m', 47, 'buy', pd.to_datetime('2022-08-21 09:10:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('ETHUSDT', '5m', 97, 'sell', pd.to_datetime('2022-08-21 12:30:00'),
               'STOCH_RSI_LinearReg', [], [], [], []
               ),
              ('ETHUSDT', '5m', 512, 'buy', pd.to_datetime('2022-08-22 23:05:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('ETHUSDT', '5m', 645, 'buy', pd.to_datetime('2022-08-23 10:10:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('ETHUSDT', '5m', 840, 'sell', pd.to_datetime('2022-08-24 02:25:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('ETHUSDT', '5m', 985, 'sell', pd.to_datetime('2022-08-24 12:00:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('ETHUSDT', '5m', 990, 'buy', pd.to_datetime('2022-08-24 12:25:00'),
               'STOCH_RSI_LinearReg', [], [], [], []
               )
              ]


buy_btc = pd.read_pickle('signal_stat/btc_buy_stat.pkl')
sell_btc = pd.read_pickle('signal_stat/btc_sell_stat.pkl')
buy_eth = pd.read_pickle('signal_stat/eth_buy_stat.pkl')
sell_eth = pd.read_pickle('signal_stat/eth_sell_stat.pkl')
expected_write_stat = [{'buy': buy_btc, 'sell': sell_btc},
                       {'buy': buy_eth, 'sell': sell_eth}]


@pytest.mark.parametrize('signal_points, expected',
                         [
                          (points_btc, expected_write_stat[0]),
                          (points_eth, expected_write_stat[1])
                          ], ids=repr)
def test_write_stat(signal_points, expected):
    ss = SignalStat(**configs)
    dfs = {'stat': {'buy': pd.DataFrame(columns=['time', 'ticker', 'timeframe', 'pattern']),
                    'sell': pd.DataFrame(columns=['time', 'ticker', 'timeframe', 'pattern'])},
           'BTCUSDT': {'5m': {'data': {'buy': df_btc, 'sell': df_btc}, 'levels': []}},
           'ETHUSDT': {'5m': {'data': {'buy': df_eth, 'sell': df_eth}, 'levels': []}}}

    result = ss.write_stat(dfs, signal_points)
    assert result['stat']['buy'].equals(expected['buy'])
    assert result['stat']['sell'].equals(expected['sell'])


total_buy = pd.concat([buy_btc, buy_eth])
total_sell = pd.concat([sell_btc, sell_eth])
total_buy['pattern'] = "STOCH_RSI"
total_sell['pattern'] = "STOCH_RSI"

expected1 = ([(1.8331, 4.77, 3.52),
              (1.5257, 2.74, 2.5),
              (1.6878, 1.87, 1.58),
              (1.7098, 2.14, 2.83),
              (2.3899, 2.09, 2.04),
              (2.9144, 3.52, 2.79),
              (3.0228, 3.2, 3.43),
              (2.5396, 3.21, 2.99),
              (2.5947, 5.54, 4.39),
              (1.8842, 5.39, 4.53),
              (2.098, 6.15, 5.0),
              (2.2062, 5.96, 5.17),
              (2.2062, 8.65, 1.92),
              (2.2062, 8.14, 2.64),
              (2.2062, 7.57, 3.43),
              (2.2062, 6.93, 4.34),
              (2.2429, 8.24, 2.49),
              (2.1535, 7.41, 3.66),
              (2.1384, 6.76, 4.58),
              (2.1202, 6.9, 4.38),
              (2.1202, 7.66, 3.31),
              (2.1202, 7.28, 3.85),
              (2.1202, 7.61, 3.37),
              (2.1202, 6.46, 5.0)],
             6)
expected2 = ([(1.3246, 3.11, 4.65),
              (0.9127, 4.07, 5.54),
              (0.4161, 2.96, 2.53),
              (0.4463, 0.34, 0),
              (0.5712, 2.43, 2.67),
              (0.637, 1.6, 2.04),
              (0.6979, 1.93, 2.23),
              (0.7715, 2.0, 2.3),
              (0.7785, 3.16, 2.64),
              (0.8258, 2.58, 2.08),
              (0.7723, 2.52, 0),
              (0.8457, 1.42, 2.19),
              (0.7769, 1.31, 1.76),
              (0.9689, 2.66, 3.68),
              (0.955, 2.95, 3.85),
              (0.9275, 2.44, 2.83),
              (0.9646, 2.53, 2.36),
              (0.9646, 2.76, 2.74),
              (0.9225, 2.88, 3.24),
              (0.9066, 2.48, 2.75),
              (0.9066, 2.58, 3.02),
              (0.9066, 2.47, 3.5),
              (0.9826, 3.48, 4.9),
              (0.9826, 4.75, 5.38)],
             6)


@pytest.mark.parametrize('ttype, pattern, expected',
                         [
                          ('buy', 'STOCH_RSI_LinearReg',
                           ([(0, 0, 0) for _ in range(24)], 0)),
                          ('buy', 'STOCH_RSI', expected1),
                          ('sell', 'STOCH_RSI', expected2)
                          ], ids=repr)
def test_calculate_total_stat(ttype, pattern, expected):
    ss = SignalStat(**configs)
    dfs = {'stat': {'buy': total_buy,
                    'sell': total_sell}}
    result = ss.calculate_total_stat(dfs, ttype, pattern)
    assert result == expected


get_result_price_after_period_1 = [
    [21269.31, 21280.5, 21268.08, 21300.0, 21299.03, 21296.85, 21290.94, 21309.82, 21364.07, 21373.32, 21444.72,
     21420.0, 21548.71, 21438.98, 21390.38, 21387.68, 21406.93, 21379.96, 21367.05, 21371.89, 21369.1, 21350.14,
     21361.22, 21343.7],
    [21193.12, 21222.02, 21211.22, 21253.0, 21270.5, 21251.49, 21239.08, 21255.87, 21275.71, 21314.63, 21340.05,
     21381.65, 21398.59, 21352.22, 21337.38, 21340.93, 21354.97, 21323.08, 21325.13, 21334.9, 21330.1, 21315.12,
     21318.0, 21316.22],
    [21263.08, 21253.28, 21256.95, 21281.0, 21293.67, 21252.9, 21283.87, 21277.82, 21329.56, 21340.16, 21407.62,
     21400.47, 21426.86, 21368.32, 21345.76, 21359.89, 21376.86, 21344.57, 21336.13, 21365.37, 21334.55, 21340.79,
     21337.0, 21331.57]
]
get_result_price_after_period_2 = [
    [1614.0, 1608.87, 1609.8, 1617.46, 1615.63, 1616.0, 1614.85, 1620.94, 1621.52, 1618.72, 1618.08, 1618.14, 1619.3,
     1618.0, 1619.17, 1623.56, 1622.23, 1619.82, 1617.31, 1618.47, 1621.3, 1620.44, 1620.49, 1621.82],
    [1607.42, 1600.4, 1605.29, 1606.65, 1610.5, 1611.4, 1610.01, 1611.11, 1617.09, 1616.2, 1613.8, 1615.2, 1616.1,
     1615.88, 1617.27, 1618.23, 1619.13, 1611.28, 1614.18, 1615.19, 1617.82, 1616.32, 1617.53, 1617.47],
    [1608.28, 1606.11, 1608.16, 1613.27, 1614.69, 1612.38, 1611.97, 1620.76, 1617.63, 1618.07, 1615.44, 1617.43,
     1616.37, 1617.66, 1618.6, 1619.62, 1619.17, 1614.4, 1615.67, 1618.28, 1620.44, 1618.34, 1617.54, 1621.12]
]


@pytest.mark.parametrize('df, point, signal_price, signal_smooth_price, expected',
                         [
                             (df_btc, ['BTCUSDT', '5m', 458, 'buy', pd.Timestamp('2022-08-22 17:55:00'), 'STOCH_RSI',
                                       '', '',  [], []], 21201, 21232, get_result_price_after_period_1),
                             (df_eth, ['ETHUSDT', '5m', 150, 'buy', pd.Timestamp('2022-08-22 22:30:00'), 'Pattern',
                                       '', '',  [], []], 1500, 1510, get_result_price_after_period_2),
                         ], ids=repr)
def test_get_result_price_after_period(df, point, signal_price, signal_smooth_price, expected):
    ticker, timeframe, index, ttype, time, pattern, plot_path, exchange_list, total_stat, ticker_stat = point
    ss = SignalStat(**configs)
    high_result_prices, low_result_prices, close_smooth_prices, atr = ss.get_result_price_after_period(df, index)
    assert expected[0] == high_result_prices
    assert expected[1] == low_result_prices
    assert expected[2] == close_smooth_prices


process_stat_1 = pd.read_pickle('test_btc_process_stat.pkl')
process_stat_2 = pd.read_pickle('test_eth_process_stat.pkl')


@pytest.mark.parametrize('df, point, signal_price, signal_smooth_price, expected',
                         [
                             (df_btc, ['BTCUSDT', '5m', 458, 'buy', pd.Timestamp('2022-08-22 17:55:00'), 'STOCH_RSI',
                                       '', '',  [], []], 21201, 21232, process_stat_1),
                             (df_eth, ['ETHUSDT', '5m', 150, 'buy', pd.Timestamp('2022-08-22 22:30:00'), 'Pattern',
                                       '', '',  [], []], 1500, 1510, process_stat_2),
                         ], ids=repr)
def test_process_statistics(df, point, signal_price, signal_smooth_price, expected):
    ticker, timeframe, index, ttype, time, pattern, plot_path, exchange_list, total_stat, ticker_stat = point
    ss = SignalStat(**configs)
    dfs = {'stat': {'buy': pd.DataFrame(columns=['time', 'ticker', 'timeframe', 'pattern']),
                    'sell': pd.DataFrame(columns=['time', 'ticker', 'timeframe', 'pattern'])}}
    high_result_prices, low_result_prices, close_smooth_prices, atr = ss.get_result_price_after_period(df, index)
    result = ss.process_statistics(dfs, point, signal_price, signal_smooth_price, high_result_prices, low_result_prices,
                                   close_smooth_prices, atr)
    result['stat'][ttype].to_pickle('test_eth_process_stat.pkl')
    assert result['stat'][ttype].equals(expected)


@pytest.mark.parametrize('ticker, index, point_time, pattern, prev_point, expected',
                         [
                             ('BTCUSDT', 90, pd.Timestamp('2022-08-22 22:30:00'), 'STOCH_RSI', (None, None, None),
                             True),
                             ('BTCUSDT', 50, pd.Timestamp('2022-08-22 22:30:00'), 'STOCH_RSI', (None, None, None),
                              False),
                             ('BTCUSDT', 90, pd.Timestamp('2022-08-22 21:30:00'), 'STOCH_RSI',
                              ('BTCUSDT', pd.Timestamp('2022-08-22 21:00:00'), 'STOCH_RSI'), False),
                             ('BTCUSDT', 90, pd.Timestamp('2022-08-22 21:30:00'), 'STOCH_RSI',
                              ('BTCUSDT', pd.Timestamp('2022-08-22 20:00:00'), 'STOCH_RSI'), False),
                             ('BTCUSDT', 90, pd.Timestamp('2022-08-24 14:50:00'), 'STOCH_RSI',
                              (None, None, None), False),
                             ('BTCUSDT', 90, pd.Timestamp('2022-08-24 15:00:00'), 'STOCH_RSI',
                              (None, None, None), True),
                             ('BTCUSDT', 90, pd.Timestamp('2022-08-24 15:00:00'), 'STOCH_RSI',
                              ('BTCUSDT', pd.Timestamp('2022-08-24 14:55:00'), 'STOCH_RSI'), False),
                             ('BTCUSDT', 90, pd.Timestamp('2022-08-24 15:00:00'), 'STOCH_RSI',
                              ('BTCUSDT', pd.Timestamp('2022-08-24 14:40:00'), 'STOCH_RSI'), True),
                         ], ids=repr)
def test_check_close_trades(ticker, index, point_time, pattern, prev_point, expected):
    ss = SignalStat(**configs)
    result = ss.check_close_trades(total_buy, 100, ticker, index, point_time, pattern, prev_point)
    assert result == expected
