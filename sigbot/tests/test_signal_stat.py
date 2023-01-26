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
expected_write_stat = [{'buy': buy_btc, 'sell': sell_btc}, {'buy': buy_eth, 'sell': sell_eth}]


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

expected1 = ([(1.8331, 0.18, 0.26),
              (1.5257, 0.2, 0.21),
              (1.6878, 0.28, 0.23),
              (1.7098, 0.25, 0.12),
              (2.3899, 0.35, 0.32),
              (2.9144, 0.44, 0.27),
              (3.0228, 0.35, 0.3),
              (2.5396, 0.2, 0.3),
              (2.5947, 0.16, 0.45),
              (1.8842, 0.14, 0.52),
              (2.098, 0.18, 0.69),
              (2.2062, 0.17, 0.8),
              (2.2062, 0.04, 0.78),
              (2.2062, 0.0, 0.73),
              (2.2062, 0.0, 0.65),
              (2.2062, 0.0, 0.75),
              (2.2429, 0.0, 0.88),
              (2.1535, 0.0, 0.77),
              (2.1384, 0.0, 0.76),
              (2.1202, 0.0, 0.82),
              (2.1202, 0.0, 0.81),
              (2.1202, 0.0, 0.78),
              (2.1202, 0.0, 0.82),
              (2.1202, 0.0, 0.72)],
             6)
expected2 = ([(1.3246, -0.1, 0.08),
              (0.9127, -0.13, 0.14),
              (0.4161, -0.04, 0.16),
              (0.4463, -0.03, 0.32),
              (0.5712, 0.01, 0.4),
              (0.637, 0.05, 0.49),
              (0.6979, -0.06, 0.43),
              (0.7715, -0.07, 0.54),
              (0.7785, -0.09, 0.49),
              (0.8258, -0.15, 0.62),
              (0.7723, -0.1, 0.65),
              (0.8457, -0.04, 0.61),
              (0.7769, -0.11, 0.6),
              (0.9689, -0.01, 0.85),
              (0.955, -0.08, 0.85),
              (0.9275, -0.1, 0.88),
              (0.9646, -0.17, 0.72),
              (0.9646, -0.15, 0.73),
              (0.9225, -0.15, 0.81),
              (0.9066, -0.14, 0.78),
              (0.9066, -0.13, 0.73),
              (0.9066, -0.12, 0.74),
              (0.9826, -0.1, 0.93),
              (0.9826, -0.07, 0.94)],
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
