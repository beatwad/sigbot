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

    # df_btc.to_pickle('test_BTCUSDT_5m.pkl')
    # df_eth.to_pickle('test_ETHUSDT_5m.pkl')
    # buy.to_pickle('signal_stat/eth_buy_stat.pkl')
    # sell.to_pickle('signal_stat/eth_sell_stat.pkl')
    result = ss.write_stat(dfs, signal_points)
    assert result['stat']['buy'].equals(expected['buy'])
    assert result['stat']['sell'].equals(expected['sell'])


total_buy = pd.concat([buy_btc, buy_eth])
total_sell = pd.concat([sell_btc, sell_eth])
total_buy['pattern'] = "STOCH_RSI"
total_sell['pattern'] = "STOCH_RSI"

expected1 = ([(1.1729, 0.1, 0.32), (0.9453, 0.15, 0.27), (1.1223, 0.28, 0.29), (1.1364, 0.22, 0.12),
              (1.8819, 0.37, 0.34), (2.4659, 0.56, 0.22), (2.5878, 0.46, 0.3), (2.5878, 0.36, 0.3), (2.745, 0.45, 0.48),
              (1.8137, 0.48, 0.56), (2.0771, 0.48, 0.76), (2.2103, 0.56, 0.92), (2.2103, 0.35, 0.95),
              (2.2103, 0.29, 0.88), (2.2103, 0.22, 0.78), (2.2103, 0.18, 0.92), (2.2555, 0.18, 1.08),
              (2.1459, 0.18, 0.95), (2.1275, 0.04, 0.95), (2.1054, -0.03, 1.04), (2.1054, 0.16, 1.02),
              (2.1054, 0.15, 0.98), (2.1054, 0.11, 1.02), (2.1054, 0.2, 0.87)], 4)
expected2 = ([(1.0885, -0.15, 0.09), (0.8591, -0.12, 0.17), (0.5347, 0.01, 0.18), (0.5678, -0.05, 0.29),
              (0.7063, 0.04, 0.35), (0.8447, 0.02, 0.44), (0.8424, 0.08, 0.32), (0.8424, 0.1, 0.41),
              (0.8303, 0.11, 0.43), (0.8026, 0.01, 0.54), (0.7413, 0.01, 0.63), (0.8307, -0.02, 0.62),
              (0.8084, -0.2, 0.65), (1.0562, -0.15, 0.91), (1.0793, -0.16, 0.93), (1.0793, -0.19, 0.96),
              (1.1301, -0.34, 0.79), (1.1301, -0.3, 0.8), (1.1301, -0.3, 0.87), (1.1301, -0.27, 0.81),
              (1.1301, -0.26, 0.74), (1.1301, -0.24, 0.75), (1.2411, -0.2, 1.03), (1.2411, -0.14, 1.04)], 5)


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

#
# buy_btc_close1 = buy_btc.iloc[:1]
# buy_btc_close1['time'] = pd.to_datetime('2022-08-21 3:40:00')
#
# buy_btc_close2 = buy_btc.iloc[:1]
# buy_btc_close2['time'] = pd.to_datetime('2022-08-22 22:40:00')
#
# sell_btc_close1 = sell_btc[2:]
# sell_btc_close1['time'] = pd.to_datetime('2022-08-23 11:55:00')
#
# buy_btc_exp = buy_btc.copy()
# buy_btc_exp.loc[0, 'time'] = pd.to_datetime('2022-08-22 22:30:00')
#
# sell_btc_exp = sell_btc.copy()
# sell_btc_exp.loc[2, 'time'] = pd.to_datetime('2022-08-23 11:50:00')
