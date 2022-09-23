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
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('BTCUSDT', '5m', 20, 'sell', pd.to_datetime('2022-08-21 8:25:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('BTCUSDT', '5m', 98, 'sell', pd.to_datetime('2022-08-21 11:55:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('BTCUSDT', '5m', 513, 'buy', pd.to_datetime('2022-08-22 22:30:00'),
               [('STOCH', (15, 85)), ('RSI', (10, 90)), ('SUP_RES_Robust', ())], [], [], [], []
               ),
              ('BTCUSDT', '5m', 576, 'sell', pd.to_datetime('2022-08-23 03:45:00'),
               [('STOCH', (20, 70)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('BTCUSDT', '5m', 673, 'sell', pd.to_datetime('2022-08-23 11:50:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75)), ('SUP_RES', ())], [], [], [], []
               ),
              ('BTCUSDT', '5m', 745, 'buy', pd.to_datetime('2022-08-23 17:50:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('BTCUSDT', '5m', 977, 'sell', pd.to_datetime('2022-08-24 12:05:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('BTCUSDT', '5m', 995, 'buy', pd.to_datetime('2022-08-24 14:40:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               )]

points_eth = [('ETHUSDT', '5m', 45, 'sell', pd.to_datetime('2022-08-21 09:00:00'),
               [('STOCH', (15, 85)), ('RSI', (20, 80))], [], [], [], []
               ),
              ('ETHUSDT', '5m', 47, 'buy', pd.to_datetime('2022-08-21 09:10:00'),
               [('STOCH', (15, 85)), ('RSI', (15, 85))], [], [], [], []
               ),
              ('ETHUSDT', '5m', 97, 'sell', pd.to_datetime('2022-08-21 12:30:00'),
               [('STOCH', (0, 100)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('ETHUSDT', '5m', 512, 'buy', pd.to_datetime('2022-08-22 23:05:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('ETHUSDT', '5m', 645, 'buy', pd.to_datetime('2022-08-23 10:10:00'),
               [('STOCH', (1, 99)), ('RSI', (25, 75)), ('SUP_RES', ())], [], [], [], []
               ),
              ('ETHUSDT', '5m', 840, 'sell', pd.to_datetime('2022-08-24 02:25:00'),
               [('STOCH', (15, 85)), ('RSI', (35, 80))], [], [], [], []
               ),
              ('ETHUSDT', '5m', 985, 'sell', pd.to_datetime('2022-08-24 12:00:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75)), ('SUP_RES_Robust', ())], [], [], [], []
               ),
              ('ETHUSDT', '5m', 990, 'buy', pd.to_datetime('2022-08-24 12:25:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75)), ('SUP_RES_Robust', ())], [], [], [], []
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
           'BTCUSDT': {'5m': {'data': df_btc, 'levels': []}},
           'ETHUSDT': {'5m': {'data': df_eth, 'levels': []}}}
    result = ss.write_stat(dfs, signal_points)
    assert result['stat']['buy'].equals(expected['buy'])
    assert result['stat']['sell'].equals(expected['sell'])


total_buy = pd.concat([buy_btc, buy_eth])
total_sell = pd.concat([sell_btc, sell_eth])
total_buy['pattern'] = "[('STOCH', (15, 85)), ('RSI', (25, 75))]"
total_sell['pattern'] = "[('STOCH', (15, 85)), ('RSI', (25, 75))]"
expected1 = [(100.0, 0.1, 0.32), (75.0, 0.15, 0.27), (75.0, 0.28, 0.29), (100.0, 0.22, 0.12), (100.0, 0.37, 0.34),
             (100.0, 0.56, 0.22), (100.0, 0.46, 0.3), (100.0, 0.36, 0.3), (75.0, 0.45, 0.48), (75.0, 0.48, 0.56),
             (100.0, 0.48, 0.76), (75.0, 0.56, 0.92), (75.0, 0.35, 0.95), (75.0, 0.29, 0.88), (50.0, 0.22, 0.78),
             (50.0, 0.18, 0.92), (50.0, 0.18, 1.08), (50.0, 0.18, 0.95), (50.0, 0.04, 0.95), (50.0, -0.03, 1.04),
             (50.0, 0.16, 1.02), (50.0, 0.15, 0.98), (50.0, 0.11, 1.02), (50.0, 0.2, 0.87)]
expected2 = [(100.0, -0.15, 0.09), (60.0, -0.12, 0.17), (40.0, 0.01, 0.18), (60.0, -0.05, 0.29), (40.0, 0.04, 0.35),
             (40.0, 0.02, 0.44), (40.0, 0.08, 0.32), (40.0, 0.1, 0.41), (40.0, 0.11, 0.43), (40.0, 0.01, 0.54),
             (40.0, 0.01, 0.63), (60.0, -0.02, 0.62), (60.0, -0.2, 0.65), (60.0, -0.15, 0.91), (60.0, -0.16, 0.93),
             (60.0, -0.19, 0.96), (60.0, -0.34, 0.79), (60.0, -0.3, 0.8), (60.0, -0.3, 0.87), (60.0, -0.27, 0.81),
             (60.0, -0.26, 0.74), (60.0, -0.24, 0.75), (60.0, -0.2, 1.03), (60.0, -0.14, 1.04)]


@pytest.mark.parametrize('ttype, pattern, expected',
                         [
                          ('buy', "[('STOCH', (15, 85)), ('RSI', (25, 75)), ('LinearReg', ())]",
                           [(0, 0, 0) for _ in range(24)]),
                          ('buy', "[('STOCH', (15, 85)), ('RSI', (25, 75))]", expected1),
                          ('sell', "[('STOCH', (15, 85)), ('RSI', (25, 75))]", expected2)
                          ], ids=repr)
def test_calculate_total_stat(ttype, pattern, expected):
    ss = SignalStat(**configs)
    dfs = {'stat': {'buy': total_buy,
                    'sell': total_sell}}
    result = ss.calculate_total_stat(dfs, ttype, pattern)
    assert result == expected


# total_stat = pd.concat([total_buy, total_sell])
# total_stat['price_diff'] = total_stat['result_price'] - total_stat['signal_price']
# total_stat['pct_price_diff'] = (total_stat['result_price'] -
#                                 total_stat['signal_price'])/total_stat['signal_price']*100
#
#
# @pytest.mark.parametrize('ttype, ticker, timeframe, pattern, expected',
#                          [
#                           ('buy', 'BTCUSDT', '5m', '[(STOCH, (15, 85)), (RSI, (25, 75)), (SUP_RES, ())]',
#                            (None, None, None, None)),
#                           ('buy', 'BTCUSDT', '5m', '[(STOCH, (15, 85)), (RSI, (25, 75))]',
#                            (60.0, 37.4580000000009, 0.1787213822520839, 5)),
#                           ('sell', 'BTCUSDT', '5m', '[(STOCH, (15, 85)), (RSI, (25, 75))]',
#                            (40.0, 37.4580000000009, 0.1787213822520839, 5)),
#                           ('buy', 'ETHUSDT', '5m', '[(STOCH, (15, 85)), (RSI, (25, 75))]',
#                            (66.66666666666666, 5.726666666666726, 0.3561355189197408, 6)),
#                           ('sell', 'ETHUSDT', '5m', '[(STOCH, (15, 85)), (RSI, (25, 75))]',
#                            (33.33333333333333, 5.726666666666726, 0.3561355189197408, 6))
#                           ], ids=repr)
# def test_calculate_ticker_stat(ttype, ticker, timeframe, pattern, expected):
#     ss = SignalStat(**configs)
#     dfs = {'stat': {'buy': total_stat,
#                     'sell': total_stat}}
#     result = ss.calculate_ticker_stat(dfs, ttype, ticker, timeframe, pattern)
#     assert result == expected


buy_btc_close1 = buy_btc.iloc[:1]
buy_btc_close1['time'] = pd.to_datetime('2022-08-21 3:40:00')

buy_btc_close2 = buy_btc.iloc[:1]
buy_btc_close2['time'] = pd.to_datetime('2022-08-22 22:40:00')

sell_btc_close1 = sell_btc[2:]
sell_btc_close1['time'] = pd.to_datetime('2022-08-23 11:55:00')

buy_btc_exp = buy_btc.copy()
buy_btc_exp.loc[0, 'time'] = pd.to_datetime('2022-08-22 22:30:00')

sell_btc_exp = sell_btc.copy()
sell_btc_exp.loc[2, 'time'] = pd.to_datetime('2022-08-23 11:50:00')


@pytest.mark.parametrize('df, close_df, expected',
                         [
                          (buy_btc, pd.DataFrame(), buy_btc),
                          (sell_btc, pd.DataFrame(), sell_btc),
                          (buy_btc, buy_btc_close1, pd.concat([buy_btc,
                                                               buy_btc_close1]).sort_values('time', ignore_index=True)),
                          (buy_btc, buy_btc_close2, buy_btc_exp),
                          (sell_btc, sell_btc_close1, sell_btc_exp)
                         ], ids=repr)
def test_delete_close_trades(df, close_df, expected):
    df = pd.concat([df, close_df])
    ss = SignalStat(**configs)
    result = ss.delete_close_trades(df)
    assert result.equals(expected)
