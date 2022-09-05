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
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], []
               ),
              ('BTCUSDT', '5m', 98, 'sell', pd.to_datetime('2022-08-21 11:55:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], []
               ),
              ('BTCUSDT', '5m', 513, 'buy', pd.to_datetime('2022-08-22 22:30:00'),
               [('STOCH', (15, 85)), ('RSI', (10, 90)), ('SUP_RES_Robust', ())], [], []
               ),
              ('BTCUSDT', '5m', 576, 'sell', pd.to_datetime('2022-08-23 03:45:00'),
               [('STOCH', (20, 70)), ('RSI', (25, 75))], [], []
               ),
              ('BTCUSDT', '5m', 673, 'sell', pd.to_datetime('2022-08-23 11:50:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75)), ('SUP_RES', ())], [], []
               ),
              ('BTCUSDT', '5m', 995, 'buy', pd.to_datetime('2022-08-24 14:40:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], []
               )]

points_eth = [('ETHUSDT', '5m', 55, 'sell', pd.to_datetime('2022-08-21 09:00:00'),
               [('STOCH', (15, 85)), ('RSI', (20, 80))], [], []
               ),
              ('ETHUSDT', '5m', 97, 'sell', pd.to_datetime('2022-08-21 12:30:00'),
               [('STOCH', (0, 100)), ('RSI', (25, 75))], [], []
               ),
              ('ETHUSDT', '5m', 512, 'buy', pd.to_datetime('2022-08-22 23:05:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], []
               ),
              ('ETHUSDT', '5m', 645, 'buy', pd.to_datetime('2022-08-23 10:10:00'),
               [('STOCH', (1, 99)), ('RSI', (25, 75)), ('SUP_RES', ())], [], []
               ),
              ('ETHUSDT', '5m', 840, 'buy', pd.to_datetime('2022-08-24 02:25:00'),
               [('STOCH', (15, 85)), ('RSI', (35, 80))], [], []
               ),
              ('ETHUSDT', '5m', 955, 'sell', pd.to_datetime('2022-08-24 12:00:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75)), ('SUP_RES_Robust', ())], [], []
               )]

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
def test_get_last_transaction(signal_points, expected):
    ss = SignalStat(**configs)
    dfs = {'stat': {'buy': pd.DataFrame(columns=['time', 'ticker', 'timeframe']),
                    'sell': pd.DataFrame(columns=['time', 'ticker', 'timeframe'])},
           'BTCUSDT': {'5m': {'data': df_btc, 'levels': []}},
           'ETHUSDT': {'5m': {'data': df_eth, 'levels': []}}}
    result = ss.write_stat(dfs, signal_points)
    assert result['stat']['buy'].equals(expected['buy'])
    assert result['stat']['sell'].equals(expected['sell'])


buy_btc_close1 = buy_btc.iloc[:1]
buy_btc_close1['time'] = pd.to_datetime('2022-08-21 3:40:00')

buy_btc_close2 = buy_btc.iloc[:1]
buy_btc_close2['time'] = pd.to_datetime('2022-08-21 3:50:00')

sell_btc_close1 = sell_btc[2:]
sell_btc_close1['time'] = pd.to_datetime('2022-08-23 11:55:00')

buy_btc_exp = buy_btc.copy()
buy_btc_exp.loc[0, 'time'] = pd.to_datetime('2022-08-21 3:50:00')

sell_btc_exp = sell_btc.copy()
sell_btc_exp.loc[2, 'time'] = pd.to_datetime('2022-08-23 11:55:00')


@pytest.mark.parametrize('df, close_df, expected',
                         [
                          (buy_btc, pd.DataFrame(), buy_btc),
                          (sell_btc, pd.DataFrame(), sell_btc),
                          (buy_btc, buy_btc_close1, buy_btc),
                          (buy_btc, buy_btc_close2, buy_btc_exp),
                          (sell_btc, sell_btc_close1, sell_btc_exp)
                         ], ids=repr)
def test_delete_close_trades(df, close_df, expected):
    df = pd.concat([df, close_df])
    ss = SignalStat(**configs)
    result = ss.delete_close_trades(df)
    assert result.equals(expected)
