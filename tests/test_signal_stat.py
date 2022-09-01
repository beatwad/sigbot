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
points_btc = [(0, 'buy', pd.to_datetime('2022-08-21 3:45:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], []
               ),
              (98, 'sell', pd.to_datetime('2022-08-21 11:55:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], []
               ),
              (513, 'buy', pd.to_datetime('2022-08-22 22:30:00'),
               [('STOCH', (15, 85)), ('RSI', (10, 90)), ('SUP_RES_Robust', ())], [], []
               ),
              (576, 'sell', pd.to_datetime('2022-08-23 03:45:00'),
               [('STOCH', (20, 70)), ('RSI', (25, 75))], [], []
               ),
              (673, 'sell', pd.to_datetime('2022-08-23 11:50:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75)), ('SUP_RES', ())], [], []
               ),
              (995, 'buy', pd.to_datetime('2022-08-24 14:40:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], []
               )]

points_eth = [(55, 'sell', pd.to_datetime('2022-08-21 09:00:00'),
               [('STOCH', (15, 85)), ('RSI', (20, 80))], [], []
               ),
              (97, 'sell', pd.to_datetime('2022-08-21 12:30:00'),
               [('STOCH', (0, 100)), ('RSI', (25, 75))], [], []
               ),
              (512, 'buy', pd.to_datetime('2022-08-22 23:05:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], []
               ),
              (645, 'buy', pd.to_datetime('2022-08-23 10:10:00'),
               [('STOCH', (1, 99)), ('RSI', (25, 75)), ('SUP_RES', ())], [], []
               ),
              (840, 'buy', pd.to_datetime('2022-08-24 02:25:00'),
               [('STOCH', (15, 85)), ('RSI', (35, 80))], [], []
               ),
              (955, 'sell', pd.to_datetime('2022-08-24 12:00:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75)), ('SUP_RES_Robust', ())], [], []
               )]

buy_btc = pd.read_pickle('signal_stat/btc_buy_stat.pkl')
sell_btc = pd.read_pickle('signal_stat/btc_sell_stat.pkl')
buy_eth = pd.read_pickle('signal_stat/eth_buy_stat.pkl')
sell_eth = pd.read_pickle('signal_stat/eth_sell_stat.pkl')
expected_write_stat = [{'buy': buy_btc, 'sell': sell_btc}, {'buy': buy_eth, 'sell': sell_eth}]


@pytest.mark.parametrize('ticker, timeframe, signal_points, expected',
                         [
                          ('BTCUSDT', '5m', points_btc, expected_write_stat[0]),
                          ('ETHUSDT', '5m', points_eth, expected_write_stat[1])
                          ], ids=repr)
def test_get_last_transaction(ticker, timeframe, signal_points, expected):
    ss = SignalStat(**configs)
    dfs = {'stat': {'buy': pd.DataFrame(columns=['time', 'ticker', 'timeframe']),
                    'sell': pd.DataFrame(columns=['time', 'ticker', 'timeframe'])},
           'BTCUSDT': {'5m': {'data': df_btc, 'levels': []}},
           'ETHUSDT': {'5m': {'data': df_eth, 'levels': []}}}
    result = ss.write_stat(dfs, ticker, timeframe, signal_points)
    assert result['stat']['buy'].equals(expected['buy'])
    assert result['stat']['sell'].equals(expected['sell'])
