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
points_btc = [(995, 'buy'), (98, 'sell'), (513, 'buy'), (576, 'sell'), (673, 'sell')]
points_eth = [(55, 'sell'), (97, 'sell'), (512, 'buy'), (645, 'buy'), (840, 'buy'), (955, 'sell')]

buy_btc = pd.read_pickle('signal_stat/btc_buy_stat.pkl')
sell_btc = pd.read_pickle('signal_stat/btc_sell_stat.pkl')
buy_eth = pd.read_pickle('signal_stat/eth_buy_stat.pkl')
sell_eth = pd.read_pickle('signal_stat/eth_sell_stat.pkl')
expected_write_stat = [{'buy': buy_btc, 'sell': sell_btc}, {'buy': buy_eth, 'sell': sell_eth}]


@pytest.mark.parametrize('ticker, timeframe, signal_points, expected',
                         [
                          ('BTCUSDT', '5m', points_btc, expected_write_stat[0]),
                          # ('ETHUSDT', '5m', points_eth, expected_write_stat[1])
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
