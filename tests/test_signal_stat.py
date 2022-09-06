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
              ('BTCUSDT', '5m', 995, 'buy', pd.to_datetime('2022-08-24 14:40:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               )]

points_eth = [('ETHUSDT', '5m', 55, 'sell', pd.to_datetime('2022-08-21 09:00:00'),
               [('STOCH', (15, 85)), ('RSI', (20, 80))], [], [], [], []
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
              ('ETHUSDT', '5m', 840, 'buy', pd.to_datetime('2022-08-24 02:25:00'),
               [('STOCH', (15, 85)), ('RSI', (35, 80))], [], [], [], []
               ),
              ('ETHUSDT', '5m', 955, 'sell', pd.to_datetime('2022-08-24 12:00:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75)), ('SUP_RES_Robust', ())], [], [], [], []
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
def test_write_stat(signal_points, expected):
    ss = SignalStat(**configs)
    dfs = {'stat': {'buy': pd.DataFrame(columns=['time', 'ticker', 'timeframe']),
                    'sell': pd.DataFrame(columns=['time', 'ticker', 'timeframe'])},
           'BTCUSDT': {'5m': {'data': df_btc, 'levels': []}},
           'ETHUSDT': {'5m': {'data': df_eth, 'levels': []}}}
    result = ss.write_stat(dfs, signal_points)
    assert result['stat']['buy'].equals(expected['buy'])
    assert result['stat']['sell'].equals(expected['sell'])


total_buy = pd.concat([buy_btc, buy_eth])
total_sell = pd.concat([sell_btc, sell_eth])
total_buy['pattern'] = '[(STOCH, (15, 85)), (RSI, (25, 75))]'
total_buy['result_price'].iloc[3] -= 30
total_buy['price_diff'].iloc[3] = -16.38
total_buy['pct_price_diff'].iloc[3] = -0.0104
total_sell['pattern'] = '[(STOCH, (15, 85)), (RSI, (25, 75))]'


@pytest.mark.parametrize('ttype, pattern, expected',
                         [
                          ('buy', '[(STOCH, (15, 85)), (RSI, (25, 75)), (SUP_RES, ())]', (None, None, None)),
                          ('buy', '[(STOCH, (15, 85)), (RSI, (25, 75))]', (80.0, 0.5078980660529774, 5)),
                          ('sell', '[(STOCH, (15, 85)), (RSI, (25, 75))]', (50.0, 0.25224439585284736, 6))
                          ], ids=repr)
def test_calculate_total_stat(ttype, pattern, expected):
    ss = SignalStat(**configs)
    dfs = {'stat': {'buy': total_buy,
                    'sell': total_sell}}
    result = ss.calculate_total_stat(dfs, ttype, pattern)
    assert result == expected


total_stat = pd.concat([total_buy, total_sell])
total_stat['price_diff'] = total_stat['result_price'] - total_stat['signal_price']
total_stat['pct_price_diff'] = (total_stat['result_price'] - total_stat['signal_price'])/total_stat['signal_price']*100


@pytest.mark.parametrize('ttype, ticker, timeframe, pattern, expected',
                         [
                          ('buy', 'BTCUSDT', '5m', '[(STOCH, (15, 85)), (RSI, (25, 75)), (SUP_RES, ())]',
                           (None, None, None, None)),
                          ('buy', 'BTCUSDT', '5m', '[(STOCH, (15, 85)), (RSI, (25, 75))]',
                           (60.0, 37.4580000000009, 0.1787213822520839, 5)),
                          ('sell', 'BTCUSDT', '5m', '[(STOCH, (15, 85)), (RSI, (25, 75))]',
                           (40.0, 37.4580000000009, 0.1787213822520839, 5)),
                          ('buy', 'ETHUSDT', '5m', '[(STOCH, (15, 85)), (RSI, (25, 75))]',
                           (66.66666666666666, 5.726666666666726, 0.3561355189197408, 6)),
                          ('sell', 'ETHUSDT', '5m', '[(STOCH, (15, 85)), (RSI, (25, 75))]',
                           (33.33333333333333, 5.726666666666726, 0.3561355189197408, 6))
                          ], ids=repr)
def test_calculate_ticker_stat(ttype, ticker, timeframe, pattern, expected):
    ss = SignalStat(**configs)
    dfs = {'stat': {'buy': total_stat,
                    'sell': total_stat}}
    result = ss.calculate_ticker_stat(dfs, ttype, ticker, timeframe, pattern)
    assert result == expected


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
