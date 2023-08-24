import pytest
import numpy as np
import pandas as pd
from os import environ
from datetime import datetime

import sys
sys.path.append('..')

environ["ENV"] = "test"

from data.get_data import DataFactory
from config.config import ConfigFactory
from signals.find_signal import SignalFactory
from signals.find_signal import FindSignal
from indicators.indicators import IndicatorFactory


# Set dataframe dict
df_btc_1h = pd.read_pickle('test_BTCUSDT_1h.pkl')
df_btc_5m = pd.read_pickle('test_BTCUSDT_5m.pkl')
df_eth_1h = pd.read_pickle('test_ETHUSDT_1h.pkl')
df_eth_5m = pd.read_pickle('test_ETHUSDT_5m.pkl')

# Get configs
configs = ConfigFactory.factory(environ).configs


def create_test_data():
    dfs = {'stat': {'buy': pd.DataFrame(columns=['time', 'ticker', 'timeframe']),
                    'sell': pd.DataFrame(columns=['time', 'ticker', 'timeframe'])},
           'BTCUSDT': {'1h': {'data': {'buy': df_btc_1h, 'sell': df_btc_1h}},
                       '5m': {'data': {'buy': df_btc_5m, 'sell': df_btc_5m}}},
           'ETHUSDT': {'1h': {'data': {'buy': df_eth_1h, 'sell': df_eth_1h}},
                       '5m': {'data': {'buy': df_eth_5m, 'sell': df_eth_5m}}}}

    # Create exchange API
    exchange_api = DataFactory.factory('Binance', **configs)

    # Set data quantity
    data_qty = 20

    # Higher timeframe from which we take levels
    work_timeframe = configs['Timeframes']['work_timeframe']

    # For every exchange, ticker and timeframe in base get cryptocurrency data and write it to correspond dataframe
    for ticker in ['BTCUSDT', 'ETHUSDT']:
        for timeframe in ['1h', '5m']:
            indicators = list()
            if timeframe == work_timeframe:
                indicator_list = configs['Indicator_list']
            else:
                indicator_list = ['SUP_RES']
            for indicator in indicator_list:
                ind_factory = IndicatorFactory.factory(indicator, 'buy', configs)
                if ind_factory:
                    indicators.append(ind_factory)
            # Write indicators to dataframe, update dataframe dict
            dfs = exchange_api.add_indicator_data(dfs, dfs[ticker][timeframe]['data']['buy'], 'buy', indicators, ticker,
                                                  timeframe, data_qty)
    return dfs


btc_expected = [np.array([18, 19, 20, 21, 25, 26, 27, 45, 46, 47, 48])]
eth_expected = [np.array([18, 19])]


@pytest.mark.parametrize('ticker, timeframe, high_bound, expected',
                         [
                          ('BTCUSDT', '5m', 100, []),
                          ('BTCUSDT', '5m', 60, btc_expected[0]),
                          ('ETHUSDT', '5m', 75, eth_expected[0]),
                          ], ids=repr)
def test_higher_bound(mocker, timeframe, ticker, high_bound, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    mocker.patch('log.log.create_logger')
    dfs = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', 'sell', configs)

    df = dfs[ticker][timeframe]['data']['buy'][:50]
    stoch_slowd = df['stoch_slowd']
    stoch_slowd_lag_1 = df['stoch_slowd'].shift(1)
    stoch_slowd_lag_2 = df['stoch_slowd'].shift(2)

    points = stoch_sig.higher_bound(high_bound, stoch_slowd, stoch_slowd_lag_1, stoch_slowd_lag_2)
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [np.array([30, 31, 32])]
eth_expected = [np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 43, 44, 45, 46, 47, 48, 49])]


@pytest.mark.parametrize('ticker, timeframe, low_bound, expected',
                         [
                          ('BTCUSDT', '5m', 0, []),
                          ('BTCUSDT', '5m', 15, btc_expected[0]),
                          ('ETHUSDT', '5m', 30, eth_expected[0]),
                          ], ids=repr)
def test_lower_bound(mocker, timeframe, ticker, low_bound, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', 'sell', configs)

    df = dfs[ticker][timeframe]['data']['buy'][:50]
    stoch_slowk = df['stoch_slowk']
    stoch_slowk_lag_1 = df['stoch_slowk'].shift(1)
    stoch_slowk_lag_2 = df['stoch_slowk'].shift(2)

    points = stoch_sig.lower_bound(low_bound, stoch_slowk, stoch_slowk_lag_1, stoch_slowk_lag_2)
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [np.array([23, 24, 25, 26, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46])]
eth_expected = [np.array([24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 46, 47, 48, 49])]


@pytest.mark.parametrize('ticker, timeframe, expected',
                         [
                          ('BTCUSDT', '5m', btc_expected[0]),
                          ('ETHUSDT', '5m', eth_expected[0]),
                          ], ids=repr)
def test_up_direction(mocker, timeframe, ticker, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', 'sell', configs)

    df = dfs[ticker][timeframe]['data']['buy'][:50]

    points = stoch_sig.up_direction(df['stoch_slowk_dir'])
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [np.array([20, 21, 22, 27, 28, 29, 30, 37, 47, 48, 49])]
eth_expected = [np.array([20, 21, 22, 23, 27, 28, 29, 40, 41, 42, 43, 44, 45])]


@pytest.mark.parametrize('ticker, timeframe, expected',
                         [
                          ('BTCUSDT', '5m', btc_expected[0]),
                          ('ETHUSDT', '5m', eth_expected[0]),
                          ], ids=repr)
def test_down_direction(mocker, timeframe, ticker, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', 'sell', configs)

    df = dfs[ticker][timeframe]['data']['buy'][:50]

    points = stoch_sig.down_direction(df['stoch_slowk_dir'])
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [np.array([27, 28, 37, 38, 47, 48, 65, 66, 79, 90, 91]),
                np.array([24, 25, 32, 33, 39, 40, 55, 56, 76, 77, 80, 81, 99])]
eth_expected = [np.array([27, 28, 40, 41, 57, 58, 70, 71, 82, 83, 97, 98]),
                np.array([25, 26, 31, 32, 47, 48, 67, 68, 72, 73, 91, 92, 99])]


@pytest.mark.parametrize('ticker, timeframe, up, expected',
                         [
                          ('BTCUSDT', '5m', True, btc_expected[0]),
                          ('BTCUSDT', '5m', False, btc_expected[1]),
                          ('ETHUSDT', '5m', True, eth_expected[0]),
                          ('ETHUSDT', '5m', False, eth_expected[1])
                          ], ids=repr)
def test_crossed_lines(mocker, timeframe, ticker, up, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', 'sell', configs)

    df = dfs[ticker][timeframe]['data']['buy'][:100]
    stoch_diff = df['stoch_diff']
    stoch_diff_lag_1 = df['stoch_diff'].shift(1)
    stoch_diff_lag_2 = df['stoch_diff'].shift(2)

    points = stoch_sig.crossed_lines(up, stoch_diff, stoch_diff_lag_1, stoch_diff_lag_2)
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [
                np.array([91, 343, 447, 569, 655, 666]),
                np.array([282, 381, 382, 506, 636, 643, 723, 725, 826, 865]),
                ]
eth_expected = [
                np.array([83, 547, 560, 657, 658, 916, 996]),
                np.array([26,  31,  68,  72, 146, 153, 322, 370, 374, 590, 612, 629, 631, 632, 857, 868, 886])
                ]


@pytest.mark.parametrize('ticker, timeframe, expected',
                         [
                             ('BTCUSDT', '5m', btc_expected),
                             ('ETHUSDT', '5m', eth_expected)
                         ], ids=repr)
def test_find_stoch_signal(mocker, timeframe, ticker, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs = create_test_data()
    stoch_sig_buy = SignalFactory().factory('STOCH', 'buy', configs)
    stoch_sig_sell = SignalFactory().factory('STOCH', 'sell', configs)
    if ticker == 'BTCUSDT':
        dfs[ticker][timeframe]['data']['buy'].loc[447, 'stoch_diff'] *= -1
        dfs[ticker][timeframe]['data']['buy'].loc[446, 'stoch_slowk'] += 3
        dfs[ticker][timeframe]['data']['buy'].loc[446, 'stoch_slowd'] += 3
        dfs[ticker][timeframe]['data']['buy'].loc[447, 'stoch_slowk_dir'] *= -1
        dfs[ticker][timeframe]['data']['buy'].loc[447, 'stoch_slowd_dir'] *= -1
    elif ticker == 'ETHUSDT':
        dfs[ticker][timeframe]['data']['buy'].loc[145, 'stoch_slowd'] -= 1
        dfs[ticker][timeframe]['data']['buy'].loc[145, 'stoch_slowk_dir'] *= -1
        dfs[ticker][timeframe]['data']['buy'].loc[146, 'stoch_slowk_dir'] *= -1
        dfs[ticker][timeframe]['data']['buy'].loc[146, 'stoch_slowd_dir'] *= -1
        dfs[ticker][timeframe]['data']['buy'].loc[146, 'stoch_diff'] *= -1
    df = dfs[ticker][timeframe]['data']['buy']
    buy_points = stoch_sig_buy.find_signal(df)
    sell_points = stoch_sig_sell.find_signal(df)
    buy_indexes = np.where(buy_points == 1)
    sell_indexes = np.where(sell_points == 1)
    assert np.array_equal(buy_indexes[0], expected[0])
    assert np.array_equal(sell_indexes[0], expected[1])


btc_expected = [np.array([25, 27, 46, 59, 71, 73, 78, 91, 93, 96]),
                np.array([24, 89])]
eth_expected = [np.array([3, 10, 17, 25, 38, 40, 44, 63, 65, 70, 83, 85, 88, 95]),
                np.array([2, 16, 71])]


@pytest.mark.parametrize('ticker, timeframe, expected',
                         [
                             ('BTCUSDT', '5m', btc_expected),
                             ('ETHUSDT', '5m', eth_expected)
                         ], ids=repr)
def test_find_price_change_signal(mocker, ticker, timeframe, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs = create_test_data()
    price_change_sig_buy = SignalFactory().factory('PumpDump', 'buy', configs)
    price_change_sig_sell = SignalFactory().factory('PumpDump', 'sell', configs)
    buy_points = price_change_sig_buy.find_signal(dfs[ticker][timeframe]['data']['buy'][:100])
    sell_points = price_change_sig_sell.find_signal(dfs[ticker][timeframe]['data']['buy'][:100])
    buy_indexes = np.where(buy_points == 1)
    sell_indexes = np.where(sell_points == 1)
    assert np.array_equal(buy_indexes[0], expected[0])
    assert np.array_equal(sell_indexes[0], expected[1])


btc_lr_buy_expected_1 = np.load('test_btc_buy_indexes_1.npy')
btc_lr_sell_expected_1 = np.load('test_btc_sell_indexes_1.npy')
btc_lr_buy_expected_2 = np.load('test_btc_buy_indexes_2.npy')
btc_lr_sell_expected_2 = np.load('test_btc_sell_indexes_2.npy')
eth_lr_buy_expected_1 = np.load('test_eth_buy_indexes_1.npy')
eth_lr_sell_expected_1 = np.load('test_eth_sell_indexes_1.npy')
eth_lr_buy_expected_2 = np.load('test_eth_buy_indexes_2.npy')
eth_lr_sell_expected_2 = np.load('test_eth_sell_indexes_2.npy')


@pytest.mark.parametrize('df_higher, df_working, expected',
                         [
                             (df_btc_1h, df_btc_5m, (btc_lr_buy_expected_1, btc_lr_sell_expected_1)),
                             (df_btc_1h[-37:], df_btc_5m[-889:], (btc_lr_buy_expected_2, btc_lr_sell_expected_2)),
                             (df_eth_1h, df_eth_5m, (eth_lr_buy_expected_1, eth_lr_sell_expected_1)),
                             (df_eth_1h[-47:], df_eth_5m[-800:], (eth_lr_buy_expected_2, eth_lr_sell_expected_2)),
                         ], ids=repr)
def test_find_linear_reg_signal(mocker, df_higher, df_working, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    linear_reg_sig_buy = SignalFactory().factory('Trend', 'buy', configs)
    linear_reg_sig_sell = SignalFactory().factory('Trend', 'sell', configs)
    buy_points = linear_reg_sig_buy.find_signal(df_higher, 24, df_working.shape[0], df_btc_5m.shape[0])
    sell_points = linear_reg_sig_sell.find_signal(df_higher, 24, df_working.shape[0], df_btc_5m.shape[0])
    buy_indexes = np.where(buy_points == 1)
    sell_indexes = np.where(sell_points == 1)

    assert np.array_equal(buy_indexes, expected[0])
    assert np.array_equal(sell_indexes, expected[1])


points1 = [['BTCUSDT',
              '5m',
              569,
              'buy',
              pd.Timestamp('2022-08-23 03:10:00'),
              'STOCH_RSI',
              [],
              [],
              [],
              []],
             ['BTCUSDT',
              '5m',
              569,
              'buy',
              pd.Timestamp('2022-08-23 03:10:00'),
              'STOCH_RSI_Trend',
              [],
              [],
              [],
              []],
             ['BTCUSDT',
              '1h',
              774,
              'buy',
              pd.Timestamp('2022-08-15 06:00:00'),
              'Pattern_Trend',
              [],
              [],
              [],
              []]]
points2 = [['BTCUSDT', '5m', 506, 'buy', datetime(2022, 8, 22, 21, 55),
            'STOCH_RSI', [], [], [], []],
           ['BTCUSDT', '5m', 569, 'sell', datetime(2022, 8, 23, 3, 10),
            'STOCH_RSI', [], [], [], []],
           ['BTCUSDT', '5m', 506, 'buy', datetime(2022, 8, 22, 21, 55),
            'STOCH_RSI_Trend', [], [], [], []]]
points3 = [['ETHUSDT', '5m', 370, 'buy', datetime(2022, 8, 22, 11, 15),
            'STOCH_RSI', [], [], [], []],
           ['ETHUSDT', '5m', 629, 'buy', datetime(2022, 8, 23, 8, 50),
            'STOCH_RSI', [], [], [], []],
           ['ETHUSDT', '5m', 631, 'buy', datetime(2022, 8, 23, 9),
            'STOCH_RSI', [], [], [], []],
           ['ETHUSDT', '5m', 83, 'sell', datetime(2022, 8, 21, 11, 20),
            'STOCH_RSI', [], [], [], []],
           ['ETHUSDT', '5m', 83, 'sell', datetime(2022, 8, 21, 11, 20),
            'STOCH_RSI_Trend', [], [], [], []]]
points4 = [['ETHUSDT', '5m', 629, 'buy', datetime(2022, 8, 23, 8, 50),
            'STOCH_RSI', [], [], [], []],
           ['ETHUSDT', '5m', 631, 'buy', datetime(2022, 8, 23, 9),
            'STOCH_RSI', [], [], [], []]]
expected_buy = [points1, points2, points3, points4]


points1 = [['BTCUSDT',
              '5m',
              506,
              'sell',
              pd.Timestamp('2022-08-22 21:55:00'),
              'STOCH_RSI',
              [],
              [],
              [],
              []]]
points2 = [['BTCUSDT',
              '5m',
              506,
              'sell',
              pd.Timestamp('2022-08-22 21:55:00'),
              'STOCH_RSI',
              [],
              [],
              [],
              []]]
points3 = [['ETHUSDT',
              '5m',
              370,
              'sell',
              pd.Timestamp('2022-08-22 11:15:00'),
              'STOCH_RSI',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '5m',
              629,
              'sell',
              pd.Timestamp('2022-08-23 08:50:00'),
              'STOCH_RSI',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '5m',
              631,
              'sell',
              pd.Timestamp('2022-08-23 09:00:00'),
              'STOCH_RSI',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '5m',
              370,
              'sell',
              pd.Timestamp('2022-08-22 11:15:00'),
              'STOCH_RSI_Trend',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '5m',
              629,
              'sell',
              pd.Timestamp('2022-08-23 08:50:00'),
              'STOCH_RSI_Trend',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '5m',
              631,
              'sell',
              pd.Timestamp('2022-08-23 09:00:00'),
              'STOCH_RSI_Trend',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '1h',
              198,
              'sell',
              pd.Timestamp('2022-07-22 06:00:00'),
              'Pattern_Trend',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '1h',
              199,
              'sell',
              pd.Timestamp('2022-07-22 07:00:00'),
              'Pattern_Trend',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '1h',
              376,
              'sell',
              pd.Timestamp('2022-07-29 16:00:00'),
              'Pattern_Trend',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '1h',
              397,
              'sell',
              pd.Timestamp('2022-07-30 13:00:00'),
              'Pattern_Trend',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '1h',
              414,
              'sell',
              pd.Timestamp('2022-07-31 06:00:00'),
              'Pattern_Trend',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '1h',
              415,
              'sell',
              pd.Timestamp('2022-07-31 07:00:00'),
              'Pattern_Trend',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '1h',
              462,
              'sell',
              pd.Timestamp('2022-08-02 06:00:00'),
              'Pattern_Trend',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '1h',
              463,
              'sell',
              pd.Timestamp('2022-08-02 07:00:00'),
              'Pattern_Trend',
              [],
              [],
              [],
              []]]
points4 = [['ETHUSDT',
              '5m',
              629,
              'sell',
              pd.Timestamp('2022-08-23 08:50:00'),
              'STOCH_RSI',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '5m',
              631,
              'sell',
              pd.Timestamp('2022-08-23 09:00:00'),
              'STOCH_RSI',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '5m',
              629,
              'sell',
              pd.Timestamp('2022-08-23 08:50:00'),
              'STOCH_RSI_Trend',
              [],
              [],
              [],
              []],
             ['ETHUSDT',
              '5m',
              631,
              'sell',
              pd.Timestamp('2022-08-23 09:00:00'),
              'STOCH_RSI_Trend',
              [],
              [],
              [],
              []]]
expected_sell = [points1, points2, points3, points4]


@pytest.mark.parametrize('ticker, ttype, timeframe, limit, expected',
                         [
                          ('BTCUSDT', 'sell', '5m', 1000, expected_sell[0]),
                          ('BTCUSDT', 'sell', '5m', 500, expected_sell[1]),
                          ('BTCUSDT', 'sell', '5m', 10, []),
                          ('ETHUSDT', 'sell', '5m', 1000, expected_sell[2]),
                          ('ETHUSDT', 'sell', '5m', 400, expected_sell[3]),
                          ('ETHUSDT', 'sell', '5m', 10, []),
                          ('BTCUSDT', 'buy', '5m', 500, expected_buy[0]),
                          ], ids=repr)
def test_find_signal(mocker, ttype, ticker, timeframe, limit, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs = create_test_data()
    fs_buy = FindSignal('buy', configs)
    fs_sell = FindSignal('sell', configs)
    fs_buy.patterns = [['STOCH', 'RSI'], ['STOCH', 'RSI', 'Trend'], ['Pattern', 'Trend']]
    fs_sell.patterns = [['STOCH', 'RSI'], ['STOCH', 'RSI', 'Trend'], ['Pattern', 'Trend']]
    if ttype == 'buy':
        assert fs_buy.find_signal(dfs, ticker, timeframe, limit, limit) == expected
    else:
        assert fs_sell.find_signal(dfs, ticker, timeframe, limit, limit) == expected
