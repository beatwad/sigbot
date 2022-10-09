import pytest
import numpy as np
import pandas as pd
from os import environ
from datetime import datetime
from data.get_data import DataFactory
from config.config import ConfigFactory
from signals.find_signal import SignalFactory
from signals.find_signal import FindSignal
from indicators.indicators import IndicatorFactory

environ["ENV"] = "test"
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
           'BTCUSDT': {'1h': {'data': df_btc_1h}, '5m': {'data': df_btc_5m}},
           'ETHUSDT': {'1h': {'data': df_eth_1h}, '5m': {'data': df_eth_5m}}
           }

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
                ind_factory = IndicatorFactory.factory(indicator, configs)
                if ind_factory:
                    indicators.append(ind_factory)
            # Write indicators to dataframe, update dataframe dict
            dfs, df = exchange_api.add_indicator_data(dfs, dfs[ticker][timeframe]['data'], indicators, ticker,
                                                      timeframe, data_qty, configs)

    return dfs, df


btc_expected = [np.array([18, 19, 20, 21, 25, 26, 27, 45, 46, 47, 48])]
eth_expected = [np.array([18, 19])]


@pytest.mark.parametrize('ticker, timeframe, high_bound, expected',
                         [
                          # ('BTCUSDT', '5m', 100, []),
                          ('BTCUSDT', '5m', 60, btc_expected[0]),
                          ('ETHUSDT', '5m', 75, eth_expected[0]),
                          ], ids=repr)
def test_higher_bound(mocker, timeframe, ticker, high_bound, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', configs)

    df = dfs[ticker][timeframe]['data'][:50]
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
    dfs, df = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', configs)

    df = dfs[ticker][timeframe]['data'][:50]
    stoch_slowk = df['stoch_slowk']
    stoch_slowk_lag_1 = df['stoch_slowk'].shift(1)
    stoch_slowk_lag_2 = df['stoch_slowk'].shift(2)

    points = stoch_sig.lower_bound(low_bound, stoch_slowk, stoch_slowk_lag_1, stoch_slowk_lag_2)
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [np.array([24, 25, 26, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47])]
eth_expected = [np.array([27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 47, 48, 49])]


@pytest.mark.parametrize('ticker, timeframe, expected',
                         [
                          ('BTCUSDT', '5m', btc_expected[0]),
                          ('ETHUSDT', '5m', eth_expected[0]),
                          ], ids=repr)
def test_up_direction(mocker, timeframe, ticker, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', configs)

    df = dfs[ticker][timeframe]['data'][:50]

    points = stoch_sig.up_direction(df['stoch_slowk_dir'])
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [np.array([22, 23, 27, 28, 29, 30, 31, 32, 37, 48, 49])]
eth_expected = [np.array([22, 23, 24, 25, 26, 28, 29, 41, 42, 43, 44, 45, 46])]


@pytest.mark.parametrize('ticker, timeframe, expected',
                         [
                          ('BTCUSDT', '5m', btc_expected[0]),
                          ('ETHUSDT', '5m', eth_expected[0]),
                          ], ids=repr)
def test_down_direction(mocker, timeframe, ticker, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', configs)

    df = dfs[ticker][timeframe]['data'][:50]

    points = stoch_sig.down_direction(df['stoch_slowk_dir'])
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [np.array([27, 28, 48, 49, 66, 67, 91, 92]),
                np.array([24, 25, 34, 35, 57, 58, 76, 77])]
eth_expected = [np.array([41, 42, 58, 59, 83, 84, 98]),
                np.array([31, 32, 48, 49, 67, 68, 92, 93, 99])]


@pytest.mark.parametrize('ticker, timeframe, up, expected',
                         [
                          ('BTCUSDT', '5m', True, btc_expected[0]),
                          ('BTCUSDT', '5m', False, btc_expected[1]),
                          ('ETHUSDT', '5m', True, eth_expected[0]),
                          ('ETHUSDT', '5m', False, eth_expected[1])
                          ], ids=repr)
def test_crossed_lines(mocker, timeframe, ticker, up, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', configs)

    df = dfs[ticker][timeframe]['data'][:100]
    stoch_diff = df['stoch_diff']
    stoch_diff_lag_1 = df['stoch_diff'].shift(1)
    stoch_diff_lag_2 = df['stoch_diff'].shift(2)

    points = stoch_sig.crossed_lines(up, stoch_diff, stoch_diff_lag_1, stoch_diff_lag_2)
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [np.array([382, 643, 725, 826, 865]),
                np.array([447, 667])]
eth_expected = [np.array([68, 146, 374, 611, 612, 631, 632]),
                np.array([83, 659, 995, 996])]


@pytest.mark.parametrize('ticker, timeframe, expected',
                         [
                             ('BTCUSDT', '5m', btc_expected),
                             ('ETHUSDT', '5m', eth_expected)
                         ], ids=repr)
def test_find_stoch_signal(mocker, timeframe, ticker, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', configs)
    if ticker == 'BTCUSDT':
        dfs[ticker][timeframe]['data'].loc[447, 'stoch_diff'] *= -1
        dfs[ticker][timeframe]['data'].loc[446, 'stoch_slowk'] += 3
        dfs[ticker][timeframe]['data'].loc[446, 'stoch_slowd'] += 3
        dfs[ticker][timeframe]['data'].loc[447, 'stoch_slowk_dir'] *= -1
        dfs[ticker][timeframe]['data'].loc[447, 'stoch_slowd_dir'] *= -1
    elif ticker == 'ETHUSDT':
        dfs[ticker][timeframe]['data'].loc[145, 'stoch_slowd'] -= 1
        dfs[ticker][timeframe]['data'].loc[145, 'stoch_slowk_dir'] *= -1
        dfs[ticker][timeframe]['data'].loc[146, 'stoch_slowk_dir'] *= -1
        dfs[ticker][timeframe]['data'].loc[146, 'stoch_slowd_dir'] *= -1
        dfs[ticker][timeframe]['data'].loc[146, 'stoch_diff'] *= -1
    df = dfs[ticker][timeframe]['data']
    buy_points, sell_points = stoch_sig.find_signal(df)
    buy_indexes = np.where(buy_points == 1)
    sell_indexes = np.where(sell_points == 1)
    assert np.array_equal(buy_indexes[0], expected[0])
    assert np.array_equal(sell_indexes[0], expected[1])


btc_expected = [np.array([19, 25, 26, 27, 28, 29, 46, 59, 71, 72, 73, 78, 91, 92, 93, 96]),
                np.array([24, 80, 81, 87, 89, 90, 98, 99])]
eth_expected = [np.array([3, 10, 17, 18, 19, 20, 21, 25, 38, 40, 44, 63, 64, 65, 70, 83, 84, 85, 88, 95, 96]),
                np.array([2,  5, 16, 71, 72, 73, 80, 81, 82, 90, 91, 92])]


@pytest.mark.parametrize('ticker, timeframe, expected',
                         [
                             ('BTCUSDT', '5m', btc_expected),
                             ('ETHUSDT', '5m', eth_expected)
                         ], ids=repr)
def test_find_price_change_signal(mocker, ticker, timeframe, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    price_change_sig = SignalFactory().factory('PriceChange', configs)
    buy_points, sell_points = price_change_sig.find_signal(dfs[ticker][timeframe]['data'][:100])
    buy_indexes = np.where(buy_points == 1)
    sell_indexes = np.where(sell_points == 1)
    assert np.array_equal(buy_indexes[0], expected[0])
    assert np.array_equal(sell_indexes[0], expected[1])


points1 = [['BTCUSDT', '5m', 506, 'buy', datetime(2022, 8, 22, 21, 55),
            'STOCH_RSI', [], [], [], []],
           ['BTCUSDT', '5m', 91, 'sell', datetime(2022, 8, 21, 11, 20),
            'STOCH_RSI', [], [], [], []],
            ['BTCUSDT', '5m', 569, 'sell', datetime(2022, 8, 23, 3, 10),
           'STOCH_RSI', [], [], [], []],
            ['BTCUSDT', '5m', 506, 'buy', datetime(2022, 8, 22, 21, 55),
           'STOCH_RSI_LinearReg', [], [], [], []],
            ['BTCUSDT', '5m', 91, 'sell', datetime(2022, 8, 21, 11, 20),
           'STOCH_RSI_LinearReg', [], [], [], []]]

points2 = [['BTCUSDT', '5m', 506, 'buy', datetime(2022, 8, 22, 21, 55),
            'STOCH_RSI', [], [], [], []],
           ['BTCUSDT', '5m', 569, 'sell', datetime(2022, 8, 23, 3, 10),
            'STOCH_RSI', [], [], [], []],
           ['BTCUSDT', '5m', 506, 'buy', datetime(2022, 8, 22, 21, 55),
            'STOCH_RSI_LinearReg', [], [], [], []]]
points3 = [['ETHUSDT', '5m', 370, 'buy', datetime(2022, 8, 22, 11, 15),
            'STOCH_RSI', [], [], [], []],
           ['ETHUSDT', '5m', 629, 'buy', datetime(2022, 8, 23, 8, 50),
            'STOCH_RSI', [], [], [], []],
           ['ETHUSDT', '5m', 631, 'buy', datetime(2022, 8, 23, 9),
            'STOCH_RSI', [], [], [], []],
           ['ETHUSDT', '5m', 83, 'sell', datetime(2022, 8, 21, 11, 20),
            'STOCH_RSI', [], [], [], []],
           ['ETHUSDT', '5m', 83, 'sell', datetime(2022, 8, 21, 11, 20),
            'STOCH_RSI_LinearReg', [], [], [], []]]
points4 = [['ETHUSDT', '5m', 629, 'buy', datetime(2022, 8, 23, 8, 50),
            'STOCH_RSI', [], [], [], []],
           ['ETHUSDT', '5m', 631, 'buy', datetime(2022, 8, 23, 9),
            'STOCH_RSI', [], [], [], []]]
expected = [points1, points2, points3, points4]


@pytest.mark.parametrize('ticker, timeframe, limit, expected',
                         [
                          ('BTCUSDT', '5m', 1000, expected[0]),
                          ('BTCUSDT', '5m', 500, expected[1]),
                          ('BTCUSDT', '5m', 10, []),
                          ('ETHUSDT', '5m', 1000, expected[2]),
                          ('ETHUSDT', '5m', 400, expected[3]),
                          ('ETHUSDT', '5m', 10, []),
                          ], ids=repr)
def test_find_signal(mocker, ticker, timeframe, limit, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, _ = create_test_data()
    fs = FindSignal(configs)
    fs.patterns = [['STOCH', 'RSI'], ['STOCH', 'RSI', 'LinearReg']]
    assert fs.find_signal(dfs, ticker, timeframe, limit) == expected

