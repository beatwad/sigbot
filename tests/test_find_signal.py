import pytest
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


@pytest.mark.parametrize('ticker, timeframe, index, high_bound, expected',
                         [
                          ('BTCUSDT', '5m', 2, 60, False),
                          ('BTCUSDT', '5m', 16, 60, True),
                          ('BTCUSDT', '5m', 20, 60, False),
                          ('ETHUSDT', '5m', 2, 25, False),
                          ('ETHUSDT', '5m', 367, 20, True),
                          ('ETHUSDT', '5m', 368, 25, False),
                          ('ETHUSDT', '5m', 558, 79.5, True),
                          ('ETHUSDT', '5m', 558, 80, False),
                          ], ids=repr)
def test_higher_bound(mocker, timeframe, ticker, index, high_bound, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    rsi_sig = SignalFactory().factory('RSI', configs)
    assert rsi_sig.higher_bound(dfs[ticker][timeframe]['data']['rsi'], index, high_bound) == expected


@pytest.mark.parametrize('ticker, timeframe, index, low_bound, expected',
                         [
                          ('BTCUSDT', '5m', 2, 60, False),
                          ('BTCUSDT', '5m', 17, 80, False),
                          ('BTCUSDT', '5m', 88, 80, True),
                          ('BTCUSDT', '5m', 126, 20, False),
                          ('ETHUSDT', '5m', 2, 25, False),
                          ('ETHUSDT', '5m', 329, 85, False),
                          ('ETHUSDT', '5m', 331, 85, True),
                          ('ETHUSDT', '5m', 352, 20, False),
                          ('ETHUSDT', '5m', 352, 12, True),
                          ], ids=repr)
def test_lower_bound(mocker, timeframe, ticker, index, low_bound, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()

    stoch_sig = SignalFactory().factory('STOCH', configs)
    assert stoch_sig.higher_bound(dfs[ticker][timeframe]['data']['stoch_slowk'], index, low_bound) == expected
    assert stoch_sig.higher_bound(dfs[ticker][timeframe]['data']['stoch_slowd'], index, low_bound) == expected


@pytest.mark.parametrize('ticker, timeframe, index, expected',
                         [
                          ('BTCUSDT', '5m', 2, (False, False)),
                          ('BTCUSDT', '5m', 48, (False, True)),
                          ('BTCUSDT', '5m', 55, (True, False)),
                          ('BTCUSDT', '5m', 65, (True, True)),
                          ('ETHUSDT', '5m', 2, (False, False)),
                          ('ETHUSDT', '5m', 58, (False, True)),
                          ('ETHUSDT', '5m', 67, (True, False)),
                          ('ETHUSDT', '5m', 75, (True, True)),
                          ], ids=repr)
def test_up_direction(mocker, timeframe, ticker, index, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', configs)
    assert stoch_sig.up_direction(dfs[ticker][timeframe]['data']['stoch_slowk_dir'], index) == expected[0]
    assert stoch_sig.up_direction(dfs[ticker][timeframe]['data']['stoch_slowd_dir'], index) == expected[1]


@pytest.mark.parametrize('ticker, timeframe, index, expected',
                         [
                          ('BTCUSDT', '5m', 2, (False, False)),
                          ('BTCUSDT', '5m', 161, (False, True)),
                          ('BTCUSDT', '5m', 140, (True, False)),
                          ('BTCUSDT', '5m', 150, (True, True)),
                          ('ETHUSDT', '5m', 2, (False, False)),
                          ('ETHUSDT', '5m', 52, (False, False)),
                          ('ETHUSDT', '5m', 91, (False, True)),
                          ('ETHUSDT', '5m', 103, (True, False)),
                          ('ETHUSDT', '5m', 107, (True, True)),
                          ], ids=repr)
def test_down_direction(mocker, timeframe, ticker, index, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', configs)
    assert stoch_sig.down_direction(dfs[ticker][timeframe]['data']['stoch_slowk_dir'], index) == expected[0]
    assert stoch_sig.down_direction(dfs[ticker][timeframe]['data']['stoch_slowd_dir'], index) == expected[1]


@pytest.mark.parametrize('ticker, timeframe, index, up, expected',
                         [
                          ('BTCUSDT', '5m', 2, True, False),
                          ('BTCUSDT', '5m', 28, True, True),
                          ('BTCUSDT', '5m', 33, True, False),
                          ('BTCUSDT', '5m', 34, True, False),
                          ('BTCUSDT', '5m', 34, False, True),
                          ('BTCUSDT', '5m', 57, False, True),
                          ('ETHUSDT', '5m', 41, True, True),
                          ('ETHUSDT', '5m', 42, True, True),
                          ('ETHUSDT', '5m', 43, True, False),
                          ('ETHUSDT', '5m', 43, False, False),
                          ('ETHUSDT', '5m', 52, True, False),
                          ('ETHUSDT', '5m', 52, False, False),
                          ('ETHUSDT', '5m', 62, True, False),
                          ('ETHUSDT', '5m', 62, False, False),
                          ('ETHUSDT', '5m', 66, False, False),
                          ('ETHUSDT', '5m', 67, False, True),
                          ('ETHUSDT', '5m', 68, False, True),
                          ], ids=repr)
def test_crossed_lines(mocker, timeframe, ticker, index, up, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', configs)
    assert stoch_sig.crossed_lines(dfs[ticker][timeframe]['data']['stoch_diff'], index, up) == expected


@pytest.mark.parametrize('ticker, timeframe, index, expected',
                         [
                          ('BTCUSDT', '5m', 2, (False, '', ())),
                          ('BTCUSDT', '5m', 242, (False, '', ())),
                          ('BTCUSDT', '5m', 447, (True, 'sell', (15, 85))),
                          ('ETHUSDT', '5m', 2, (False, '', ())),
                          ('ETHUSDT', '5m', 146, (True, 'buy', (15, 85))),
                          ('ETHUSDT', '5m', 203, (False, '', ())),
                          ], ids=repr)
def test_find_stoch_signal(mocker, timeframe, ticker, index, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    stoch_sig = SignalFactory().factory('STOCH', configs)
    if ticker == 'BTCUSDT' and index == 447:
        dfs[ticker][timeframe]['data'].loc[index, 'stoch_diff'] *= -1
        dfs[ticker][timeframe]['data'].loc[index-1, 'stoch_slowk'] += 3
        dfs[ticker][timeframe]['data'].loc[index-1, 'stoch_slowd'] += 3
        dfs[ticker][timeframe]['data'].loc[index, 'stoch_slowk_dir'] *= -1
        dfs[ticker][timeframe]['data'].loc[index, 'stoch_slowd_dir'] *= -1
    elif ticker == 'ETHUSDT' and index == 146:
        dfs[ticker][timeframe]['data'].loc[index-1, 'stoch_slowd'] -= 1
        dfs[ticker][timeframe]['data'].loc[index-1, 'stoch_slowk_dir'] *= -1
        dfs[ticker][timeframe]['data'].loc[index, 'stoch_slowk_dir'] *= -1
        dfs[ticker][timeframe]['data'].loc[index, 'stoch_slowd_dir'] *= -1
        dfs[ticker][timeframe]['data'].loc[index, 'stoch_diff'] *= -1
    assert stoch_sig.find_signal(dfs[ticker][timeframe]['data'], index) == expected


@pytest.mark.parametrize('ticker, timeframe, index, expected',
                         [
                          ('BTCUSDT', '5m', 2, (False, '', ())),
                          ('BTCUSDT', '5m', 242, (True, 'buy', 1)),
                          ('BTCUSDT', '5m', 459, (True, 'sell', 1)),
                          ('BTCUSDT', '5m', 25, (True, 'buy', 1)),
                          ('ETHUSDT', '5m', 2, (True, 'sell', 1)),
                          ('ETHUSDT', '5m', 3, (True, 'buy', 1)),
                          ('ETHUSDT', '5m', 82, (True, 'sell', 3)),
                          ('ETHUSDT', '5m', 128, (True, 'sell', 2)),
                          ('ETHUSDT', '5m', 143, (True, 'buy', 2)),
                          ('ETHUSDT', '5m', 146, (False, '', ())),
                          ('ETHUSDT', '5m', 203, (False, '', ())),
                          ], ids=repr)
def test_find_price_change_signal(mocker, timeframe, ticker, index, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    price_change_sig = SignalFactory().factory('PriceChange', configs)
    assert price_change_sig.find_signal(dfs[ticker][timeframe]['data'], index) == expected


@pytest.mark.parametrize('ticker, timeframe, index, expected',
                         [
                             ('BTCUSDT', '5m', 2, (False, '', ())),
                             ('BTCUSDT', '5m', 242, (True, 'buy', 1)),
                             ('BTCUSDT', '5m', 459, (True, 'sell', 1)),
                             ('BTCUSDT', '5m', 25, (True, 'buy', 1)),
                             ('ETHUSDT', '5m', 2, (True, 'sell', 1)),
                             ('ETHUSDT', '5m', 3, (True, 'buy', 1)),
                             ('ETHUSDT', '5m', 82, (True, 'sell', 3)),
                             ('ETHUSDT', '5m', 128, (True, 'sell', 2)),
                             ('ETHUSDT', '5m', 143, (True, 'buy', 2)),
                             ('ETHUSDT', '5m', 146, (False, '', ())),
                             ('ETHUSDT', '5m', 203, (False, '', ())),
                         ], ids=repr)
def test_find_price_change_signal_vect_1(mocker, timeframe, ticker, index, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    price_change_sig = SignalFactory().factory('PriceChange', configs)
    assert price_change_sig.find_signal_vect_1(dfs[ticker][timeframe]['data']) == expected


@pytest.mark.parametrize('ticker, timeframe, index, expected',
                         [
                             ('BTCUSDT', '5m', 2, (False, '', ())),
                             ('BTCUSDT', '5m', 242, (True, 'buy', 1)),
                             ('BTCUSDT', '5m', 459, (True, 'sell', 1)),
                             ('BTCUSDT', '5m', 25, (True, 'buy', 1)),
                             ('ETHUSDT', '5m', 2, (True, 'sell', 1)),
                             ('ETHUSDT', '5m', 3, (True, 'buy', 1)),
                             ('ETHUSDT', '5m', 82, (True, 'sell', 3)),
                             ('ETHUSDT', '5m', 128, (True, 'sell', 2)),
                             ('ETHUSDT', '5m', 143, (True, 'buy', 2)),
                             ('ETHUSDT', '5m', 146, (False, '', ())),
                             ('ETHUSDT', '5m', 203, (False, '', ())),
                         ], ids=repr)
def test_find_price_change_signal_vect_2(mocker, timeframe, ticker, index, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    price_change_sig = SignalFactory().factory('PriceChange', configs)
    assert price_change_sig.find_signal_vect_2(dfs[ticker][timeframe]['data']) == expected


@pytest.mark.parametrize('ticker, timeframe, index, expected',
                         [
                             ('BTCUSDT', '5m', 2, (False, '', ())),
                             ('BTCUSDT', '5m', 242, (True, 'buy', 1)),
                             ('BTCUSDT', '5m', 459, (True, 'sell', 1)),
                             ('BTCUSDT', '5m', 25, (True, 'buy', 1)),
                             ('ETHUSDT', '5m', 2, (True, 'sell', 1)),
                             ('ETHUSDT', '5m', 3, (True, 'buy', 1)),
                             ('ETHUSDT', '5m', 82, (True, 'sell', 3)),
                             ('ETHUSDT', '5m', 128, (True, 'sell', 2)),
                             ('ETHUSDT', '5m', 143, (True, 'buy', 2)),
                             ('ETHUSDT', '5m', 146, (False, '', ())),
                             ('ETHUSDT', '5m', 203, (False, '', ())),
                         ], ids=repr)
def test_find_price_change_signal_vect_3(mocker, timeframe, ticker, index, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, df = create_test_data()
    price_change_sig = SignalFactory().factory('PriceChange', configs)
    assert price_change_sig.find_signal_vect_3(dfs[ticker][timeframe]['data']) == expected


# @pytest.mark.parametrize('ticker, timeframe, index, buy, expected',
#                          [
#                           ('BTCUSDT', '5m', 71, True, False),
#                           ('BTCUSDT', '5m', 242, True, True),
#                           ('BTCUSDT', '5m', 247, True, False),
#                           ('BTCUSDT', '5m', 447, False, True),
#                           ('BTCUSDT', '5m', 462, False, False),
#                           ('ETHUSDT', '5m', 30, True, True),
#                           ('ETHUSDT', '5m', 30, False, True),
#                           ('ETHUSDT', '5m', 68, True, False),
#                           ('ETHUSDT', '5m', 133, False, False),
#                           ], ids=repr)
# def test_check_levels(mocker, timeframe, ticker, index, buy, expected):
#     mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
#     dfs, _ = create_test_data()
#     sup_res_sig = SignalFactory().factory('SUP_RES', configs)
#     df = dfs[ticker][timeframe]['data']
#     levels = dfs[ticker][timeframe]['levels']
#     level_proximity = np.mean(df['high'] - df['low']) * sup_res_sig.proximity_multiplier
#     assert sup_res_sig.check_levels(df, index, levels, level_proximity, buy) == expected
#
#
# @pytest.mark.parametrize('ticker, timeframe, index, buy, expected',
#                          [
#                           ('BTCUSDT', '5m', 71, True, False),
#                           ('BTCUSDT', '5m', 242, True, True),
#                           ('BTCUSDT', '5m', 247, True, False),
#                           ('BTCUSDT', '5m', 253, True, False),
#                           ('BTCUSDT', '5m', 447, False, True),
#                           ('BTCUSDT', '5m', 448, False, False),
#                           ('ETHUSDT', '5m', 30, True, False),
#                           ('ETHUSDT', '5m', 30, False, False),
#                           ('ETHUSDT', '5m', 68, True, False),
#                           ('ETHUSDT', '5m', 133, False, False),
#                           ], ids=repr)
# def test_check_levels_robust(mocker, timeframe, ticker, index, buy, expected):
#     mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
#     dfs, _ = create_test_data()
#     sup_res_sig = SignalFactory().factory('SUP_RES_Robust', configs)
#     df = dfs[ticker][timeframe]['data']
#     levels = dfs[ticker][timeframe]['levels']
#     level_proximity = np.mean(df['high'] - df['low']) * sup_res_sig.proximity_multiplier
#     assert sup_res_sig.check_levels(df, index, levels, level_proximity, buy) == expected


points1 = [['BTCUSDT', '5m', 506, 'buy', datetime(2022, 8, 22, 21, 55),
            ['STOCH', 'RSI'], [], [], [], []],
           ['BTCUSDT', '5m', 91, 'sell', datetime(2022, 8, 21, 11, 20),
            ['STOCH', 'RSI'], [], [], [], []],
            ['BTCUSDT', '5m', 569, 'sell', datetime(2022, 8, 23, 3, 10),
            ['STOCH', 'RSI'], [], [], [], []],
            ['BTCUSDT', '5m', 506, 'buy', datetime(2022, 8, 22, 21, 55),
            ['STOCH', 'RSI', 'LinearReg'], [], [], [], []],
            ['BTCUSDT', '5m', 91, 'sell', datetime(2022, 8, 21, 11, 20),
            ['STOCH', 'RSI', 'LinearReg'], [], [], [], []]]

points2 = [['BTCUSDT', '5m', 506, 'buy', datetime(2022, 8, 22, 21, 55),
            ['STOCH', 'RSI'], [], [], [], []],
           ['BTCUSDT', '5m', 569, 'sell', datetime(2022, 8, 23, 3, 10),
            ['STOCH', 'RSI'], [], [], [], []],
           ['BTCUSDT', '5m', 506, 'buy', datetime(2022, 8, 22, 21, 55),
            ['STOCH', 'RSI', 'LinearReg'], [], [], [], []]]
points3 = [['ETHUSDT', '5m', 370, 'buy', datetime(2022, 8, 22, 11, 15),
            ['STOCH', 'RSI'], [], [], [], []],
           ['ETHUSDT', '5m', 629, 'buy', datetime(2022, 8, 23, 8, 50),
            ['STOCH', 'RSI'], [], [], [], []],
           ['ETHUSDT', '5m', 631, 'buy', datetime(2022, 8, 23, 9),
            ['STOCH', 'RSI'], [], [], [], []],
           ['ETHUSDT', '5m', 83, 'sell', datetime(2022, 8, 21, 11, 20),
            ['STOCH', 'RSI'], [], [], [], []],
           ['ETHUSDT', '5m', 83, 'sell', datetime(2022, 8, 21, 11, 20),
            ['STOCH', 'RSI', 'LinearReg'], [], [], [], []]]
points4 = [['ETHUSDT', '5m', 629, 'buy', datetime(2022, 8, 23, 8, 50),
            ['STOCH', 'RSI'], [], [], [], []],
           ['ETHUSDT', '5m', 631, 'buy', datetime(2022, 8, 23, 9),
            ['STOCH', 'RSI'], [], [], [], []]]
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


@pytest.mark.parametrize('ticker, timeframe, limit, expected',
                         [
                          ('BTCUSDT', '5m', 1000, expected[0]),
                          ('BTCUSDT', '5m', 500, expected[1]),
                          ('BTCUSDT', '5m', 10, []),
                          ('ETHUSDT', '5m', 1000, expected[2]),
                          ('ETHUSDT', '5m', 400, expected[3]),
                          ('ETHUSDT', '5m', 10, []),
                          ], ids=repr)
def test_find_signal_vect(mocker, ticker, timeframe, limit, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    dfs, _ = create_test_data()
    fs = FindSignal(configs)
    fs.patterns = [['STOCH', 'RSI'], ['STOCH', 'RSI', 'LinearReg']]
    assert fs.find_signal_vect(dfs, ticker, timeframe, limit) == expected

