import pytest
import pandas as pd
from os import environ
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

dfs = {'stat': {'buy': pd.DataFrame(columns=['time', 'ticker', 'timeframe']),
                'sell': pd.DataFrame(columns=['time', 'ticker', 'timeframe'])},
       'BTCUSDT': {'1h': {'data': df_btc_1h}, '5m': {'data': df_btc_5m}},
       'ETHUSDT': {'1h': {'data': df_eth_1h}, '5m': {'data': df_eth_5m}}
       }

# Get configs
configs = ConfigFactory.factory(environ).configs
# Get dict of exchange APIs
exchange_apis = dict()
exchange_api = DataFactory.factory('Binance', **configs)

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
                                                  timeframe, configs)


@pytest.mark.parametrize('dfs, ticker, timeframe, index, high_bound, expected',
                         [
                          (dfs, 'BTCUSDT', '5m', 2, 60, False),
                          (dfs, 'BTCUSDT', '5m', 16, 60, True),
                          (dfs, 'BTCUSDT', '5m', 20, 60, False),
                          (dfs, 'ETHUSDT', '5m', 2, 25, False),
                          (dfs, 'ETHUSDT', '5m', 367, 20, True),
                          (dfs, 'ETHUSDT', '5m', 368, 25, False),
                          (dfs, 'ETHUSDT', '5m', 558, 79.5, True),
                          (dfs, 'ETHUSDT', '5m', 558, 80, False),
                          ], ids=repr)
def test_higher_bound(dfs, timeframe, ticker, index, high_bound, expected):
    rsi_sig = SignalFactory().factory('RSI', configs)
    assert rsi_sig.higher_bound(dfs[ticker][timeframe]['data']['rsi'], index, high_bound) == expected


@pytest.mark.parametrize('dfs, ticker, timeframe, index, low_bound, expected',
                         [
                          (dfs, 'BTCUSDT', '5m', 2, 60, False),
                          (dfs, 'BTCUSDT', '5m', 17, 80, False),
                          (dfs, 'BTCUSDT', '5m', 88, 80, True),
                          (dfs, 'BTCUSDT', '5m', 126, 20, False),
                          (dfs, 'ETHUSDT', '5m', 2, 25, False),
                          (dfs, 'ETHUSDT', '5m', 329, 85, False),
                          (dfs, 'ETHUSDT', '5m', 331, 85, True),
                          (dfs, 'ETHUSDT', '5m', 352, 20, False),
                          (dfs, 'ETHUSDT', '5m', 352, 12, True),
                          ], ids=repr)
def test_lower_bound(dfs, timeframe, ticker, index, low_bound, expected):
    stoch_sig = SignalFactory().factory('STOCH', configs)
    assert stoch_sig.higher_bound(dfs[ticker][timeframe]['data']['stoch_slowk'], index, low_bound) == expected
    assert stoch_sig.higher_bound(dfs[ticker][timeframe]['data']['stoch_slowd'], index, low_bound) == expected


@pytest.mark.parametrize('dfs, ticker, timeframe, index, expected',
                         [
                          (dfs, 'BTCUSDT', '5m', 2, (False, False)),
                          (dfs, 'BTCUSDT', '5m', 48, (False, True)),
                          (dfs, 'BTCUSDT', '5m', 55, (True, False)),
                          (dfs, 'BTCUSDT', '5m', 65, (True, True)),
                          (dfs, 'ETHUSDT', '5m', 2, (False, False)),
                          (dfs, 'ETHUSDT', '5m', 58, (False, True)),
                          (dfs, 'ETHUSDT', '5m', 67, (True, False)),
                          (dfs, 'ETHUSDT', '5m', 75, (True, True)),
                          ], ids=repr)
def test_up_direction(dfs, timeframe, ticker, index, expected):
    stoch_sig = SignalFactory().factory('STOCH', configs)
    assert stoch_sig.up_direction(dfs[ticker][timeframe]['data']['stoch_slowk_dir'], index) == expected[0]
    assert stoch_sig.up_direction(dfs[ticker][timeframe]['data']['stoch_slowd_dir'], index) == expected[1]


@pytest.mark.parametrize('dfs, ticker, timeframe, index, expected',
                         [
                          (dfs, 'BTCUSDT', '5m', 2, (False, False)),
                          (dfs, 'BTCUSDT', '5m', 161, (False, True)),
                          (dfs, 'BTCUSDT', '5m', 140, (True, False)),
                          (dfs, 'BTCUSDT', '5m', 150, (True, True)),
                          (dfs, 'ETHUSDT', '5m', 2, (False, False)),
                          (dfs, 'ETHUSDT', '5m', 52, (False, False)),
                          (dfs, 'ETHUSDT', '5m', 91, (False, True)),
                          (dfs, 'ETHUSDT', '5m', 103, (True, False)),
                          (dfs, 'ETHUSDT', '5m', 107, (True, True)),
                          ], ids=repr)
def test_down_direction(dfs, timeframe, ticker, index, expected):
    stoch_sig = SignalFactory().factory('STOCH', configs)
    assert stoch_sig.down_direction(dfs[ticker][timeframe]['data']['stoch_slowk_dir'], index) == expected[0]
    assert stoch_sig.down_direction(dfs[ticker][timeframe]['data']['stoch_slowd_dir'], index) == expected[1]


@pytest.mark.parametrize('dfs, ticker, timeframe, index, up, expected',
                         [
                          (dfs, 'BTCUSDT', '5m', 2, True, False),
                          (dfs, 'BTCUSDT', '5m', 28, True, True),
                          (dfs, 'BTCUSDT', '5m', 33, True, False),
                          (dfs, 'BTCUSDT', '5m', 34, True, False),
                          (dfs, 'BTCUSDT', '5m', 34, False, True),
                          (dfs, 'BTCUSDT', '5m', 57, False, True),
                          (dfs, 'ETHUSDT', '5m', 41, True, True),
                          (dfs, 'ETHUSDT', '5m', 42, True, True),
                          (dfs, 'ETHUSDT', '5m', 43, True, False),
                          (dfs, 'ETHUSDT', '5m', 43, False, False),
                          (dfs, 'ETHUSDT', '5m', 52, True, False),
                          (dfs, 'ETHUSDT', '5m', 52, False, False),
                          (dfs, 'ETHUSDT', '5m', 62, True, False),
                          (dfs, 'ETHUSDT', '5m', 62, False, False),
                          (dfs, 'ETHUSDT', '5m', 66, False, False),
                          (dfs, 'ETHUSDT', '5m', 67, False, True),
                          (dfs, 'ETHUSDT', '5m', 68, False, True),
                          ], ids=repr)
def test_crossed_lines(dfs, timeframe, ticker, index, up, expected):
    stoch_sig = SignalFactory().factory('STOCH', configs)
    assert stoch_sig.crossed_lines(dfs[ticker][timeframe]['data']['diff'], index, up) == expected


@pytest.mark.parametrize('dfs, ticker, timeframe, index, expected',
                         [
                          (dfs, 'BTCUSDT', '5m', 2, (False, '', ())),
                          (dfs, 'BTCUSDT', '5m', 242, (False, '', ())),
                          (dfs, 'ETHUSDT', '5m', 2, (False, '', ())),
                          (dfs, 'ETHUSDT', '5m', 146, (True, 'buy', (15, 85))),
                          ], ids=repr)
def test_find_stoch_signal(dfs, timeframe, ticker, index, expected):
    stoch_sig = SignalFactory().factory('STOCH', configs)
    if index == 146:
        dfs[ticker][timeframe]['data'].loc[145, 'stoch_slowd'] -= 1
        dfs[ticker][timeframe]['data'].loc[145, 'stoch_slowk_dir'] *= -1
        dfs[ticker][timeframe]['data'].loc[146, 'stoch_slowk_dir'] *= -1
        dfs[ticker][timeframe]['data'].loc[146, 'stoch_slowd_dir'] *= -1
        dfs[ticker][timeframe]['data'].loc[146, 'diff'] *= -1
    assert stoch_sig.find_signal(dfs[ticker][timeframe]['data'], index) == expected

# ETH
# [(74, 'sell', Timestamp('2020-02-13 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (242, 'sell', Timestamp('2020-07-30 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (247, 'sell', Timestamp('2020-08-04 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (248, 'sell', Timestamp('2020-08-05 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (401, 'sell', Timestamp('2021-01-05 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (407, 'sell', Timestamp('2021-01-11 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (521, 'sell', Timestamp('2021-05-05 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (528, 'sell', Timestamp('2021-05-12 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (928, 'buy', Timestamp('2022-06-16 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (929, 'buy', Timestamp('2022-06-17 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))])]
# BTC
# [(242, 'sell', Timestamp('2020-07-30 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (354, 'sell', Timestamp('2020-11-19 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (358, 'sell', Timestamp('2020-11-23 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (401, 'sell', Timestamp('2021-01-05 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (407, 'sell', Timestamp('2021-01-11 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (690, 'sell', Timestamp('2021-10-21 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))]), (931, 'buy', Timestamp('2022-06-19 03:00:00'), [('STOCH', (15, 85)), ('RSI', (25, 75))])]