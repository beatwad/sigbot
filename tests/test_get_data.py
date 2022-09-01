import datetime
import pytest
import pandas as pd
from os import environ
from freezegun import freeze_time
from data.get_data import DataFactory
from config.config import ConfigFactory
from models.cryptocurrency import CryptoCurrency
from indicators.indicators import IndicatorFactory


# Set environment variable
environ["ENV"] = "test"

configs = ConfigFactory.factory(environ).configs

dfs = {'stat': {'buy': pd.DataFrame(columns=['time', 'ticker', 'timeframe']),
                'sell': pd.DataFrame(columns=['time', 'ticker', 'timeframe'])}}

df_btc_5m = pd.read_pickle('test_BTCUSDT_5m.pkl')
df_btc_1h = pd.read_pickle('test_BTCUSDT_1h.pkl')
df_eth_5m = pd.read_pickle('test_ETHUSDT_5m.pkl')
df_eth_1h = pd.read_pickle('test_ETHUSDT_1h.pkl')

# Higher timeframe from which we take levels
work_timeframe = configs['Timeframes']['work_timeframe']


# test get_limit method

@pytest.mark.parametrize('exchange, df, timeframe, expected',
                         [
                          ('Binance', pd.DataFrame(columns=df_btc_5m.columns), '5m',
                           configs['Data']['Binance']['params']['limit']),
                          ('Binance', df_btc_5m, '5m', 3),
                          ('Binance', df_btc_5m, '1h', 2),
                          ('Binance', df_btc_5m, '4h', 7),
                          ('Binance', df_btc_5m, '1d', 5),
                          ], ids=repr)
def test_get_limit(exchange, df, timeframe, expected):
    gd = DataFactory.factory(exchange, **configs)
    gd.timestamp_dict['5m'] = datetime.datetime(2022, 8, 24, 15, 0, 0, 0)
    gd.timestamp_dict['1h'] = datetime.datetime(2022, 8, 24, 14, 0, 0, 0)
    gd.timestamp_dict['4h'] = datetime.datetime(2022, 8, 23, 14, 0, 0, 0)
    gd.timestamp_dict['1d'] = datetime.datetime(2022, 8, 20, 10, 0, 0, 0)

    @freeze_time("2022-08-24-15:11:00")
    def get_limit():
        return gd.get_limit(df, timeframe)

    limit = get_limit()
    assert limit == expected


# test process_data method

cryptocurrencies = [CryptoCurrency('BTC', '5m', 2), CryptoCurrency('BTC', '5m', 2),
                    CryptoCurrency('BTC', '5m', 100), CryptoCurrency('BTC', '5m', 200),
                    CryptoCurrency('ETH', '5m', 200)]

cryptocurrencies[0].time = df_btc_5m['time'].loc[498:500].tolist()
cryptocurrencies[0].open_values = df_btc_5m['open'].loc[498:500].tolist()
cryptocurrencies[0].high_values = df_btc_5m['high'].loc[498:500].tolist()
cryptocurrencies[0].low_values = df_btc_5m['low'].loc[498:500].tolist()
cryptocurrencies[0].close_values = df_btc_5m['close'].loc[498:500].tolist()
cryptocurrencies[0].volume_values = df_btc_5m['volume'].loc[498:500].tolist()

cryptocurrencies[1].time = df_btc_5m['time'].loc[500:502].tolist()
cryptocurrencies[1].open_values = df_btc_5m['open'].loc[500:502].tolist()
cryptocurrencies[1].high_values = df_btc_5m['high'].loc[500:502].tolist()
cryptocurrencies[1].low_values = df_btc_5m['low'].loc[500:502].tolist()
cryptocurrencies[1].close_values = df_btc_5m['close'].loc[500:502].tolist()
cryptocurrencies[1].volume_values = df_btc_5m['volume'].loc[500:502].tolist()

cryptocurrencies[2].time = df_btc_5m['time'].loc[500:600].tolist()
cryptocurrencies[2].open_values = df_btc_5m['open'].loc[500:600].tolist()
cryptocurrencies[2].high_values = df_btc_5m['high'].loc[500:600].tolist()
cryptocurrencies[2].low_values = df_btc_5m['low'].loc[500:600].tolist()
cryptocurrencies[2].close_values = df_btc_5m['close'].loc[500:600].tolist()
cryptocurrencies[2].volume_values = df_btc_5m['volume'].loc[500:600].tolist()

cryptocurrencies[3].time = df_btc_5m['time'].loc[400:600].tolist()
cryptocurrencies[3].open_values = df_btc_5m['open'].loc[400:600].tolist()
cryptocurrencies[3].high_values = df_btc_5m['high'].loc[400:600].tolist()
cryptocurrencies[3].low_values = df_btc_5m['low'].loc[400:600].tolist()
cryptocurrencies[3].close_values = df_btc_5m['close'].loc[400:600].tolist()
cryptocurrencies[3].volume_values = df_btc_5m['volume'].loc[400:600].tolist()

cryptocurrencies[4].time = df_eth_5m['time'].loc[400:600].tolist()
cryptocurrencies[4].open_values = df_eth_5m['open'].loc[400:600].tolist()
cryptocurrencies[4].high_values = df_eth_5m['high'].loc[400:600].tolist()
cryptocurrencies[4].low_values = df_eth_5m['low'].loc[400:600].tolist()
cryptocurrencies[4].close_values = df_eth_5m['close'].loc[400:600].tolist()
cryptocurrencies[4].volume_values = df_eth_5m['volume'].loc[400:600].tolist()

df = df_btc_5m[['time', 'open', 'high', 'low', 'close', 'volume']]
df['time'] = df['time'] + pd.to_timedelta(3, unit='h')


@pytest.mark.parametrize('df, cryptocurrency, expected',
                         [
                          (pd.DataFrame(), cryptocurrencies[0],
                           df.loc[498:500].reset_index(drop=True)),
                          (df.loc[:500], cryptocurrencies[0], df.loc[:500]),
                          (df.loc[:500], cryptocurrencies[1], df.loc[2:502].reset_index(drop=True)),
                          (df.loc[:500], cryptocurrencies[2], df.loc[100:600].reset_index(drop=True)),
                          (df.loc[:500], cryptocurrencies[3], df.loc[100:600].reset_index(drop=True)),
                          ], ids=repr)
def test_process_data(mocker, df, cryptocurrency, expected):
    mocker.patch('api.binance_api.Binance.get_crypto_currency', return_value=cryptocurrency)
    gd = DataFactory.factory('Binance', **configs)
    assert gd.process_data(cryptocurrency, df).equals(expected)


# test get_data method


@pytest.mark.parametrize('df, ticker, timeframe, cryptocurrency, limit, expected',
                         [
                          (pd.DataFrame(), 'BTCUSDT', '5m', cryptocurrencies[0], 1000,
                           (df.loc[498:500].reset_index(drop=True), 1000)),
                          (df.loc[:500], 'BTCUSDT', '5m', cryptocurrencies[0], 0, (df.loc[:500], 0)),
                          (df.loc[:500], 'BTCUSDT', '5m', cryptocurrencies[0], 1, (df.loc[:500], 1)),
                          (df.loc[:500], 'BTCUSDT', '5m', cryptocurrencies[1], 2,
                           (df.loc[2:502].reset_index(drop=True), 2)),
                          (df.loc[:500], 'BTCUSDT', '5m', cryptocurrencies[2], 100,
                           (df.loc[100:600].reset_index(drop=True), 100)),
                          (df.loc[:500], 'BTCUSDT', '5m', cryptocurrencies[3], 200,
                           (df.loc[100:600].reset_index(drop=True), 200)),
                          ], ids=repr)
def test_get_data_get_data(mocker, df, ticker, timeframe, cryptocurrency, limit, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    mocker.patch('data.get_data.GetBinanceData.get_limit', return_value=limit)
    mocker.patch('api.binance_api.Binance.get_crypto_currency', return_value=cryptocurrency)
    gd = DataFactory.factory('Binance', **configs)
    res = gd.get_data(df, ticker, timeframe)
    assert res[0].equals(expected[0])
    assert res[1] == expected[1]


# test_add_indicator_data
df_btc_5 = df_btc_5m[['time', 'open', 'high', 'low', 'close', 'volume']]
df_btc_5['time'] = df_btc_5['time'] + pd.to_timedelta(3, unit='h')
df_btc_1 = df_btc_1h[['time', 'open', 'high', 'low', 'close', 'volume']]
df_btc_1['time'] = df_btc_1['time'] + pd.to_timedelta(3, unit='h')
df_eth_5 = df_eth_5m[['time', 'open', 'high', 'low', 'close', 'volume']]
df_eth_5['time'] = df_eth_5['time'] + pd.to_timedelta(3, unit='h')
df_eth_1 = df_eth_1h[['time', 'open', 'high', 'low', 'close', 'volume']]
df_eth_1['time'] = df_eth_1['time'] + pd.to_timedelta(3, unit='h')

dfss = {'BTCUSDT': {'5m': {'data': df_btc_5}, '1h': {'data': df_btc_1}},
        'ETHUSDT': {'5m': {'data': df_eth_5}, '1h': {'data': df_eth_1}}}

df_ind1 = pd.read_pickle('test_ETHUSDT_1h_indicators.pkl')
df_ind2 = pd.read_pickle('test_ETHUSDT_5m_indicators.pkl')
df_ind3 = pd.read_pickle('test_BTCUSDT_1h_indicators.pkl')
df_ind4 = pd.read_pickle('test_BTCUSDT_5m_indicators.pkl')

level1 = [[1356.17, 1], [1400.46, 2], [1420.83, 2], [1444.91, 2], [1477.4, 2], [1514.89, 2], [1534.94, 2],
          [1564.81, 2], [1597.74, 2], [1623.29, 2], [1661.04, 2], [1692.42, 2], [1722.63, 2], [1744.0, 2],
          [1765.99, 2]]
level2 = [[1530.51, 3], [1549.25, 2], [1557.31, 2], [1564.81, 3], [1571.28, 2], [1578.6, 2], [1589.2, 2],
          [1597.74, 3], [1608.9, 2], [1616.39, 2], [1623.29, 3], [1630.24, 2], [1638.87, 2], [1646.52, 2]]
level3 = [[20805.0, 2], [21063.11, 2], [21269.13, 2], [21441.69, 2], [21748.05, 2], [22028.14, 2], [22257.15, 1],
          [22447.44, 2], [22725.5, 2], [22973.08, 2], [23207.29, 2], [23447.62, 2], [23647.68, 2], [23892.43, 2],
          [24222.0, 2], [24442.66, 1]]
level4 = [[20901.22, 2], [21011.22, 2], [21063.11, 3], [21153.7, 2], [21207.2, 2], [21269.13, 3], [21337.38, 2],
          [21407.59, 3], [21475.55, 3], [21548.71, 2], [21671.84, 2], [21800.0, 1], [21748.05, 2]]
level5 = [[1530.51, 2], [1549.25, 2], [1557.31, 2], [1564.81, 2], [1571.28, 2], [1578.6, 2], [1589.2, 2],
          [1597.74, 2], [1608.9, 2], [1616.39, 2], [1623.29, 2], [1630.24, 2], [1638.87, 2], [1646.52, 2]]

df_inds = [df_ind1, df_ind2, df_ind3, df_ind4]
levels = [level1, level2, level3, level4, level5]


@pytest.mark.parametrize('dfs, df, ticker, timeframe, cryptocurrencies, expected',
                         [
                          (dfss, df_eth_5, 'ETHUSDT', '5m', cryptocurrencies[4], (df_inds[1], levels[4])),
                          (dfss, df_eth_1, 'ETHUSDT', '1h', cryptocurrencies[4], (df_inds[0], levels[0])),
                          (dfss, df_eth_5, 'ETHUSDT', '5m', cryptocurrencies[4], (df_inds[1], levels[1])),
                          (dfss, df_btc_1, 'BTCUSDT', '1h', cryptocurrencies[3], (df_inds[2], levels[2])),
                          (dfss, df_btc_5, 'BTCUSDT', '5m', cryptocurrencies[3], (df_inds[3], levels[3])),
                          ], ids=repr)
def test_add_indicator_data(mocker, dfs, df, ticker, timeframe, cryptocurrencies, expected):
    indicators = list()
    if timeframe == work_timeframe:
        indicator_list = configs['Indicator_list']
    else:
        indicator_list = ['SUP_RES']

    for indicator in indicator_list:
        ind_factory = IndicatorFactory.factory(indicator, configs)
        if ind_factory:
            indicators.append(ind_factory)

    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    mocker.patch('data.get_data.GetBinanceData.get_limit', return_value=200)
    mocker.patch('api.binance_api.Binance.get_crypto_currency', return_value=cryptocurrencies)
    gd = DataFactory.factory('Binance', **configs)
    data = gd.get_data(df.loc[:500], ticker, timeframe)[0]
    dfs[ticker][timeframe]['data'] = gd.add_indicator_data(dfs, data, indicators, ticker, timeframe, configs)[1]
    assert dfs[ticker][timeframe]['data'].equals(expected[0])
    assert dfs[ticker][timeframe]['levels'] == expected[1]
