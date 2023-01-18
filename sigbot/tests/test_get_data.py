import datetime
import pytest
import pandas as pd
from os import environ
from freezegun import freeze_time
from data.get_data import DataFactory
from config.config import ConfigFactory
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

@pytest.mark.parametrize('exchange, df, ticker, timeframe, expected',
                         [
                          ('Binance', pd.DataFrame(columns=df_btc_5m.columns), 'BTCUSDT', '5m',
                           configs['Data']['Binance']['params']['limit']),
                          ('Binance', df_btc_5m, 'BTCUSDT', '5m', 3),
                          ('Binance', df_btc_5m, 'ETHUSDT', '1h', 2),
                          ('Binance', df_btc_5m, 'ETHUSDT', '4h', 7),
                          ('Binance', df_btc_5m, 'ETHUSDT', '1d', 5),
                          ], ids=repr)
def test_get_limit(mocker, exchange, df, ticker, timeframe, expected):
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)

    tickers = ['BTCUSDT', 'ETHUSDT']
    gd = DataFactory.factory(exchange, **configs)
    gd.fill_ticker_dict(tickers)

    gd.ticker_dict[ticker]['5m'] = datetime.datetime(2022, 8, 24, 15, 0, 0, 0)
    gd.ticker_dict[ticker]['1h'] = datetime.datetime(2022, 8, 24, 14, 0, 0, 0)
    gd.ticker_dict[ticker]['4h'] = datetime.datetime(2022, 8, 23, 14, 0, 0, 0)
    gd.ticker_dict[ticker]['1d'] = datetime.datetime(2022, 8, 20, 10, 0, 0, 0)

    @freeze_time("2022-08-24-15:11:00")
    def get_limit():
        return gd.get_limit(df, ticker, timeframe)

    limit = get_limit()
    assert limit == expected


# test process_data method
cryptocurrencies1 = df_btc_5m[['time', 'open', 'high', 'low', 'close', 'volume']].loc[498:499].reset_index(drop=True)
cryptocurrencies2 = df_btc_5m[['time', 'open', 'high', 'low', 'close', 'volume']].loc[500:501].reset_index(drop=True)
cryptocurrencies3 = df_btc_5m[['time', 'open', 'high', 'low', 'close', 'volume']].loc[500:599].reset_index(drop=True)
cryptocurrencies4 = df_btc_5m[['time', 'open', 'high', 'low', 'close', 'volume']].loc[400:599].reset_index(drop=True)
cryptocurrencies5 = df_eth_5m[['time', 'open', 'high', 'low', 'close', 'volume']].loc[400:599].reset_index(drop=True)
cryptocurrencies6 = df_eth_1h[['time', 'open', 'high', 'low', 'close', 'volume']].loc[400:599].reset_index(drop=True)

cryptocurrencies = [cryptocurrencies1, cryptocurrencies2, cryptocurrencies3, cryptocurrencies4, cryptocurrencies5, cryptocurrencies6]

df = df_btc_5m[['time', 'open', 'high', 'low', 'close', 'volume']]
df['time'] = df['time'] + pd.to_timedelta(3, unit='h')


@pytest.mark.parametrize('df, index, limit, expected',
                         [
                          (pd.DataFrame(), 0, 500,
                           df.loc[498:499].reset_index(drop=True)),
                          (df.loc[:499], 0, 500, df.loc[:499]),
                          (df.loc[:499], 1, 500, df.loc[2:501].reset_index(drop=True)),
                          (df.loc[:499], 1, 1000, df.loc[:501].reset_index(drop=True)),
                          (df.loc[:499], 2, 500, df.loc[100:599].reset_index(drop=True)),
                          (df.loc[:499], 3, 500, df.loc[100:599].reset_index(drop=True)),
                          (df.loc[:499], 3, 1000, df.loc[:599].reset_index(drop=True)),
                          ], ids=repr)
def test_process_data(mocker, df, index, limit, expected):
    tmp = cryptocurrencies[index].copy()
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    mocker.patch('api.binance_api.Binance.get_klines', return_value=tmp)
    gd = DataFactory.factory('Binance', **configs)
    gd.limit = limit
    res = gd.process_data(tmp, df)
    assert res.equals(expected)


# test get_data method


@pytest.mark.parametrize('df, ticker, timeframe, index, limit, expected',
                         [
                          (pd.DataFrame(), 'BTCUSDT', '5m', 0, 1000,
                           (df.loc[498:499].reset_index(drop=True), 1000)),
                          (df.loc[:499], 'BTCUSDT', '5m', 0, 0, (df.loc[:499], 0)),
                          (df.loc[:499], 'BTCUSDT', '5m', 0, 1, (df.loc[:499], 1)),
                          (df.loc[:499], 'BTCUSDT', '5m', 1, 2,
                           (df.loc[2:501].reset_index(drop=True), 2)),
                          (df.loc[:499], 'BTCUSDT', '5m', 2, 100,
                           (df.loc[100:599].reset_index(drop=True), 100)),
                          (df.loc[:499], 'BTCUSDT', '5m', 3, 200,
                           (df.loc[100:599].reset_index(drop=True), 200)),
                          ], ids=repr)
def test_get_data_get_data(mocker, df, ticker, timeframe, index, limit, expected):
    tmp = cryptocurrencies[index].copy()
    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    mocker.patch('data.get_data.GetBinanceData.get_limit', return_value=limit)
    mocker.patch('api.binance_api.Binance.get_klines', return_value=tmp)
    tickers = ['BTCUSDT', 'ETHUSDT']
    gd = DataFactory.factory('Binance', **configs)
    gd.limit = 500
    gd.fill_ticker_dict(tickers)

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

dfss = {'BTCUSDT': {'1h': {'data': {'buy': df_btc_1h, 'sell': df_btc_1h}},
                    '5m': {'data': {'buy': df_btc_5m, 'sell': df_btc_5m}}},
        'ETHUSDT': {'1h': {'data': {'buy': df_eth_1h, 'sell': df_eth_1h}},
                    '5m': {'data': {'buy': df_eth_5m, 'sell': df_eth_5m}}}}

df_ind1 = pd.read_pickle('test_ETHUSDT_1h_indicators.pkl')
df_ind2 = pd.read_pickle('test_ETHUSDT_5m_indicators.pkl')
df_ind3 = pd.read_pickle('test_BTCUSDT_1h_indicators.pkl')
df_ind4 = pd.read_pickle('test_BTCUSDT_5m_indicators.pkl')

level1 = [[1356.17, 1], [1420.83, 2], [1444.91, 2], [1477.4, 2], [1514.89, 2], [1553.79, 2], [1581.09, 2],
          [1615.57, 2], [1639.48, 2], [1674.58, 2], [1711.04, 2], [1757.99, 2], [1783.0, 2], [1817.72, 2],
          [1849.66, 2], [1880.55, 2], [1905.38, 2], [1942.0, 1]]
level2 = [[1530.51, 2], [1551.18, 3], [1563.76, 2], [1572.27, 2], [1584.2, 3], [1591.81, 2], [1601.18, 2],
          [1609.81, 3], [1617.83, 3], [1624.62, 2], [1632.84, 3], [1641.29, 3]]
level3 = [[20805.0, 2], [21129.65, 2], [21552.53, 2], [21808.01, 2], [22028.14, 2], [22449.09, 2],
          [22760.0, 2], [23139.0, 2], [23402.0, 2], [23622.96, 2], [23861.83, 2], [24078.0, 2], [24334.46, 2],
          [24869.5, 2]]
level4 = [[20937.46, 2], [21063.11, 2], [21130.9, 3], [21237.15, 2], [21324.96, 2], [21407.59, 2], [21488.0, 2],
          [21550.0, 3], [21671.84, 2], [21800.0, 3]]

df_inds = [df_ind1, df_ind2, df_ind3, df_ind4]
levels = [level1, level2, level3, level4]


@pytest.mark.parametrize('df, ticker, timeframe, expected',
                         [
                          (df_eth_1, 'ETHUSDT', '1h', (df_inds[0].loc[:499], levels[0])),
                          (df_eth_5, 'ETHUSDT', '5m', (df_inds[1].loc[:499], levels[1])),
                          (df_btc_1, 'BTCUSDT', '1h', (df_inds[2].loc[:499], levels[2])),
                          (df_btc_5, 'BTCUSDT', '5m', (df_inds[3].loc[:499], levels[3]))
                          ], ids=repr)
def test_add_indicator_data(mocker, df, ticker, timeframe, expected):
    dfs = dfss.copy()
    dfs[ticker][timeframe]['data']['buy'] = df.loc[:499]
    indicators = list()
    if timeframe == work_timeframe:
        indicator_list = configs['Indicator_list']
    else:
        indicator_list = ['LinearReg', 'MACD', 'Pattern']

    for indicator in indicator_list:
        ind_factory = IndicatorFactory.factory(indicator, 'buy', configs)
        if ind_factory:
            indicators.append(ind_factory)

    mocker.patch('api.binance_api.Binance.connect_to_api', return_value=None)
    mocker.patch('data.get_data.GetBinanceData.get_limit', return_value=210)
    mocker.patch('api.binance_api.Binance.get_klines', return_value=df.loc[488:699])
    mocker.patch('data.get_data.GetData.add_utc_3', return_value=df.loc[488:699])

    tickers = ['BTCUSDT', 'ETHUSDT']
    gd = DataFactory.factory('Binance', **configs)
    gd.limit = 500
    gd.fill_ticker_dict(tickers)

    data = gd.get_data(df.loc[:499], ticker, timeframe)[0]
    data_qty = 20
    res = gd.add_indicator_data(dfs, data, 'buy', indicators, ticker,
                                timeframe, data_qty)[ticker][timeframe]['data']['buy']
    dfs[ticker][timeframe]['data']['buy'] = res
    # res.to_pickle('test_BTCUSDT_1h_indicators.pkl')
    assert res.equals(expected[0])
