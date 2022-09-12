import pytest
import numpy as np
import pandas as pd
from os import environ
from config.config import ConfigFactory
from indicators.indicators import IndicatorFactory


# Set environment variable
environ["ENV"] = "test"

configs = ConfigFactory.factory(environ).configs

# Test RSI indicator
df_btc_5m = pd.read_pickle('test_BTCUSDT_5m.pkl')
df_btc_1h = pd.read_pickle('test_BTCUSDT_1h.pkl')
df_btc_4h = pd.read_pickle('test_BTCUSDT_4h.pkl')
df_eth_5m = pd.read_pickle('test_ETHUSDT_5m.pkl')
df_eth_1h = pd.read_pickle('test_ETHUSDT_1h.pkl')
df_eth_4h = pd.read_pickle('test_ETHUSDT_4h.pkl')
df_eth_1d = pd.read_pickle('test_ETHUSDT_1d.pkl')


@pytest.mark.parametrize('df, ticker, timeframe, expected',
                         [
                          (df_btc_5m, 'BTC', '5m', [61.75, 28.44, 62.72]),
                          (df_btc_1h, 'BTC', '1h', [47.18, 62.53, 54.08]),
                          (df_btc_4h, 'BTC', '4h', [56.92, 38.92, 49.16]),
                          (df_eth_5m, 'ETH', '5m', [52.11, 43.74, 55.91]),
                          (df_eth_1h, 'ETH', '1h', [56.50, 57.35, 57.65]),
                          (df_eth_1d, 'ETH', '1d', [54.73, 73.01, 50.11]),
                          ], ids=repr)
def test_rsi(df, timeframe, ticker, expected):
    indicator = IndicatorFactory.factory('RSI', configs)
    data_qty = 20
    df = indicator.get_indicator(df, ticker, timeframe, data_qty)
    assert round(df.loc[223, 'rsi'], 2) == expected[0]
    assert round(df.loc[500, 'rsi'], 2) == expected[1]
    assert round(df.loc[998, 'rsi'], 2) == expected[2]


# Test Stochastic indicator
@pytest.mark.parametrize('df, ticker, timeframe, expected',
                         [
                          (df_btc_5m, 'BTC', '5m', [[20.34, 19.40], [89.86, 90.22]]),
                          (df_btc_5m, 'BTC', '5m', [[20.34, 19.40], [89.86, 90.22]]),
                          (df_btc_1h, 'BTC', '1h', [[73.74, 70.07], [42.30, 37.13]]),
                          (df_btc_4h, 'BTC', '4h', [[19.76, 16.08], [56.57, 60.00]]),
                          (df_eth_5m, 'ETH', '5m', [[50.73, 41.10], [70.10, 78.92]]),
                          (df_eth_1h, 'ETH', '1h', [[68.13, 69.07], [49.57, 43.51]]),
                          (df_eth_1d, 'ETH', '1d', [[92.95, 90.47], [29.37, 25.43]]),
                          ], ids=repr)
def test_stoch(df, timeframe, ticker, expected):
    indicator = IndicatorFactory.factory('STOCH', configs)
    data_qty = 20
    df = indicator.get_indicator(df, ticker, timeframe, data_qty)
    assert round(df.loc[500, 'stoch_slowk'], 2) == expected[0][0]
    assert round(df.loc[500, 'stoch_slowd'], 2) == expected[0][1]
    assert round(df.loc[998, 'stoch_slowk'], 2) == expected[1][0]
    assert round(df.loc[998, 'stoch_slowd'], 2) == expected[1][1]


# Test adding support and resistance levels
levels1 = [[20937.46, 2], [21011.22, 2], [21063.11, 2], [21145.0, 2], [21205.0, 2], [21271.64, 2],
           [21326.32, 2], [21377.73, 2], [21444.0, 2], [21499.25, 2], [21555.84, 2], [21654.91, 2],
           [21800.0, 1]]
levels2 = [[19645.8, 2], [20539.85, 2], [20890.14, 2], [21167.58, 2], [21452.0, 2], [21684.87, 2],
           [22028.14, 2], [22449.09, 2], [22837.34, 2], [23180.4, 2], [23448.0, 2], [23809.0, 2],
           [24150.0, 2], [24427.03, 2], [24745.0, 1], [24982.7, 2]]
levels3 = [[940.95, 1], [1005.24, 2], [1072.11, 2], [1127.19, 2], [1196.0, 2], [1287.82, 2], [1356.17, 2],
           [1488.0, 2], [1549.25, 2], [1626.14, 2], [1713.16, 2], [1824.15, 2], [1879.73, 2], [1959.43, 2],
           [2017.41, 2], [2088.5, 2], [2145.29, 2], [2200.0, 1], [2451.37, 2], [2517.0, 2], [2579.03, 2],
           [2632.95, 2], [2704.36, 2], [2771.74, 2], [2855.07, 2], [2954.65, 2], [3039.54, 2], [3107.43, 2],
           [3180.0, 2], [3242.95, 2], [3307.22, 2], [3405.97, 2], [3523.33, 2], [3580.34, 2]]
levels4 = [[86.0, 2], [232.32, 2], [370.23, 2], [565.43, 2], [936.15, 2], [1072.11, 2], [1207.0, 2], [1356.17, 2],
           [1529.92, 2], [1744.85, 2], [1877.69, 2], [2012.47, 2], [2159.0, 2], [2300.0, 2], [2507.0, 2],
           [2709.26, 2], [2954.65, 2], [3180.0, 2], [3393.6, 2], [3555.0, 2], [3695.0, 2], [3900.73, 2], [4137.91, 2],
           [4488.0, 2], [4634.83, 2], [4868.0, 1]]
levels = [levels1, levels2, levels3, levels4]


@pytest.mark.parametrize('df, ticker, timeframe, expected',
                         [
                          (pd.DataFrame(columns=df_btc_5m.columns), 'BTC', '5m', []),
                          (df_btc_5m, 'BTC', '5m', levels[0]),
                          (df_btc_1h, 'BTC', '1h', levels[1]),
                          (df_eth_4h, 'ETH', '4h', levels[2]),
                          (df_eth_1d, 'ETH', '1d', levels[3]),
                          ], ids=repr)
def test_sup_res_find_levels(df, timeframe, ticker, expected):
    indicator = IndicatorFactory.factory('SUP_RES', configs)
    level_proximity = np.mean(df['high'] - df['low'])
    res = indicator.find_levels(df, level_proximity)
    assert res == expected


# Test adding support and resistance levels from higher timeframe to current timeframe
s1 = np.mean(df_btc_5m['high'] - df_btc_5m['low'])
s2 = np.mean(df_eth_4h['high'] - df_eth_4h['low'])


merged_levels1 = [[20937.46, 3], [21011.22, 2], [21063.11, 2], [21145.0, 3], [21205.0, 3],
                  [21271.64, 2], [21326.32, 2], [21377.73, 2], [21444.0, 3], [21499.25, 3],
                  [21555.84, 2], [21654.91, 3], [21800.0, 1]]
merged_levels2 = [[940.95, 3], [1005.24, 2], [1072.11, 3], [1127.19, 2], [1196.0, 3], [1287.82, 2],
                  [1356.17, 3], [1488.0, 3], [1549.25, 3], [1626.14, 2], [1713.16, 3], [1824.15, 3],
                  [1879.73, 3], [1959.43, 3], [2017.41, 3], [2088.5, 2], [2145.29, 3], [2200.0, 3],
                  [2451.37, 2], [2517.0, 3], [2579.03, 2], [2632.95, 2], [2704.36, 3], [2771.74, 2],
                  [2855.07, 2], [2954.65, 3], [3039.54, 2], [3107.43, 2], [3180.0, 3],  [3242.95, 2],
                  [3307.22, 2], [3405.97, 3], [3523.33, 3], [3580.34, 3], [2300.0, 2]]
merged_levels3 = [[20937.46, 2], [21011.22, 2], [21063.11, 2], [21145.0, 2], [21205.0, 2],
                  [21271.64, 2], [21326.32, 2], [21377.73, 2], [21444.0, 2], [21499.25, 2],
                  [21555.84, 2], [21654.91, 2], [21800.0, 1]]
merged_levels4 = [[940.95, 1], [1005.24, 2], [1072.11, 2], [1127.19, 2], [1196.0, 2], [1287.82, 2],
                  [1356.17, 2], [1488.0, 2], [1549.25, 2], [1626.14, 2], [1713.16, 2], [1824.15, 2],
                  [1879.73, 2], [1959.43, 2], [2017.41, 2], [2088.5, 2], [2145.29, 2], [2200.0, 1],
                  [2451.37, 2], [2517.0, 2], [2579.03, 2], [2632.95, 2], [2704.36, 2], [2771.74, 2],
                  [2855.07, 2], [2954.65, 2], [3039.54, 2], [3107.43, 2], [3180.0, 2],  [3242.95, 2],
                  [3307.22, 2], [3405.97, 2], [3523.33, 2], [3580.34, 2]]


@pytest.mark.parametrize('levels, higher_levels, s, expected',
                         [
                          ([], [], s1, []),
                          (levels[0], [], s1, levels[0]),
                          ([], levels[1], s1, levels[1]),
                          (levels[0], levels[1], s1, merged_levels1),
                          (levels[2], levels[3], s2, merged_levels2),
                          ], ids=repr)
def test_add_higher_levels(levels, higher_levels, s, expected):
    indicator = IndicatorFactory.factory('SUP_RES', configs)
    merged_levels = indicator.add_higher_levels(levels, higher_levels, s)
    assert merged_levels == expected

# Test adding levels in complex


@pytest.mark.parametrize('df, ticker, timeframe, higher_levels, merge, expected',
                         [
                          (df_btc_5m, 'BTC', '5m', levels[1], True, merged_levels1),
                          (df_eth_4h, 'ETH', '4h', levels[3], True, merged_levels2),
                          (df_btc_5m, 'BTC', '5m', levels[1], False, merged_levels3),
                          (df_eth_4h, 'ETH', '4h', levels[3], False, merged_levels4),
                          ], ids=repr)
def test_sup_res_find_levels(df, timeframe, ticker, higher_levels, merge, expected):
    indicator = IndicatorFactory.factory('SUP_RES', configs)
    data_qty = 20
    res = indicator.get_indicator(df, ticker, timeframe, data_qty, higher_levels, merge)
    assert res == expected
