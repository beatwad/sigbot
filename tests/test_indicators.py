import pytest
import pandas as pd
from os import environ
from config.config import ConfigFactory
from indicators.indicators import IndicatorFactory


# Set environment variable
environ["ENV"] = "test"

configs = ConfigFactory.factory(environ).configs

df_btc_5m = pd.read_pickle('test_BTCUSDT_5m.pkl')
df_btc_1h = pd.read_pickle('test_BTCUSDT_1h.pkl')
df_btc_4h = pd.read_pickle('test_BTCUSDT_4h.pkl')
df_eth_5m = pd.read_pickle('test_ETHUSDT_5m.pkl')
df_eth_1h = pd.read_pickle('test_ETHUSDT_1h.pkl')
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
    df = indicator.get_indicator(df, ticker, timeframe)
    assert round(df.loc[223, 'rsi'], 2) == expected[0]
    assert round(df.loc[500, 'rsi'], 2) == expected[1]
    assert round(df.loc[998, 'rsi'], 2) == expected[2]


@pytest.mark.parametrize('df, ticker, timeframe, expected',
                         [
                          (df_btc_5m, 'BTC', '5m', [[20.34, 19.40], [89.86, 90.22]]),
                          (df_btc_1h, 'BTC', '1h', [[73.74, 70.07], [42.30, 37.13]]),
                          (df_btc_4h, 'BTC', '4h', [[19.76, 16.08], [56.57, 60.00]]),
                          (df_eth_5m, 'ETH', '5m', [[50.73, 41.10], [70.10, 78.92]]),
                          (df_eth_1h, 'ETH', '1h', [[68.13, 69.07], [49.57, 43.51]]),
                          (df_eth_1d, 'ETH', '1d', [[92.95, 90.47], [29.37, 25.43]]),
                          ], ids=repr)
def test_stoch(df, timeframe, ticker, expected):
    indicator = IndicatorFactory.factory('STOCH', configs)
    df = indicator.get_indicator(df, ticker, timeframe)
    assert round(df.loc[500, 'stoch_slowk'], 2) == expected[0][0]
    assert round(df.loc[500, 'stoch_slowd'], 2) == expected[0][1]
    assert round(df.loc[998, 'stoch_slowk'], 2) == expected[1][0]
    assert round(df.loc[998, 'stoch_slowd'], 2) == expected[1][1]

