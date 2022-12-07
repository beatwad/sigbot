import pytest
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
    indicator = IndicatorFactory.factory('RSI', 'buy', configs)
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
    indicator = IndicatorFactory.factory('STOCH', 'buy', configs)
    data_qty = 20
    df = indicator.get_indicator(df, ticker, timeframe, data_qty)
    assert round(df.loc[500, 'stoch_slowk'], 2) == expected[0][0]
    assert round(df.loc[500, 'stoch_slowd'], 2) == expected[0][1]
    assert round(df.loc[998, 'stoch_slowk'], 2) == expected[1][0]
    assert round(df.loc[998, 'stoch_slowd'], 2) == expected[1][1]


@pytest.mark.parametrize('df, ticker, timeframe, expected',
                         [
                          (df_btc_5m, 'BTC', '5m', [[0.00261, 0.169, 0.236, -0.00444, 0.00343,
                                                    -0.5287, 0.55779, -0.66101, 0.6836],
                                                    [3e-05, -0.0793, -0.0144, -0.00444, 0.00343,
                                                     -0.5287, 0.55779, -0.66101, 0.6836],
                                                    [-0.00096, -0.0799, 0.0639, -0.00444, 0.00343,
                                                     -0.5287, 0.55779, -0.66101, 0.6836]]),
                          (df_eth_5m, 'ETH', '5m', [[5e-05, 0.0191, -0.1887, -0.00629, 0.00707,
                                                    -0.79934, 1.11272, -0.99277, 1.37686],
                                                    [-0.00299, 0.2201, 0.2561, -0.00629, 0.00707,
                                                     -0.79934, 1.11272, -0.99277, 1.37686],
                                                    [-0.00093, -0.3034, -0.3124, -0.00629, 0.00707,
                                                     -0.79934, 1.11272, -0.99277, 1.37686]]),
                          ], ids=repr)
def test_price_change(df, timeframe, ticker, expected):
    indicator = IndicatorFactory.factory('PriceChange', 'buy', configs)
    data_qty = 500
    df = indicator.get_indicator(df, ticker, timeframe, data_qty)
    assert round(df.loc[223, 'close_price_change_lag_1'], 5) == expected[0][0]
    assert round(df.loc[223, 'q_low_lag_1'], 5) == expected[0][3]
    assert round(df.loc[223, 'q_high_lag_1'], 5) == expected[0][4]
    assert round(df.loc[500, 'close_price_change_lag_1'], 5) == expected[1][0]
    assert round(df.loc[500, 'q_low_lag_1'], 5) == expected[1][3]
    assert round(df.loc[500, 'q_high_lag_1'], 5) == expected[1][4]
    assert round(df.loc[998, 'close_price_change_lag_1'], 5) == expected[2][0]
    assert round(df.loc[998, 'q_low_lag_1'], 5) == expected[2][3]
    assert round(df.loc[998, 'q_high_lag_1'], 5) == expected[2][4]


@pytest.mark.parametrize('df, ticker, timeframe, expected',
                         [
                          (df_btc_5m, 'BTC', '5m', [[0.00084, 0.00328],
                                                    [0.00415, 0.00328],
                                                    [0.00053, 0.00328]]),
                          (df_eth_5m, 'ETH', '5m', [[0.00082, 0.00438],
                                                    [0.00358, 0.00438],
                                                    [0.00095, 0.00438]]),
                          ], ids=repr)
def test_high_volume(df, timeframe, ticker, expected):
    indicator = IndicatorFactory.factory('HighVolume', 'buy', configs)
    data_qty = 500
    df = indicator.get_indicator(df, ticker, timeframe, data_qty)
    assert round(df.loc[122, 'normalized_vol'], 5) == expected[0][0]
    assert round(df.loc[122, 'quantile_vol'], 5) == expected[0][1]
    assert round(df.loc[375, 'normalized_vol'], 5) == expected[1][0]
    assert round(df.loc[375, 'quantile_vol'], 5) == expected[1][1]
    assert round(df.loc[856, 'normalized_vol'], 5) == expected[2][0]
    assert round(df.loc[856, 'quantile_vol'], 5) == expected[2][1]
