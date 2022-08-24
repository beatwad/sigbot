import pytest
from os import environ
import pandas as pd
from data.get_data import DataFactory
from config.config import ConfigFactory
import datetime
from freezegun import freeze_time


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

@pytest.mark.parametrize('exchange, df, timeframe, expected',
                         [
                          ('Binance', df_btc_5m, '5m', 3),
                          ('Binance', df_btc_5m, '1h', 2),
                          ('Binance', df_btc_5m, '4h', 7),
                          ('Binance', df_btc_5m, '1d', 5),
                          ], ids=repr)
def test_get_interval(exchange, df, timeframe, expected):
    gd = DataFactory.factory(exchange, **configs)
    gd.timestamp_dict['5m'] = datetime.datetime(2022, 8, 24, 15, 0, 0, 0)
    gd.timestamp_dict['1h'] = datetime.datetime(2022, 8, 24, 14, 0, 0, 0)
    gd.timestamp_dict['4h'] = datetime.datetime(2022, 8, 23, 14, 0, 0, 0)
    gd.timestamp_dict['1d'] = datetime.datetime(2022, 8, 20, 10, 0, 0, 0)

    @freeze_time("2022-08-24-15:11:00")
    def get_interval():
        return gd.get_interval(df, timeframe)

    interval = get_interval()
    assert interval == expected


