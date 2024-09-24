import pytest
import pandas as pd
from os import environ
from unittest.mock import MagicMock
from config.config import ConfigFactory

# Set environment variable
environ["ENV"] = "test"

from telegram_api.telegram_api import TelegramBot

# Get configs
configs = ConfigFactory.factory(environ).configs


points_btc = [('BTCUSDT', '5m', 0, 'buy', pd.to_datetime('2022-08-21 3:45:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 20, 'sell', pd.to_datetime('2022-08-21 8:25:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 98, 'sell', pd.to_datetime('2022-08-21 11:55:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 513, 'buy', pd.to_datetime('2022-08-22 22:30:00'),
               'STOCH_RSI_Trend', [], [], [], []
               ),
              ('BTCUSDT', '5m', 576, 'sell', pd.to_datetime('2022-08-23 03:45:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 673, 'sell', pd.to_datetime('2022-08-23 11:50:00'),
               'STOCH_RSI_Trend', [], [], [], []
               ),
              ('BTCUSDT', '5m', 745, 'buy', pd.to_datetime('2022-08-23 17:50:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 977, 'sell', pd.to_datetime('2022-08-24 12:05:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('BTCUSDT', '5m', 995, 'buy', pd.to_datetime('2022-08-24 14:40:00'),
               'STOCH_RSI', [], [], [], []
               )]

points_eth = [('ETHUSDT', '5m', 45, 'sell', pd.to_datetime('2022-08-21 09:00:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('ETHUSDT', '5m', 47, 'buy', pd.to_datetime('2022-08-21 09:10:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('ETHUSDT', '5m', 97, 'sell', pd.to_datetime('2022-08-21 12:30:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('ETHUSDT', '5m', 512, 'buy', pd.to_datetime('2022-08-22 23:05:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('ETHUSDT', '5m', 645, 'buy', pd.to_datetime('2022-08-23 10:10:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('ETHUSDT', '5m', 840, 'sell', pd.to_datetime('2022-08-24 02:25:00'),
               'STOCH_RSI', [], [], [], []
               ),
              ('ETHUSDT', '5m', 985, 'sell', pd.to_datetime('2022-08-24 12:00:00'),
               'STOCH_RSI_Trend', [], [], [], []
               ),
              ('ETHUSDT', '5m', 990, 'buy', pd.to_datetime('2022-08-24 12:25:00'),
               'STOCH_RSI_Trend', [], [], [], []
               )
              ]

buy_btc_df = pd.read_pickle('test_BTCUSDT_5m_telegram_notifications.pkl')
buy_eth_df = pd.read_pickle('test_ETHUSDT_5m_telegram_notifications.pkl')


@pytest.mark.parametrize('ticker, expected',
                         [
                          ('BTC-USDT', 'BTC-USDT'),
                          ('BTC/USDT', 'BTC-USDT'),
                          ('BTCUSDT', 'BTC-USDT')
                          ],
                         ids=repr)
def test_process_ticker(mocker, ticker, expected):
    mocker.patch('telegram.Bot.__init__', return_value=None)
    tb = TelegramBot('', database={}, trade_mode=[0], locker=MagicMock(), **configs)
    assert tb.process_ticker(ticker) == expected


@pytest.mark.parametrize('ticker, expected',
                         [
                          ('BTCUSDT', 'BTCUSDT'),
                          ('BTC-USDT-SWAP', 'BTCUSDT'),
                          ('BTC_USDT', 'BTCUSDT'),
                          ('BTC-USDT', 'BTCUSDT'),
                          ('10000BTC-USDT', '10000BTCUSDT.P'),
                          ('100BTC-USDT', '100BTCUSDT'),
                          ],
                         ids=repr)
def test_clean_ticker(mocker, ticker, expected):
    mocker.patch('telegram.Bot.__init__', return_value=None)
    tb = TelegramBot('', database={}, trade_mode=[0], locker=MagicMock(), **configs)
    assert tb.clean_ticker(ticker) == expected
