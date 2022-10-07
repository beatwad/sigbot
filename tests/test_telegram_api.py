import sys
import pytest
from os import environ
from config.config import ConfigFactory
import pandas as pd
# Set environment variable
environ["ENV"] = "test"
from telegram_api.telegram_api import TelegramBot

# Get configs
configs = ConfigFactory.factory(environ).configs


points_btc = [('BTCUSDT', '5m', 0, 'buy', pd.to_datetime('2022-08-21 3:45:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('BTCUSDT', '5m', 20, 'sell', pd.to_datetime('2022-08-21 8:25:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('BTCUSDT', '5m', 98, 'sell', pd.to_datetime('2022-08-21 11:55:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('BTCUSDT', '5m', 513, 'buy', pd.to_datetime('2022-08-22 22:30:00'),
               [('STOCH', (15, 85)), ('RSI', (10, 90)), ('LinearReg', ())], [], [], [], []
               ),
              ('BTCUSDT', '5m', 576, 'sell', pd.to_datetime('2022-08-23 03:45:00'),
               [('STOCH', (20, 70)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('BTCUSDT', '5m', 673, 'sell', pd.to_datetime('2022-08-23 11:50:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75)), ('SUP_RES', ())], [], [], [], []
               ),
              ('BTCUSDT', '5m', 745, 'buy', pd.to_datetime('2022-08-23 17:50:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('BTCUSDT', '5m', 977, 'sell', pd.to_datetime('2022-08-24 12:05:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('BTCUSDT', '5m', 995, 'buy', pd.to_datetime('2022-08-24 14:40:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               )]

points_eth = [('ETHUSDT', '5m', 45, 'sell', pd.to_datetime('2022-08-21 09:00:00'),
               [('STOCH', (15, 85)), ('RSI', (20, 80))], [], [], [], []
               ),
              ('ETHUSDT', '5m', 47, 'buy', pd.to_datetime('2022-08-21 09:10:00'),
               [('STOCH', (15, 85)), ('RSI', (15, 85))], [], [], [], []
               ),
              ('ETHUSDT', '5m', 97, 'sell', pd.to_datetime('2022-08-21 12:30:00'),
               [('STOCH', (0, 100)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('ETHUSDT', '5m', 512, 'buy', pd.to_datetime('2022-08-22 23:05:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], []
               ),
              ('ETHUSDT', '5m', 645, 'buy', pd.to_datetime('2022-08-23 10:10:00'),
               [('STOCH', (1, 99)), ('RSI', (25, 75)), ('SUP_RES', ())], [], [], [], []
               ),
              ('ETHUSDT', '5m', 840, 'sell', pd.to_datetime('2022-08-24 02:25:00'),
               [('STOCH', (15, 85)), ('RSI', (35, 80))], [], [], [], []
               ),
              ('ETHUSDT', '5m', 985, 'sell', pd.to_datetime('2022-08-24 12:00:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75)), ('LinearReg', ())], [], [], [], []
               ),
              ('ETHUSDT', '5m', 990, 'buy', pd.to_datetime('2022-08-24 12:25:00'),
               [('STOCH', (15, 85)), ('RSI', (25, 75)), ('LinearReg', ())], [], [], [], []
               )
              ]

buy_btc_df = pd.read_pickle('test_BTCUSDT_5m_telegram_notifications.pkl')
buy_eth_df = pd.read_pickle('test_ETHUSDT_5m_telegram_notifications.pkl')


@pytest.mark.parametrize('signal_points, expected',
                         [
                          (points_btc, buy_btc_df),
                          (points_eth, buy_eth_df)
                          ], ids=repr)
def test_add_to_notification_history(mocker, signal_points, expected):
    mocker.patch('telegram.ext.updater.Updater.__init__', return_value=None)
    mocker.patch('telegram.ext.updater.Updater.__getattribute__', return_value=None)
    tb = TelegramBot('', database={}, **configs)
    for point in points_btc:
        ticker = point[0]
        timeframe = point[1]
        sig_type = point[3]
        sig_time = point[4]
        pattern = point[5]
        tb.add_to_notification_history(sig_time, sig_type, ticker, timeframe, pattern)
    assert tb.notification_df.equals(expected)


point1 = ('BTCUSDT', '5m', 0, 'buy', pd.to_datetime('2022-08-26 3:45:00'),
          [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], [])
point2 = ('BTCUSDT', '5m', 0, 'buy', pd.to_datetime('2022-08-21 3:45:00'),
          [('STOCH', (15, 85)), ('RSI', (25, 75))], [], [], [], [])
point3 = ('ETHUSDT', '5m', 45, 'sell', pd.to_datetime('2022-08-26 09:00:00'),
          [('STOCH', (15, 85)), ('RSI', (20, 80))], [], [], [], [])
point4 = ('ETHUSDT', '5m', 45, 'sell', pd.to_datetime('2022-08-21 09:00:00'),
          [('STOCH', (15, 85)), ('RSI', (20, 80))], [], [], [], [])


@pytest.mark.parametrize('notification_df, points, point, expected',
                         [
                          (buy_btc_df, points_btc, point1, True),
                          (buy_btc_df, points_btc, point2, False),
                          (buy_eth_df, points_eth, point3, True),
                          (buy_eth_df, points_eth, point4, False)
                          ], ids=repr)
def test_check_previous_notifications(mocker, notification_df, points, point, expected):
    mocker.patch('telegram.ext.updater.Updater.__init__', return_value=None)
    mocker.patch('telegram.ext.updater.Updater.__getattribute__', return_value=None)
    tb = TelegramBot('', database={}, **configs)
    for p in points:
        ticker = p[0]
        timeframe = p[1]
        sig_type = p[3]
        sig_time = p[4]
        pattern = str(p[5])
        tb.add_to_notification_history(sig_time, sig_type, ticker, timeframe, pattern)
    ticker = point[0]
    timeframe = point[1]
    sig_type = point[3]
    sig_time = point[4]
    pattern = str(point[5])
    assert tb.check_previous_notifications(sig_time, sig_type, ticker, timeframe, pattern) == expected


@pytest.mark.parametrize('notification_df, points, point, expected',
                         [
                          (buy_btc_df, points_btc, point1, True),
                          (buy_btc_df, points_btc, point2, False),
                          (buy_eth_df, points_eth, point3, True),
                          (buy_eth_df, points_eth, point4, False)
                          ], ids=repr)
def test_check_notifications(mocker, notification_df, points, point, expected):
    mocker.patch('telegram.ext.updater.Updater.__init__', return_value=None)
    mocker.patch('telegram.ext.updater.Updater.__getattribute__', return_value=None)
    mocker.patch('telegram_api.telegram_api.TelegramBot.add_to_notification_history', return_value=None)
    mocker.patch('telegram_api.telegram_api.TelegramBot.send_notification', return_value=None)
    mocker.patch('telegram_api.telegram_api.TelegramBot.send_message', return_value=None)
    tb = TelegramBot('', database={}, **configs)
    for p in points:
        tb.notification_list.append(p)
    tb.check_notifications()
    assert tb.notification_list == []
