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


@pytest.mark.parametrize('signal_points, expected',
                         [
                          (points_btc, buy_btc_df),
                          (points_eth, buy_eth_df)
                          ], ids=repr)
def test_add_to_notification_history(mocker, signal_points, expected):
    mocker.patch('telegram.Bot.__init__', return_value=None)
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
          'STOCH_RSI', [], [], [], [])
point2 = ('BTCUSDT', '5m', 0, 'buy', pd.to_datetime('2022-08-21 3:45:00'),
          'STOCH_RSI', [], [], [], [])
point3 = ('ETHUSDT', '5m', 45, 'sell', pd.to_datetime('2022-08-26 09:00:00'),
          'STOCH_RSI', [], [], [], [])
point4 = ('ETHUSDT', '5m', 45, 'sell', pd.to_datetime('2022-08-21 09:00:00'),
          'STOCH_RSI', [], [], [], [])


@pytest.mark.parametrize('notification_df, points, point, expected',
                         [
                          (buy_btc_df, points_btc, point1, True),
                          (buy_btc_df, points_btc, point2, False),
                          (buy_eth_df, points_eth, point3, True),
                          (buy_eth_df, points_eth, point4, False)
                          ], ids=repr)
def test_check_previous_notifications(mocker, notification_df, points, point, expected):
    mocker.patch('telegram.Bot.__init__', return_value=None)
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
    mocker.patch('telegram.Bot.__init__', return_value=None)
    mocker.patch('telegram_api.telegram_api.TelegramBot.add_to_notification_history', return_value=None)
    mocker.patch('telegram_api.telegram_api.TelegramBot.send_notification', return_value=None)
    mocker.patch('telegram_api.telegram_api.TelegramBot.send_message', return_value=None)
    tb = TelegramBot('', database={}, **configs)
    for p in points:
        tb.notification_list.append(p)
    tb.check_notifications()
    assert tb.notification_list == []


point_m1 = ('BTCUSDT', '5m', 0, 'buy', pd.to_datetime('2022-08-24 14:45:00'),
            'Pattern', [], [], [], [])
point_m2 = ('BTCUSDT', '5m', 0, 'buy', pd.to_datetime('2022-08-24 16:45:00'),
            'Pattern', [], [], [], [])
point_m3 = ('ETHUSDT', '5m', 45, 'sell', pd.to_datetime('2022-08-24 14:45:00'),
            'STOCH_RSI', [], [], [], [])
point_m4 = ('ETHUSDT', '5m', 45, 'sell', pd.to_datetime('2022-08-24 15:45:00'),
            '', [], [], [], [])
point_m5 = ('BTCUSDT', '5m', 45, 'buy', pd.to_datetime('2022-08-22 22:40:00'),
            'Pump_Dump', [], [], [], [])
point_m6 = ('BTCUSDT', '5m', 45, 'buy', pd.to_datetime('2022-08-22 00:00:01'),
            'Pump_Dump', [], [], [], [])
point_m7 = ('BTCUSDT', '5m', 45, 'buy', pd.to_datetime('2022-08-22 23:59:00'),
            'Pump_Dump', [], [], [], [])
point_m8 = ('BTCUSDT', '5m', 45, 'sell', pd.to_datetime('2022-08-22 23:59:00'),
            'Pump_Dump', [], [], [], [])
point_m9 = ('BTCUSDT', '15m', 45, 'buy', pd.to_datetime('2022-08-22 23:59:00'),
            'Pump_Dump', [], [], [], [])
point_m10 = ('BTCUSDT', '15m', 45, 'buy', pd.to_datetime('2022-08-22 23:59:00'),
             'STOCH_RSI', [], [], [], [])
point_m11 = ('ETHUSDT', '15m', 45, 'buy', pd.to_datetime('2022-08-23 00:30:00'),
             'Pump_Dump', [], [], [], [])


@pytest.mark.parametrize('notification_df, points, point, expected',
                         [
                          (buy_btc_df, points_btc, point_m1, ['Pattern', 'STOCH_RSI']),
                          (buy_btc_df, points_btc, point_m2, []),
                          (buy_eth_df, points_eth, point_m3, []),
                          (buy_eth_df, points_eth, point_m4, []),
                          (buy_eth_df, points_eth, point_m5, ['Pump_Dump', 'STOCH_RSI_Trend']),
                          (buy_eth_df, points_eth, point_m6, []),
                          (buy_eth_df, points_eth, point_m7, ['Pump_Dump', 'STOCH_RSI_Trend']),
                          (buy_eth_df, points_eth, point_m8, []),
                          (buy_eth_df, points_eth, point_m9, ['Pump_Dump', 'STOCH_RSI_Trend']),
                          (buy_eth_df, points_eth, point_m10, []),
                          (buy_eth_df, points_eth, point_m11, ['Pattern', 'Pump_Dump', 'RSI_STOCH'])
                          ], ids=repr)
def test_check_multiple_notifications(mocker, notification_df, points, point, expected):
    mocker.patch('telegram.Bot.__init__', return_value=None)
    mocker.patch('telegram_api.telegram_api.TelegramBot.send_notification', return_value=None)
    mocker.patch('telegram_api.telegram_api.TelegramBot.send_message', return_value=None)
    tb = TelegramBot('', database={}, **configs)
    tb.notification_df = notification_df
    tb.add_to_notification_history(pd.to_datetime('2022-08-22 23:58:00'), 'buy', 'ETHUSDT', '1h', 'Pattern')
    tb.add_to_notification_history(pd.to_datetime('2022-08-22 23:01:00'), 'buy', 'ETHUSDT', '5m', 'RSI_STOCH')
    ticker = point[0]
    sig_type = point[3]
    sig_time = point[4]
    sig_pattern = str(point[5])
    tb.check_multiple_notifications(sig_time, sig_type, ticker, sig_pattern)
    assert tb.check_multiple_notifications(sig_time, sig_type, ticker, sig_pattern) == expected
