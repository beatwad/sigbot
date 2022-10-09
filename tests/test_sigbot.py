import pytest
from os import environ

from config.config import ConfigFactory
environ["ENV"] = "test"
from bot.bot import SigBot

# Get configs
configs = ConfigFactory.factory(environ).configs


# expected = (['ETHUSDT', 'TRXUSDT', 'QTUMUSDT', 'ONTUSDT',  'TTTUSDT'],
#             ['BTCUSDT', 'VETUSDT', 'HOTUSDT', 'FETUSDT', 'SVMUSDT', 'NNUSDT'],
#             ['ETH', 'QTUM', 'ONT', 'VET', 'HOT', 'SVM', 'TRX', 'TTT'])
expected = (['TRXUSDT', 'TTTUSDT'],
            ['BTCUSDT', 'ETHUSDT', 'QTUMUSDT', 'ONTUSDT', 'VETUSDT', 'HOTUSDT', 'FETUSDT', 'SVMUSDT', 'NNUSDT'],
            ['ETH', 'QTUM', 'ONT', 'VET', 'HOT', 'SVM', 'TRX', 'TTT'])


@pytest.mark.parametrize('expected',
                         [
                          expected,
                          ], ids=repr)
def test_filter_used_tickers(mocker, expected):
    mocker.patch('telegram_api.telegram_api.TelegramBot.start', return_value=None)
    mocker.patch('telegram_api.telegram_api.TelegramBot.__init__', return_value=None)
    mocker.patch('signal_stat.signal_stat.SignalStat.__init__', return_value=None)
    mocker.patch('signal_stat.signal_stat.SignalStat.load_statistics', return_value=(None, None))
    mocker.patch('data.get_data.GetData.__init__', return_value=None)
    mocker.patch('data.get_data.GetData.get_tickers', return_value=([], []))
    mocker.patch('data.get_data.GetBinanceData.__init__', return_value=None)
    mocker.patch('data.get_data.GetBinanceData.fill_ticker_dict', return_value=None)
    mocker.patch('data.get_data.GetOKEXData.__init__', return_value=None)
    mocker.patch('data.get_data.GetOKEXData.fill_ticker_dict', return_value=None)
    main = None
    mn = SigBot(main, **configs)
    mn.exchanges = {'Binance': {'API': None, 'tickers': [], 'all_tickers': []},
                    'OKEX': {'API': None, 'tickers': [], 'all_tickers': []}}
    mn.used_tickers = ['ETH', 'QTUM', 'ONT', 'VET', 'HOT', 'SVM']
    tickers = ['ETHUSDT', 'TRXUSDT', 'QTUMUSDT', 'ONTUSDT', 'VETUSDT', 'HOTUSDT', 'SVMUSDT', 'TTTUSDT']
    prev_tickers = ['BTCUSDT', 'ETHUSDT', 'QTUMUSDT', 'ONTUSDT', 'VETUSDT', 'HOTUSDT', 'FETUSDT', 'SVMUSDT', 'NNUSDT']
    # not_used_tickers, prev_tickers
    assert mn.filter_used_tickers(tickers, prev_tickers, 2) == (expected[0], expected[1])
    assert mn.used_tickers == expected[2]


tickers1 = ['ETHUSDT', 'TRXUSDT', 'QTUMUSDT', 'ONTUSDT', 'VETUSDT', 'HOTUSDT', 'SVMUSDT', 'TTTUSDT']
tickers2 = ['ETHUSDT', 'TRXUSDT']
tickers3 = ['ETHUSDT', 'TRXUSDT', 'QTUMUSDT']
tickers = [tickers1, tickers2, tickers3]

indexes1 = [1, 4, 5]
indexes2 = [1]
indexes3 = []
indexes = [indexes1, indexes2, indexes3]

expected1 = ['ETHUSDT', 'QTUMUSDT', 'ONTUSDT', 'SVMUSDT', 'TTTUSDT']
expected2 = ['ETHUSDT']
expected3 = ['ETHUSDT', 'TRXUSDT', 'QTUMUSDT']
expected = [expected1, expected2, expected3]


@pytest.mark.parametrize('tickers, indexes, expected',
                         [
                          (tickers[0], indexes[0], expected[0]),
                          (tickers[1], indexes[1], expected[1]),
                          (tickers[2], indexes[2], expected[2])
                          ], ids=repr)
def test_clean_prev_exchange_tickers(mocker, tickers, indexes, expected):
    mocker.patch('telegram_api.telegram_api.TelegramBot.start', return_value=None)
    mocker.patch('telegram_api.telegram_api.TelegramBot.__init__', return_value=None)
    mocker.patch('signal_stat.signal_stat.SignalStat.__init__', return_value=None)
    mocker.patch('signal_stat.signal_stat.SignalStat.load_statistics', return_value=(None, None))
    mocker.patch('data.get_data.GetData.__init__', return_value=None)
    mocker.patch('data.get_data.GetData.get_tickers', return_value=([], []))
    mocker.patch('data.get_data.GetBinanceData.__init__', return_value=None)
    mocker.patch('data.get_data.GetBinanceData.fill_ticker_dict', return_value=None)
    mocker.patch('data.get_data.GetOKEXData.__init__', return_value=None)
    mocker.patch('data.get_data.GetOKEXData.fill_ticker_dict', return_value=None)
    main = None
    mn = SigBot(main, **configs)
    mn.exchanges = {'Binance': {'API': None, 'tickers': [], 'all_tickers': []},
                    'OKEX': {'API': None, 'tickers': [], 'all_tickers': []}}
    # not_used_tickers, prev_tickers
    assert mn.clean_prev_exchange_tickers(tickers, indexes) == expected
