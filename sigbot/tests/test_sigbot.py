import pytest
from os import environ
from config.config import ConfigFactory

environ["ENV"] = "test"

from bot.bot import SigBot

# Get configs
configs = ConfigFactory.factory(environ).configs

expected = (['TRXUSDT', 'TTTUSDT'], [265, 10],
            ['ETH', 'QTUM', 'ONT', 'VET', 'HOT', 'SVM', 'TRX', 'TTT'])


@pytest.mark.parametrize('expected',
                         [
                          expected,
                          ], ids=repr)
def test_filter_used_tickers(mocker, expected):
    mocker.patch('telegram_api.telegram_api.TelegramBot.__init__', return_value=None)
    mocker.patch('signal_stat.signal_stat.SignalStat.__init__', return_value=None)
    mocker.patch('signal_stat.signal_stat.SignalStat.load_statistics', return_value=(None, None))
    mocker.patch('data.get_data.GetData.__init__', return_value=None)
    mocker.patch('data.get_data.GetData.get_tickers', return_value=([], [], []))
    mocker.patch('data.get_data.GetBinanceData.__init__', return_value=None)
    mocker.patch('data.get_data.GetBinanceData.fill_ticker_dict', return_value=None)
    mocker.patch('data.get_data.GetOKEXData.__init__', return_value=None)
    mocker.patch('data.get_data.GetOKEXData.fill_ticker_dict', return_value=None)
    mocker.patch('joblib.load', return_value=None)
    mocker.patch('builtins.open', mocker.mock_open(read_data=None))
    mocker.patch('json.load', return_value=None)
    main = None
    mn = SigBot(main, **configs)
    mn.exchanges = {'Binance': {'API': None, 'tickers': [], 'all_tickers': []},
                    'OKEX': {'API': None, 'tickers': [], 'all_tickers': []}}
    mn.used_tickers = ['ETH', 'QTUM', 'ONT', 'VET', 'HOT', 'SVM']
    tickers = ['ETHUSDT', 'TRXUSDT', 'QTUMUSDT', 'ONTUSDT', 'VETUSDT', 'HOTUSDT', 'SVMUSDT', 'TTTUSDT']
    prev_tickers = [1488, 265, 322, 228, 309, 300, 1, 10]
    # not_used_tickers, prev_tickers
    assert mn.filter_used_tickers(tickers, prev_tickers) == (expected[0], expected[1])
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
