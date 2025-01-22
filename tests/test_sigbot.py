from datetime import datetime
from os import environ
from unittest.mock import MagicMock

import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from config.config import ConfigFactory

environ["ENV"] = "test"

from bot.bot import SigBot

# Get configs
configs = ConfigFactory.factory(environ).configs

expected = (
    ["TRXUSDT", "TTTUSDT"],
    [265, 10],
    ["ETH", "QTUM", "ONT", "VET", "HOT", "SVM", "TRX", "TTT"],
)


@pytest.fixture
def sigbot(mocker):
    mocker.patch("telegram_api.telegram_api.TelegramBot.__init__", return_value=None)
    mocker.patch("signal_stat.signal_stat.SignalStat.__init__", return_value=None)
    mocker.patch("signal_stat.signal_stat.SignalStat.load_statistics", return_value=(None, None))
    mocker.patch("data.get_data.GetData.__init__", return_value=None)
    mocker.patch("data.get_data.GetData.get_tickers", return_value=([], [], []))
    mocker.patch("data.get_data.GetBinanceData.__init__", return_value=None)
    mocker.patch("data.get_data.GetBinanceData.fill_ticker_dict", return_value=None)
    mocker.patch("data.get_data.GetOKEXData.__init__", return_value=None)
    mocker.patch("data.get_data.GetOKEXData.fill_ticker_dict", return_value=None)
    mocker.patch("joblib.load", return_value=None)
    mocker.patch("builtins.open", mocker.mock_open(read_data=None))
    mocker.patch("json.load", return_value=None)
    main = MagicMock(cycle_number=2)
    sigbot = SigBot(main, **configs)
    sigbot.exchanges = {
        "Binance": {"API": None, "tickers": [], "all_tickers": []},
        "BinanceFutures": {"API": None, "tickers": [], "all_tickers": []},
    }
    sigbot.used_tickers = ["ETH", "QTUM", "ONT", "VET", "HOT", "SVM"]
    ticker = "BTCUSDT"
    ttype = "buy"
    sigbot.database = {
        ticker: {
            "5m": {"data": {ttype: pd.DataFrame({"time": [pd.to_datetime("2024-09-24 10:00")]})}},
            "1h": {
                "data": {
                    ttype: pd.DataFrame(
                        {
                            "time": [pd.to_datetime("2024-09-24 07:00")],
                            "macd": [1],
                            "time_4h": [1],
                            "linear_reg": [1],
                            "linear_reg_angle": [1],
                            "macdhist": [1],
                            "macd_dir": [1],
                            "macdsignal": [1],
                            "macdsignal_dir": [1],
                        }
                    )
                }
            },
        }
    }
    return sigbot


@pytest.mark.parametrize(
    "expected",
    [
        expected,
    ],
    ids=repr,
)
def test_filter_used_tickers(sigbot, expected):
    tickers = [
        "ETHUSDT",
        "TRXUSDT",
        "QTUMUSDT",
        "ONTUSDT",
        "VETUSDT",
        "HOTUSDT",
        "SVMUSDT",
        "TTTUSDT",
    ]
    prev_tickers = [148, 265, 322, 228, 309, 300, 1, 10]
    assert sigbot.filter_used_tickers(tickers, prev_tickers) == (
        expected[0],
        expected[1],
    )
    assert sigbot.used_tickers == expected[2]


def test_get_api_and_tickers(sigbot):
    """Test get_api_and_tickers method with mock exchanges"""
    sigbot.get_api_and_tickers()
    assert "Binance" in sigbot.exchanges
    assert "BinanceFutures" in sigbot.exchanges
    assert "tickers" in sigbot.exchanges["Binance"]
    assert "tickers" in sigbot.exchanges["BinanceFutures"]


def test_get_new_data(sigbot):
    """Test get_new_data method with mock exchange API"""
    mock_api = MagicMock()
    mock_api.get_data.return_value = (pd.DataFrame(), 10)
    ticker = "BTCUSDT"
    timeframe = "1h"
    dt_now = datetime.now()

    df, data_qty = sigbot.get_new_data(mock_api, ticker, timeframe, dt_now)

    assert data_qty == 10


def test_filter_old_signals(sigbot):
    """Test filter_old_signals with mock data"""
    dt_now = datetime.now()
    sig_points = [
        (None, None, None, None, dt_now, "MACD_Buy"),
        (None, None, None, None, dt_now - relativedelta(hours=1), "MACD_Buy"),
        (None, None, None, None, dt_now - relativedelta(years=1), "MACD_Buy"),
    ]

    filtered_points = sigbot.filter_old_signals(sig_points)

    assert len(filtered_points) == 2


def test_add_indicators(sigbot):
    df = pd.DataFrame(
        {
            "time": ["2024-09-24 10:00", "2024-09-24 11:00"],
            "open": [100, 105],
            "high": [110, 115],
            "low": [95, 98],
            "close": [108, 110],
        }
    )
    ticker = "BTCUSDT"
    timeframe = "5m"
    exchange_data = {"API": MagicMock()}
    data_qty = 10
    opt_flag = False

    exchange_data["API"].add_indicator_data.return_value = sigbot.database
    database, returned_data_qty = sigbot.add_indicators(
        df, "buy", ticker, timeframe, exchange_data, data_qty, opt_flag
    )

    exchange_data["API"].add_indicator_data.assert_called_once()
    assert returned_data_qty == 37.5


def test_get_buy_signals(sigbot):
    ticker = "BTCUSDT"
    timeframe = "1h"
    sigbot.find_signal_buy.find_signal = MagicMock(return_value=["signal_1"])

    signals = sigbot.get_buy_signals(ticker, timeframe, 10, 5)

    sigbot.find_signal_buy.find_signal.assert_called_once_with(
        sigbot.database, ticker, timeframe, 10, 5
    )
    assert signals == ["signal_1"]


# Test get_sell_signals
def test_get_sell_signals(sigbot):
    ticker = "BTCUSDT"
    timeframe = "1h"
    sigbot.find_signal_sell.find_signal = MagicMock(return_value=["signal_2"])

    signals = sigbot.get_sell_signals(ticker, timeframe, 10, 5)

    sigbot.find_signal_sell.find_signal.assert_called_once_with(
        sigbot.database, ticker, timeframe, 10, 5
    )
    assert signals == ["signal_2"]


# Test sb_add_statistics
def test_sb_add_statistics(sigbot):
    sig_points = ["signal_1", "signal_2"]
    sigbot.stat.write_stat = MagicMock(return_value=sigbot.database)

    database = sigbot.sb_add_statistics(sig_points)

    sigbot.stat.write_stat.assert_called_once_with(sigbot.database, sig_points, None)
    assert database == sigbot.database


# Test calc_statistics
def test_calc_statistics(sigbot):
    sig_points = [[None, None, None, "buy", None, "MACD", None, None, []]]
    sigbot.stat.calculate_total_stat = MagicMock(return_value=(None, None))

    updated_sig_points = sigbot.calc_statistics(sig_points)

    assert len(updated_sig_points[0][8]) == 1


# Test create_exchange_monitors
def test_create_exchange_monitors(sigbot):
    spot_ex, fut_ex = sigbot.create_exchange_monitors()

    assert len(fut_ex) == 1
    assert len(spot_ex) == 1


# Test add_higher_time
def test_add_higher_time(sigbot):
    ticker = "BTCUSDT"
    ttype = "buy"

    sigbot.add_higher_time(ticker, ttype)

    assert "macd" in sigbot.database[ticker]["5m"]["data"][ttype].columns
    assert "linear_reg_angle" in sigbot.database[ticker]["5m"]["data"][ttype].columns


# Test make_prediction
def test_make_prediction(sigbot):
    sig_points = [["BTCUSDT", "1h", 0, "buy", None, "STOCH_RSI", None, [], None, []]]
    sigbot.model.make_prediction = MagicMock(return_value=sig_points)

    updated_points = sigbot.make_prediction(sig_points, "ByBitPerpetual")

    sigbot.model.make_prediction.assert_called_once()
    assert updated_points == sig_points


# Test delete_redundant_symbols_from_ticker
def test_delete_redundant_symbols_from_ticker(sigbot):
    ticker = "BTC-USDT-SWAP"
    cleaned_ticker = sigbot.delete_redundant_symbols_from_ticker(ticker)

    assert cleaned_ticker == "BTCUSDT"
