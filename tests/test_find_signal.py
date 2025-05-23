from os import environ

import numpy as np
import pandas as pd
import pytest

environ["ENV"] = "test"

from config.config import ConfigFactory

# from bot.bot import SigBot
from data.get_data import DataFactory
from indicators.indicators import IndicatorFactory
from signals.find_signal import FindSignal, SignalFactory

# Set dataframe dict
df_btc_1h = pd.read_pickle("test_BTCUSDT_1h.pkl")
df_btc_5m = pd.read_pickle("test_BTCUSDT_5m.pkl")
df_eth_1h = pd.read_pickle("test_ETHUSDT_1h.pkl")
df_eth_5m = pd.read_pickle("test_ETHUSDT_5m.pkl")

# Get configs
configs = ConfigFactory.factory(environ).configs


def create_test_data(data_qty=20):
    dfs = {
        "stat": {
            "buy": pd.DataFrame(columns=["time", "ticker", "timeframe"]),
            "sell": pd.DataFrame(columns=["time", "ticker", "timeframe"]),
        },
        "BTCUSDT": {
            "1h": {"data": {"buy": df_btc_1h, "sell": df_btc_1h}},
            "5m": {"data": {"buy": df_btc_5m, "sell": df_btc_5m}},
        },
        "ETHUSDT": {
            "1h": {"data": {"buy": df_eth_1h, "sell": df_eth_1h}},
            "5m": {"data": {"buy": df_eth_5m, "sell": df_eth_5m}},
        },
    }

    # Create exchange API
    exchange_api = DataFactory.factory("Binance", **configs)

    # Higher timeframe from which we take levels
    work_timeframe = configs["Timeframes"]["work_timeframe"]

    # For every exchange, ticker and timeframe in base get cryptocurrency data and write it to correspond dataframe
    for ticker in ["BTCUSDT", "ETHUSDT"]:
        for timeframe in ["1h", "5m"]:
            indicators = list()
            if timeframe == work_timeframe:
                indicator_list = configs["Indicator_list"]
            else:
                indicator_list = ["Trend"]
            for indicator in indicator_list:
                ind_factory = IndicatorFactory.factory(indicator, "buy", configs)
                if ind_factory:
                    indicators.append(ind_factory)
            # Write indicators to dataframe, update dataframe dict
            dfs = exchange_api.add_indicator_data(
                dfs,
                dfs[ticker][timeframe]["data"]["buy"],
                "buy",
                indicators,
                ticker,
                timeframe,
                data_qty,
            )
    return dfs


btc_expected = [np.array([18, 19, 20, 21, 25, 26, 27, 45, 46, 47, 48])]
eth_expected = [np.array([18, 19])]


@pytest.mark.parametrize(
    "ticker, timeframe, high_bound, expected",
    [
        ("BTCUSDT", "5m", 100, []),
        ("BTCUSDT", "5m", 60, btc_expected[0]),
        ("ETHUSDT", "5m", 75, eth_expected[0]),
    ],
    ids=repr,
)
def test_higher_bound(mocker, timeframe, ticker, high_bound, expected):
    mocker.patch("api.binance_api.Binance.connect_to_api", return_value=None)
    # mocker.patch("log.log.create_logger")
    dfs = create_test_data()
    stoch_sig = SignalFactory().factory("STOCH", "sell", configs)

    df = dfs[ticker][timeframe]["data"]["buy"][:50]
    stoch_slowd = df["stoch_slowd"]
    stoch_slowd_lag_1 = df["stoch_slowd"].shift(1)
    stoch_slowd_lag_2 = df["stoch_slowd"].shift(2)

    points = stoch_sig.higher_bound(high_bound, stoch_slowd, stoch_slowd_lag_1, stoch_slowd_lag_2)
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [np.array([30, 31, 32])]
eth_expected = [
    np.array(
        [
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        ]
    )
]


@pytest.mark.parametrize(
    "ticker, timeframe, low_bound, expected",
    [
        ("BTCUSDT", "5m", 0, []),
        ("BTCUSDT", "5m", 15, btc_expected[0]),
        ("ETHUSDT", "5m", 30, eth_expected[0]),
    ],
    ids=repr,
)
def test_lower_bound(mocker, timeframe, ticker, low_bound, expected):
    mocker.patch("api.binance_api.Binance.connect_to_api", return_value=None)
    dfs = create_test_data()
    stoch_sig = SignalFactory().factory("STOCH", "sell", configs)

    df = dfs[ticker][timeframe]["data"]["buy"][:50]
    stoch_slowk = df["stoch_slowk"]
    stoch_slowk_lag_1 = df["stoch_slowk"].shift(1)
    stoch_slowk_lag_2 = df["stoch_slowk"].shift(2)

    points = stoch_sig.lower_bound(low_bound, stoch_slowk, stoch_slowk_lag_1, stoch_slowk_lag_2)
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [
    np.array([23, 24, 25, 26, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46])
]
eth_expected = [np.array([24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 46, 47, 48, 49])]


@pytest.mark.parametrize(
    "ticker, timeframe, expected",
    [
        ("BTCUSDT", "5m", btc_expected[0]),
        ("ETHUSDT", "5m", eth_expected[0]),
    ],
    ids=repr,
)
def test_up_direction(mocker, timeframe, ticker, expected):
    mocker.patch("api.binance_api.Binance.connect_to_api", return_value=None)
    dfs = create_test_data()
    stoch_sig = SignalFactory().factory("STOCH", "sell", configs)

    df = dfs[ticker][timeframe]["data"]["buy"][:50]

    points = stoch_sig.up_direction(df["stoch_slowk_dir"])
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [np.array([20, 21, 22, 27, 28, 29, 30, 37, 47, 48, 49])]
eth_expected = [np.array([20, 21, 22, 23, 27, 28, 29, 40, 41, 42, 43, 44, 45])]


@pytest.mark.parametrize(
    "ticker, timeframe, expected",
    [
        ("BTCUSDT", "5m", btc_expected[0]),
        ("ETHUSDT", "5m", eth_expected[0]),
    ],
    ids=repr,
)
def test_down_direction(mocker, timeframe, ticker, expected):
    mocker.patch("api.binance_api.Binance.connect_to_api", return_value=None)
    dfs = create_test_data()
    stoch_sig = SignalFactory().factory("STOCH", "sell", configs)

    df = dfs[ticker][timeframe]["data"]["buy"][:50]

    points = stoch_sig.down_direction(df["stoch_slowk_dir"])
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [
    np.array([27, 28, 37, 38, 47, 48, 65, 66, 79, 90, 91]),
    np.array([24, 25, 32, 33, 39, 40, 55, 56, 76, 77, 80, 81, 99]),
]
eth_expected = [
    np.array([27, 28, 40, 41, 57, 58, 70, 71, 82, 83, 97, 98]),
    np.array([25, 26, 31, 32, 47, 48, 67, 68, 72, 73, 91, 92, 99]),
]


@pytest.mark.parametrize(
    "ticker, timeframe, up, expected",
    [
        ("BTCUSDT", "5m", True, btc_expected[0]),
        ("BTCUSDT", "5m", False, btc_expected[1]),
        ("ETHUSDT", "5m", True, eth_expected[0]),
        ("ETHUSDT", "5m", False, eth_expected[1]),
    ],
    ids=repr,
)
def test_crossed_lines(mocker, timeframe, ticker, up, expected):
    mocker.patch("api.binance_api.Binance.connect_to_api", return_value=None)
    dfs = create_test_data()
    stoch_sig = SignalFactory().factory("STOCH", "sell", configs)

    df = dfs[ticker][timeframe]["data"]["buy"][:100]
    stoch_diff = df["stoch_diff"]
    stoch_diff_lag_1 = df["stoch_diff"].shift(1)
    stoch_diff_lag_2 = df["stoch_diff"].shift(2)

    points = stoch_sig.crossed_lines(up, stoch_diff, stoch_diff_lag_1, stoch_diff_lag_2)
    indexes = np.where(points == 1)
    assert np.array_equal(indexes[0], expected)


btc_expected = [
    np.array([91, 343, 447, 569, 655, 666]),
    np.array([282, 381, 382, 506, 636, 643, 723, 725, 826, 865]),
]
eth_expected = [
    np.array([83, 547, 560, 657, 658, 916, 996]),
    np.array(
        [
            26,
            31,
            68,
            72,
            146,
            153,
            322,
            370,
            374,
            590,
            612,
            629,
            631,
            632,
            857,
            868,
            886,
        ]
    ),
]


@pytest.mark.parametrize(
    "ticker, timeframe, expected",
    [("BTCUSDT", "5m", btc_expected), ("ETHUSDT", "5m", eth_expected)],
    ids=repr,
)
def test_find_stoch_signal(mocker, timeframe, ticker, expected):
    mocker.patch("api.binance_api.Binance.connect_to_api", return_value=None)
    dfs = create_test_data()
    stoch_sig_buy = SignalFactory().factory("STOCH", "buy", configs)
    stoch_sig_sell = SignalFactory().factory("STOCH", "sell", configs)
    if ticker == "BTCUSDT":
        dfs[ticker][timeframe]["data"]["buy"].loc[447, "stoch_diff"] *= -1
        dfs[ticker][timeframe]["data"]["buy"].loc[446, "stoch_slowk"] += 3
        dfs[ticker][timeframe]["data"]["buy"].loc[446, "stoch_slowd"] += 3
        dfs[ticker][timeframe]["data"]["buy"].loc[447, "stoch_slowk_dir"] *= -1
        dfs[ticker][timeframe]["data"]["buy"].loc[447, "stoch_slowd_dir"] *= -1
    elif ticker == "ETHUSDT":
        dfs[ticker][timeframe]["data"]["buy"].loc[145, "stoch_slowd"] -= 1
        dfs[ticker][timeframe]["data"]["buy"].loc[145, "stoch_slowk_dir"] *= -1
        dfs[ticker][timeframe]["data"]["buy"].loc[146, "stoch_slowk_dir"] *= -1
        dfs[ticker][timeframe]["data"]["buy"].loc[146, "stoch_slowd_dir"] *= -1
        dfs[ticker][timeframe]["data"]["buy"].loc[146, "stoch_diff"] *= -1
    df = dfs[ticker][timeframe]["data"]["buy"]
    buy_points = stoch_sig_buy.find_signal(df)
    sell_points = stoch_sig_sell.find_signal(df)
    buy_indexes = np.where(buy_points == 1)
    sell_indexes = np.where(sell_points == 1)
    assert np.array_equal(buy_indexes[0], expected[1])
    assert np.array_equal(sell_indexes[0], expected[0])


btc_expected = [np.array([25, 27, 46, 59, 71, 73, 78, 91, 93, 96]), np.array([24, 89])]
eth_expected = [
    np.array([3, 10, 17, 25, 38, 40, 44, 63, 65, 70, 83, 85, 88, 95]),
    np.array([2, 16, 71]),
]


@pytest.mark.parametrize(
    "ticker, timeframe, expected",
    [("BTCUSDT", "5m", btc_expected), ("ETHUSDT", "5m", eth_expected)],
    ids=repr,
)
def test_find_price_change_signal(mocker, ticker, timeframe, expected):
    mocker.patch("api.binance_api.Binance.connect_to_api", return_value=None)
    dfs = create_test_data()
    price_change_sig_buy = SignalFactory().factory("PumpDump", "buy", configs)
    price_change_sig_sell = SignalFactory().factory("PumpDump", "sell", configs)
    buy_points = price_change_sig_buy.find_signal(dfs[ticker][timeframe]["data"]["buy"][:100])
    sell_points = price_change_sig_sell.find_signal(dfs[ticker][timeframe]["data"]["buy"][:100])
    buy_indexes = np.where(buy_points == 1)
    sell_indexes = np.where(sell_points == 1)
    assert np.array_equal(buy_indexes[0], expected[0])
    assert np.array_equal(sell_indexes[0], expected[1])


@pytest.mark.parametrize(
    "timeframe, ticker, multiplier, expected",
    [
        # ('5m', 'BTCUSDT', 1, 0),
        ("5m", "BTCUSDT", 10, 100),
        # ('5m', 'ETHUSDT', 1, 100),
        # ('5m', 'ETHUSDT', 0.0001, 0),
    ],
    ids=repr,
)
def test_filter_by_volume_24(mocker, ticker, timeframe, multiplier, expected):
    mocker.patch("api.binance_api.Binance.connect_to_api", return_value=None)
    dfs = create_test_data()

    price_volume_sig_buy = SignalFactory().factory("Volume24", "buy", configs)
    price_volume_sig_sell = SignalFactory().factory("Volume24", "sell", configs)
    buy_df = dfs[ticker][timeframe]["data"]["buy"][-100:].copy()
    buy_df["volume_24"] = buy_df["volume_24"] * multiplier
    buy_points = price_volume_sig_buy.find_signal(buy_df)
    sell_df = dfs[ticker][timeframe]["data"]["sell"][-100:].copy()
    sell_df["volume_24"] = sell_df["volume_24"] * multiplier
    sell_points = price_volume_sig_sell.find_signal(sell_df)
    assert np.sum(buy_points) == expected
    assert np.sum(sell_points) == expected


btc_lr_buy_expected_1 = np.load("test_btc_buy_indexes_1.npy")
btc_lr_sell_expected_1 = np.load("test_btc_sell_indexes_1.npy")
btc_lr_buy_expected_2 = np.load("test_btc_buy_indexes_2.npy")
btc_lr_sell_expected_2 = np.load("test_btc_sell_indexes_2.npy")
eth_lr_buy_expected_1 = np.load("test_eth_buy_indexes_1.npy")
eth_lr_sell_expected_1 = np.load("test_eth_sell_indexes_1.npy")
eth_lr_buy_expected_2 = np.load("test_eth_buy_indexes_2.npy")
eth_lr_sell_expected_2 = np.load("test_eth_sell_indexes_2.npy")


@pytest.mark.parametrize(
    "ticker, offset1, offset2, expected",
    [
        ("BTCUSDT", 0, 0, (btc_lr_buy_expected_1, btc_lr_sell_expected_1)),
        ("BTCUSDT", 80, 889, (btc_lr_buy_expected_2, btc_lr_sell_expected_2)),
        ("ETHUSDT", 0, 0, (eth_lr_buy_expected_1, eth_lr_sell_expected_1)),
        ("ETHUSDT", 70, 800, (eth_lr_buy_expected_2, eth_lr_sell_expected_2)),
    ],
    ids=repr,
)
def test_find_linear_reg_signal(mocker, ticker, offset1, offset2, expected):
    if ticker == "BTCUSDT":
        df_higher = pd.read_pickle("test_BTCUSDT_1h.pkl")[-offset1:].reset_index(drop=True)
        df_working = pd.read_pickle("test_BTCUSDT_5m.pkl")[-offset2:].reset_index(drop=True)
    else:
        df_higher = pd.read_pickle("test_ETHUSDT_1h.pkl")[-offset1:].reset_index(drop=True)
        df_working = pd.read_pickle("test_ETHUSDT_5m.pkl")[-offset2:].reset_index(drop=True)

    mocker.patch("api.binance_api.Binance.connect_to_api", return_value=None)
    linear_reg_sig_buy = SignalFactory().factory("Trend", "buy", configs)
    linear_reg_sig_sell = SignalFactory().factory("Trend", "sell", configs)
    # prepare trade points
    df_higher["time_higher"] = df_higher["time"]
    higher_features = [
        "time",
        "time_higher",
        "linear_reg",
        "linear_reg_angle",
        "macd",
        "macdhist",
        "macd_dir",
        "macdsignal",
        "macdsignal_dir",
    ]
    df_working[higher_features] = pd.merge(
        df_working[["time"]], df_higher[higher_features], how="left", on="time"
    )
    higher_features.remove("time")
    for f in higher_features:
        df_working[f] = df_working[f].ffill()
        df_working[f] = df_working[f].bfill()
    # test find_signal function
    buy_points = linear_reg_sig_buy.find_signal(df_working)
    sell_points = linear_reg_sig_sell.find_signal(df_working)
    buy_indexes = np.where(buy_points == 1)
    sell_indexes = np.where(sell_points == 1)
    assert np.array_equal(buy_indexes, expected[0])
    assert np.array_equal(sell_indexes, expected[1])


point_b1 = [
    [
        "BTCUSDT",
        "5m",
        506,
        "buy",
        pd.Timestamp("2022-08-22 21:55:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        0,
    ],
]
point_b2 = [
    [
        "BTCUSDT",
        "5m",
        506,
        "buy",
        pd.Timestamp("2022-08-22 21:55:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        0,
    ],
]
point_b3 = [
    [
        "ETHUSDT",
        "5m",
        370,
        "buy",
        pd.Timestamp("2022-08-22 11:15:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        0,
    ],
    [
        "ETHUSDT",
        "5m",
        629,
        "buy",
        pd.Timestamp("2022-08-23 08:50:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        0,
    ],
    [
        "ETHUSDT",
        "5m",
        631,
        "buy",
        pd.Timestamp("2022-08-23 09:00:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        0,
    ],
    [
        "ETHUSDT",
        "5m",
        172,
        "buy",
        pd.Timestamp("2022-08-21 18:45:00"),
        "Pattern_Trend",
        [],
        [],
        [],
        0,
    ],
    [
        "ETHUSDT",
        "5m",
        557,
        "buy",
        pd.Timestamp("2022-08-23 02:50:00"),
        "Pattern_Trend",
        [],
        [],
        [],
        0,
    ],
]
point_b4 = [
    [
        "ETHUSDT",
        "5m",
        629,
        "buy",
        pd.Timestamp("2022-08-23 08:50:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        0,
    ],
    [
        "ETHUSDT",
        "5m",
        631,
        "buy",
        pd.Timestamp("2022-08-23 09:00:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        0,
    ],
    [
        "ETHUSDT",
        "5m",
        557,
        "buy",
        pd.Timestamp("2022-08-23 02:50:00"),
        "Pattern_Trend",
        [],
        [],
        [],
        0,
    ],
]
expected_buy = [point_b1, point_b2, point_b3, point_b4]


point_s1 = [
    [
        "BTCUSDT",
        "5m",
        91,
        "sell",
        pd.Timestamp("2022-08-21 11:20:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        0,
    ],
    [
        "BTCUSDT",
        "5m",
        569,
        "sell",
        pd.Timestamp("2022-08-23 03:10:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        0,
    ],
]
point_s2 = [
    [
        "BTCUSDT",
        "5m",
        569,
        "sell",
        pd.Timestamp("2022-08-23 03:10:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        0,
    ]
]

point_s3 = [
    [
        "ETHUSDT",
        "5m",
        83,
        "sell",
        pd.Timestamp("2022-08-21 11:20:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        0,
    ],
    [
        "ETHUSDT",
        "5m",
        362,
        "sell",
        pd.Timestamp("2022-08-22 10:35:00"),
        "Pattern_Trend",
        [],
        [],
        [],
        0,
    ],
]
point_s4 = []
expected_sell = [point_s1, point_s2, point_s3, point_s4]


@pytest.mark.parametrize(
    "ticker, ttype, timeframe, limit, expected",
    [
        ("BTCUSDT", "sell", "5m", 1000, expected_sell[0]),
        ("BTCUSDT", "sell", "5m", 500, expected_sell[1]),
        ("BTCUSDT", "sell", "5m", 10, []),
        ("ETHUSDT", "sell", "5m", 1000, expected_sell[2]),
        ("ETHUSDT", "sell", "5m", 400, expected_sell[3]),
        ("ETHUSDT", "sell", "5m", 10, []),
        ("BTCUSDT", "buy", "5m", 1000, expected_buy[0]),
        ("BTCUSDT", "buy", "5m", 500, expected_buy[1]),
        ("ETHUSDT", "buy", "5m", 1000, expected_buy[2]),
        ("ETHUSDT", "buy", "5m", 500, expected_buy[3]),
        ("BTCUSDT", "buy", "5m", 10, []),
        ("ETHUSDT", "buy", "5m", 10, []),
    ],
    ids=repr,
)
def test_find_signal(mocker, ttype, ticker, timeframe, limit, expected):
    mocker.patch("api.binance_api.Binance.connect_to_api", return_value=None)
    dfs = create_test_data(1000)
    fs_buy = FindSignal("buy", configs)
    fs_sell = FindSignal("sell", configs)
    fs_buy.patterns = [
        ["STOCH", "RSI"],
        ["STOCH", "RSI", "Trend"],
        ["Pattern", "Trend"],
    ]
    fs_sell.patterns = [
        ["STOCH", "RSI"],
        ["STOCH", "RSI", "Trend"],
        ["Pattern", "Trend"],
    ]
    if ttype == "buy":
        res = fs_buy.find_signal(dfs, ticker, timeframe, limit, limit)
        assert res == expected
    else:
        res = fs_sell.find_signal(dfs, ticker, timeframe, limit, limit)
        assert res == expected
