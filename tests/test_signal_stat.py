from os import environ

import pandas as pd
import pytest

from config.config import ConfigFactory
from signal_stat.signal_stat import SignalStat

# Set environment variable
environ["ENV"] = "test"

configs = ConfigFactory.factory(environ).configs

df_btc = pd.read_pickle("test_BTCUSDT_5m.pkl")
df_eth = pd.read_pickle("test_ETHUSDT_5m.pkl")
points_btc = [
    (
        "BTCUSDT",
        "5m",
        0,
        "buy",
        pd.to_datetime("2022-08-21 3:45:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "BTCUSDT",
        "5m",
        20,
        "sell",
        pd.to_datetime("2022-08-21 8:25:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "BTCUSDT",
        "5m",
        98,
        "sell",
        pd.to_datetime("2022-08-21 11:55:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "BTCUSDT",
        "5m",
        513,
        "buy",
        pd.to_datetime("2022-08-22 22:30:00"),
        "STOCH_RSI_Trend",
        [],
        [],
        [],
        [],
    ),
    (
        "BTCUSDT",
        "5m",
        576,
        "sell",
        pd.to_datetime("2022-08-23 03:45:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "BTCUSDT",
        "5m",
        673,
        "sell",
        pd.to_datetime("2022-08-23 11:50:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "BTCUSDT",
        "5m",
        745,
        "buy",
        pd.to_datetime("2022-08-23 17:50:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "BTCUSDT",
        "5m",
        977,
        "sell",
        pd.to_datetime("2022-08-24 12:05:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "BTCUSDT",
        "5m",
        995,
        "buy",
        pd.to_datetime("2022-08-24 14:40:00"),
        "STOCH_RSI_Trend",
        [],
        [],
        [],
        [],
    ),
]

points_eth = [
    (
        "ETHUSDT",
        "5m",
        45,
        "sell",
        pd.to_datetime("2022-08-21 09:00:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "ETHUSDT",
        "5m",
        47,
        "buy",
        pd.to_datetime("2022-08-21 09:10:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "ETHUSDT",
        "5m",
        97,
        "sell",
        pd.to_datetime("2022-08-21 12:30:00"),
        "STOCH_RSI_Trend",
        [],
        [],
        [],
        [],
    ),
    (
        "ETHUSDT",
        "5m",
        512,
        "buy",
        pd.to_datetime("2022-08-22 23:05:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "ETHUSDT",
        "5m",
        645,
        "buy",
        pd.to_datetime("2022-08-23 10:10:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "ETHUSDT",
        "5m",
        840,
        "sell",
        pd.to_datetime("2022-08-24 02:25:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "ETHUSDT",
        "5m",
        985,
        "sell",
        pd.to_datetime("2022-08-24 12:00:00"),
        "STOCH_RSI",
        [],
        [],
        [],
        [],
    ),
    (
        "ETHUSDT",
        "5m",
        990,
        "buy",
        pd.to_datetime("2022-08-24 12:25:00"),
        "STOCH_RSI_Trend",
        [],
        [],
        [],
        [],
    ),
]


buy_btc = pd.read_pickle("signal_stat/btc_buy_stat.pkl")
sell_btc = pd.read_pickle("signal_stat/btc_sell_stat.pkl")
buy_eth = pd.read_pickle("signal_stat/eth_buy_stat.pkl")
sell_eth = pd.read_pickle("signal_stat/eth_sell_stat.pkl")
expected_write_stat = [
    {"buy": buy_btc, "sell": sell_btc},
    {"buy": buy_eth, "sell": sell_eth},
]


@pytest.mark.parametrize(
    "signal_points, expected",
    [(points_btc, expected_write_stat[0]), (points_eth, expected_write_stat[1])],
    ids=repr,
)
def test_write_stat(signal_points, expected):
    ss = SignalStat(**configs)
    dfs = {
        "stat": {
            "buy": pd.DataFrame(columns=["time", "ticker", "timeframe", "pattern"]),
            "sell": pd.DataFrame(columns=["time", "ticker", "timeframe", "pattern"]),
        },
        "BTCUSDT": {"5m": {"data": {"buy": df_btc, "sell": df_btc}, "levels": []}},
        "ETHUSDT": {"5m": {"data": {"buy": df_eth, "sell": df_eth}, "levels": []}},
    }

    result = ss.write_stat(dfs, signal_points)
    assert result["stat"]["buy"].equals(expected["buy"])
    assert result["stat"]["sell"].equals(expected["sell"])


total_buy = pd.concat([buy_btc, buy_eth])
total_sell = pd.concat([sell_btc, sell_eth])
total_buy["pattern"] = "STOCH_RSI"
total_sell["pattern"] = "STOCH_RSI"

expected1 = (
    [
        (1.1383, 0.02, 0.28),
        (0.9676, 0.05, 0.27),
        (1.1553, 0.01, 0.16),
        (1.1653, 0.07, 0.11),
        (1.7068, 0.17, 0.15),
        (2.1309, 0.28, 0.3),
        (2.2194, 0.26, 0.35),
        (2.2194, 0.16, 0.36),
        (2.3336, 0.28, 0.54),
        (1.7045, 0.3, 0.5),
        (2.0737, 0.42, 0.65),
        (2.1815, 0.4, 0.8),
        (2.1815, 0.37, 0.72),
        (2.1815, 0.32, 0.71),
        (2.1815, 0.22, 0.62),
        (2.1815, 0.24, 0.87),
        (1.9476, 0.15, 0.96),
        (1.8268, 0.08, 0.87),
        (1.8091, 0.07, 0.87),
        (1.7959, 0.1, 0.95),
        (1.7959, 0.18, 0.82),
        (1.7959, 0.26, 0.84),
        (1.7959, 0.25, 0.81),
        (1.7959, 0.31, 0.68),
    ],
    6,
)

expected2 = (
    [
        (1.3995, -0.01, 0.14),
        (1.2466, 0.05, 0.17),
        (0.6562, 0.09, 0.28),
        (0.6626, 0.1, 0.23),
        (0.8611, -0.01, 0.38),
        (1.0893, -0.01, 0.28),
        (1.1227, -0.11, 0.29),
        (1.1227, -0.04, 0.37),
        (1.1227, 0.0, 0.42),
        (1.173, -0.05, 0.62),
        (1.0637, 0.05, 0.49),
        (1.1651, -0.05, 0.6),
        (1.1298, -0.05, 0.52),
        (1.4099, -0.24, 0.78),
        (1.436, -0.28, 0.84),
        (1.3771, -0.17, 0.71),
        (1.4322, -0.21, 0.66),
        (1.4322, -0.24, 0.74),
        (1.4322, -0.29, 0.75),
        (1.4322, -0.28, 0.64),
        (1.4996, -0.35, 0.63),
        (1.6144, -0.36, 0.66),
        (1.7587, -0.48, 0.92),
        (1.7619, -0.31, 0.94),
    ],
    6,
)


@pytest.mark.parametrize(
    "ttype, pattern, expected",
    [
        ("buy", "STOCH_RSI_Trend", ([(0, 0, 0) for _ in range(24)], 0)),
        ("buy", "STOCH_RSI", expected1),
        ("sell", "STOCH_RSI", expected2),
    ],
    ids=repr,
)
def test_calculate_total_stat(ttype, pattern, expected):
    ss = SignalStat(**configs)
    dfs = {"stat": {"buy": total_buy, "sell": total_sell}}
    result = ss.calculate_total_stat(dfs, ttype, pattern)
    assert result == expected


get_result_price_after_period_1 = [
    [
        21269.31,
        21280.5,
        21268.08,
        21300.0,
        21299.03,
        21296.85,
        21290.94,
        21309.82,
        21364.07,
        21373.32,
        21444.72,
        21420.0,
        21548.71,
        21438.98,
        21390.38,
        21387.68,
        21406.93,
        21379.96,
        21367.05,
        21371.89,
        21369.1,
        21350.14,
        21361.22,
        21343.7,
    ],
    [
        21193.12,
        21222.02,
        21211.22,
        21253.0,
        21270.5,
        21251.49,
        21239.08,
        21255.87,
        21275.71,
        21314.63,
        21340.05,
        21381.65,
        21398.59,
        21352.22,
        21337.38,
        21340.93,
        21354.97,
        21323.08,
        21325.13,
        21334.9,
        21330.1,
        21315.12,
        21318.0,
        21316.22,
    ],
    [
        21263.08,
        21253.28,
        21256.95,
        21281.0,
        21293.67,
        21252.9,
        21283.87,
        21277.82,
        21329.56,
        21340.16,
        21407.62,
        21400.47,
        21426.86,
        21368.32,
        21345.76,
        21359.89,
        21376.86,
        21344.57,
        21336.13,
        21365.37,
        21334.55,
        21340.79,
        21337.0,
        21331.57,
    ],
]
get_result_price_after_period_2 = [
    [
        1614.0,
        1608.87,
        1609.8,
        1617.46,
        1615.63,
        1616.0,
        1614.85,
        1620.94,
        1621.52,
        1618.72,
        1618.08,
        1618.14,
        1619.3,
        1618.0,
        1619.17,
        1623.56,
        1622.23,
        1619.82,
        1617.31,
        1618.47,
        1621.3,
        1620.44,
        1620.49,
        1621.82,
    ],
    [
        1607.42,
        1600.4,
        1605.29,
        1606.65,
        1610.5,
        1611.4,
        1610.01,
        1611.11,
        1617.09,
        1616.2,
        1613.8,
        1615.2,
        1616.1,
        1615.88,
        1617.27,
        1618.23,
        1619.13,
        1611.28,
        1614.18,
        1615.19,
        1617.82,
        1616.32,
        1617.53,
        1617.47,
    ],
    [
        1608.28,
        1606.11,
        1608.16,
        1613.27,
        1614.69,
        1612.38,
        1611.97,
        1620.76,
        1617.63,
        1618.07,
        1615.44,
        1617.43,
        1616.37,
        1617.66,
        1618.6,
        1619.62,
        1619.17,
        1614.4,
        1615.67,
        1618.28,
        1620.44,
        1618.34,
        1617.54,
        1621.12,
    ],
]


@pytest.mark.parametrize(
    "df, point, signal_price, signal_smooth_price, expected",
    [
        (
            df_btc,
            [
                "BTCUSDT",
                "5m",
                458,
                "buy",
                pd.Timestamp("2022-08-22 17:55:00"),
                "STOCH_RSI",
                "",
                "",
                [],
                [],
            ],
            21201,
            21232,
            get_result_price_after_period_1,
        ),
        (
            df_eth,
            [
                "ETHUSDT",
                "5m",
                150,
                "buy",
                pd.Timestamp("2022-08-22 22:30:00"),
                "Pattern",
                "",
                "",
                [],
                [],
            ],
            1500,
            1510,
            get_result_price_after_period_2,
        ),
    ],
    ids=repr,
)
def test_get_result_price_after_period(df, point, signal_price, signal_smooth_price, expected):
    (
        ticker,
        timeframe,
        index,
        ttype,
        time,
        pattern,
        plot_path,
        exchange_list,
        total_stat,
        ticker_stat,
    ) = point
    ss = SignalStat(**configs)
    (
        high_result_prices,
        low_result_prices,
        close_smooth_prices,
        atr,
    ) = ss.get_result_price_after_period(df, index)
    assert expected[0] == high_result_prices
    assert expected[1] == low_result_prices
    assert expected[2] == close_smooth_prices


process_stat_1 = pd.read_pickle("test_btc_process_stat.pkl")
process_stat_2 = pd.read_pickle("test_eth_process_stat.pkl")


@pytest.mark.parametrize(
    "df, point, signal_price, signal_smooth_price, expected",
    [
        (
            df_btc,
            [
                "BTCUSDT",
                "5m",
                458,
                "buy",
                pd.Timestamp("2022-08-22 17:55:00"),
                "STOCH_RSI",
                "",
                "",
                [],
                [],
            ],
            21201,
            21232,
            process_stat_1,
        ),
        (
            df_eth,
            [
                "ETHUSDT",
                "5m",
                150,
                "buy",
                pd.Timestamp("2022-08-22 22:30:00"),
                "Pattern",
                "",
                "",
                [],
                [],
            ],
            1500,
            1510,
            process_stat_2,
        ),
    ],
    ids=repr,
)
def test_process_statistics(df, point, signal_price, signal_smooth_price, expected):
    (
        ticker,
        timeframe,
        index,
        ttype,
        time,
        pattern,
        plot_path,
        exchange_list,
        total_stat,
        ticker_stat,
    ) = point
    ss = SignalStat(**configs)
    dfs = {
        "stat": {
            "buy": pd.DataFrame(columns=["time", "ticker", "timeframe", "pattern"]),
            "sell": pd.DataFrame(columns=["time", "ticker", "timeframe", "pattern"]),
        }
    }
    (
        high_result_prices,
        low_result_prices,
        close_smooth_prices,
        atr,
    ) = ss.get_result_price_after_period(df, index)
    result = ss.process_statistics(
        dfs,
        point,
        signal_price,
        signal_smooth_price,
        high_result_prices,
        low_result_prices,
        close_smooth_prices,
        atr,
    )
    assert result["stat"][ttype].equals(expected)


@pytest.mark.parametrize(
    "ticker, index, point_time, pattern, prev_point, expected",
    [
        (
            "BTCUSDT",
            90,
            pd.Timestamp("2022-08-22 22:30:00"),
            "STOCH_RSI",
            (None, None, None),
            True,
        ),
        (
            "BTCUSDT",
            50,
            pd.Timestamp("2022-08-22 22:30:00"),
            "STOCH_RSI",
            (None, None, None),
            False,
        ),
        (
            "BTCUSDT",
            90,
            pd.Timestamp("2022-08-22 21:30:00"),
            "STOCH_RSI",
            ("BTCUSDT", pd.Timestamp("2022-08-22 21:00:00"), "STOCH_RSI"),
            False,
        ),
        (
            "BTCUSDT",
            90,
            pd.Timestamp("2022-08-22 21:30:00"),
            "STOCH_RSI",
            ("BTCUSDT", pd.Timestamp("2022-08-22 20:00:00"), "STOCH_RSI"),
            False,
        ),
        (
            "BTCUSDT",
            90,
            pd.Timestamp("2022-08-24 14:50:00"),
            "STOCH_RSI",
            (None, None, None),
            False,
        ),
        (
            "BTCUSDT",
            90,
            pd.Timestamp("2022-08-24 15:00:00"),
            "STOCH_RSI",
            (None, None, None),
            True,
        ),
        (
            "BTCUSDT",
            90,
            pd.Timestamp("2022-08-24 15:00:00"),
            "STOCH_RSI",
            ("BTCUSDT", pd.Timestamp("2022-08-24 14:55:00"), "STOCH_RSI"),
            False,
        ),
        (
            "BTCUSDT",
            90,
            pd.Timestamp("2022-08-24 15:00:00"),
            "STOCH_RSI",
            ("BTCUSDT", pd.Timestamp("2022-08-24 14:40:00"), "STOCH_RSI"),
            True,
        ),
    ],
    ids=repr,
)
def test_check_close_trades(ticker, index, point_time, pattern, prev_point, expected):
    ss = SignalStat(**configs)
    result = ss.check_close_trades(total_buy, 100, ticker, index, point_time, pattern, prev_point)
    assert result == expected
