"""
This module provides functionality for interacting with the Binance Futures
cryptocurrency exchange API. It defines the `ByBitPerpetual` class, which
extends the `ApiBase` class to retrieve and manipulate market data such as
ticker symbols, K-line (candlestick) data, and historical data for specified
intervals.
"""

from datetime import datetime
from math import ceil, log10
from os import environ
from time import sleep
from typing import Dict, List, Tuple, Union

import pandas as pd
import pybit
from pybit import unified_trading
from pybit.exceptions import InvalidRequestError

from api.api_base import ApiBase
from config.config import ConfigFactory
from loguru import logger

# Set environment variable
configs = ConfigFactory.factory(environ).configs


class ByBitPerpetual(ApiBase):
    """Class for accessing ByBitPerpetual cryptocurrency exchange API"""

    client: unified_trading.HTTP = ""

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize the ByBit API connection and retrieve trading settings
        from the configuration.

        Parameters
        ----------
        api_key : str, optional
            API key for the ByBit account (default is "Key").
        api_secret : str, optional
            API secret for the ByBit account (default is "Secret").
        """
        self.global_limit = 1000
        self.api_key = api_key
        self.api_secret = api_secret
        self.connect_to_api(self.api_key, self.api_secret)
        self.symbols_info: Dict[str, dict] = {}
        self.query_symbols_info()
        self.leverage = configs["Trade"]["leverage"]
        self.risk = configs["Trade"]["risk"]
        self.TP = configs["Trade"]["TP"]
        self.SL = configs["Trade"]["SL"]
        self.quote_coin = configs["Trade"]["quote_coin"]
        self.currency = configs["Trade"]["currency"]
        self.one_way_mode = configs["Trade"]["one_way_mode"]
        self.is_isolated = configs["Trade"]["is_isolated"]
        self.order_timeout_hours = configs["Trade"]["order_timeout_hours"]
        self.position_timeout_hours = configs["Trade"]["position_timeout_hours"]
        self.num_retries = 3  # number of retries to place an order

    def connect_to_api(self, api_key: str, api_secret: str) -> None:
        """
        Connect to the ByBit API.

        Parameters
        ----------
        api_key : str
            API key for the ByBit account.
        api_secret : str
            API secret for the ByBit account.
        """
        test = environ["ENV"] == "debug"
        self.client = unified_trading.HTTP(api_key=api_key, api_secret=api_secret, testnet=test)

    def get_ticker_names(self, min_volume: float) -> Tuple[List[str], List[float], List[str]]:
        """
        Retrieve ticker symbols and their corresponding volumes,
        filtering by minimum volume.

        Parameters
        ----------
        min_volume : float
            Minimum trading volume to filter tickers.

        Returns
        -------
        tuple of lists
            A tuple containing:
            - A list of filtered symbols.
            - A list of their respective 24-hour volumes.
            - A list of all symbols before filtering.
        """
        tickers = pd.DataFrame(self.client.get_tickers(category="linear")["result"]["list"])
        all_tickers = tickers["symbol"].to_list()

        tickers = tickers[(tickers["symbol"].str.endswith("USDT"))]
        tickers["volume24h"] = tickers["volume24h"].astype(float)
        tickers["lastPrice"] = tickers["lastPrice"].astype(float)
        ticker_vol = tickers["volume24h"] * tickers["lastPrice"]
        tickers = tickers[ticker_vol >= min_volume // 2]

        filtered_symbols = self.check_symbols(tickers["symbol"])
        tickers = tickers[tickers["symbol"].isin(filtered_symbols)]
        tickers = tickers.reset_index(drop=True)

        return tickers["symbol"].to_list(), tickers["volume24h"].to_list(), all_tickers

    def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        Retrieve K-line (candlestick) data for a given symbol and interval.

        Parameters
        ----------
        symbol : str
            The symbol of the cryptocurrency (e.g., 'BTCUSDT').
        interval : str
            The interval for the K-lines (e.g., '1h', '1d').
        limit : int
            The maximum number of data points to retrieve.

        Returns
        -------
        pd.DataFrame
            DataFrame containing time, open, high, low, close, and volume.
        """
        interval = self.convert_interval(interval)
        tickers = pd.DataFrame(
            self.client.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)[
                "result"
            ]["list"]
        )
        tickers = tickers.rename(
            {0: "time", 1: "open", 2: "high", 3: "low", 4: "close", 6: "volume"}, axis=1
        )
        return tickers[["time", "open", "high", "low", "close", "volume"]][::-1].reset_index(
            drop=True
        )

    def get_historical_klines(
        self, symbol: str, interval: str, limit: int, min_time: datetime
    ) -> pd.DataFrame:
        """
        Retrieve historical K-line data for a given symbol and
        interval, before a specified minimum time.

        Parameters
        ----------
        symbol : str
            The symbol of the cryptocurrency (e.g., 'BTCUSDT').
        interval : str
            The interval for the K-lines (e.g., '1h', '1d').
        limit : int
            The maximum number of data points to retrieve in each request.
        min_time : datetime
            The earliest time for which data should be retrieved.

        Returns
        -------
        pd.DataFrame
            DataFrame containing historical time, open, high, low, close, and volume.
        """
        interval_secs = self.convert_interval_to_secs(interval)
        interval = self.convert_interval(interval)
        tmp_limit = limit
        prev_time, earliest_time = None, datetime.now()
        timestamp_ = int(self.get_timestamp() / 3600) * 3600
        tickers = pd.DataFrame()

        while earliest_time > min_time:
            start = (timestamp_ - (tmp_limit * interval_secs)) * 1000
            tmp = pd.DataFrame(
                self.client.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    start=start,
                    limit=limit,
                )["result"]["list"]
            )
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp[0].min()
            earliest_time = self.convert_timestamp_to_time(earliest_time, unit="ms")
            if prev_time == earliest_time:  # prevent endless loop
                break

            if tickers.shape[0] > 0:  # drop duplicates
                tickers = tickers[tickers[0] > tmp[0].max()]

            tickers = pd.concat([tickers, tmp])
            tmp_limit += limit

        tickers = tickers.rename(
            {0: "time", 1: "open", 2: "high", 3: "low", 4: "close", 6: "volume"}, axis=1
        )
        return tickers[["time", "open", "high", "low", "close", "volume"]][::-1].reset_index(
            drop=True
        )

    def get_historical_funding_rate(
        self, symbol: str, limit: int, min_time: datetime
    ) -> pd.DataFrame:
        """
        Retrieve historical funding rate information for a cryptocurrency
        pair before a specified minimum time.

        Parameters
        ----------
        symbol : str
            The symbol of the cryptocurrency pair (e.g., 'BTCUSDT').
        limit : int
            The maximum number of data points to retrieve in each request.
        min_time : datetime
            The earliest time for which data should be retrieved.

        Returns
        -------
        pd.DataFrame
            DataFrame containing time and funding rate for the specified symbol.
        """
        interval_secs = 8 * 3600 * 1000
        prev_time, earliest_time = None, datetime.now()
        end_time = int(self.get_timestamp() / 3600) * 3600 * 1000
        funding_rates = pd.DataFrame()

        while earliest_time > min_time:
            tmp = pd.DataFrame(
                self.client.get_funding_rate_history(
                    category="linear", symbol=symbol, limit=limit, endTime=end_time
                )["result"]["list"]
            )
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp["fundingRateTimestamp"].min()
            earliest_time = self.convert_timestamp_to_time(earliest_time, unit="ms")
            if prev_time == earliest_time:  # prevent endless loop
                break

            if funding_rates.shape[0] > 0:  # drop duplicates
                funding_rates = funding_rates[
                    funding_rates["fundingRateTimestamp"] > tmp["fundingRateTimestamp"].max()
                ]

            funding_rates = pd.concat([funding_rates, tmp])
            end_time -= limit * interval_secs

        funding_rates = funding_rates.rename(
            {"fundingRateTimestamp": "time", "fundingRate": "funding_rate"}, axis=1
        )
        return funding_rates[["time", "funding_rate"]][::-1].reset_index(drop=True)

    # ===== Trading =====
    def pos_mode_switch(self, symbol: str) -> None:
        """
        Switch position mode between one-side and hedge mode.

        Parameters
        ----------
        symbol : str
            The symbol for which the position mode should be switched.
        """
        mode = 0 if self.one_way_mode else 3
        try:
            self.client.switch_position_mode(category="linear", symbol=symbol, mode=mode)
        except InvalidRequestError as exc:
            if exc.status_code == 110025:
                pass
            else:
                raise exc

    def set_margin(self, symbol: str) -> None:
        """
        Set margin level for the current symbol.

        Parameters
        ----------
        symbol : str
            The symbol for which the margin should be set.
        """
        trade_mode = 1 if self.is_isolated else 0
        try:
            self.client.switch_margin_mode(
                category="linear",
                symbol=symbol,
                tradeMode=trade_mode,
                buyLeverage=str(self.leverage),
                sellLeverage=str(self.leverage),
            )
        except InvalidRequestError as exc:
            if exc.status_code == 110026:
                pass
            else:
                raise exc

    def set_leverage(self, symbol: str) -> None:
        """
        Set leverage for the current symbol.

        Parameters
        ----------
        symbol : str
            The symbol for which the leverage should be set.
        """
        try:
            self.client.set_leverage(
                category="linear",
                symbol=symbol,
                buy_leverage=str(self.leverage),
                sell_leverage=str(self.leverage),
            )
        except InvalidRequestError as exc:
            if exc.status_code == 110043:
                pass
            else:
                raise exc

    def set_settings(self, symbol: str) -> None:
        """
        Set all necessary trading settings (leverage, margin) for the given symbol.

        Parameters
        ----------
        symbol : str
            The symbol for which the settings should be applied.
        """
        self._get_round_digits_qty(symbol)
        self.set_margin(symbol)
        self.set_leverage(symbol)

    def query_symbols_info(self) -> None:
        """
        Retrieve and store information about all symbols available for trading.
        """
        info = self.client.get_instruments_info(category="linear")["result"]
        for i in info["list"]:
            self.symbols_info[i["symbol"]] = i

    def _get_min_trading_qty(self, symbol: str) -> Tuple[float, float]:
        """
        Retrieve the minimum trading quantity and notional value for a given symbol.

        Parameters
        ----------
        symbol : str
            The symbol for which the minimum quantities are required.

        Returns
        -------
        tuple of float
            Minimum order quantity and minimum notional value for the symbol.
        """
        s_i = self.symbols_info[symbol]
        min_qty = s_i["lotSizeFilter"]["minOrderQty"]
        min_quot_qty = s_i["lotSizeFilter"]["minNotionalValue"]
        return float(min_qty), float(min_quot_qty)

    def get_tick_size(self, symbol: str) -> float:
        """
        Retrieve the tick size (price increment) for a given symbol.

        Parameters
        ----------
        symbol : str
            The symbol for which the tick size is required.

        Returns
        -------
        float
            The tick size of the symbol.
        """
        s_i = self.symbols_info[symbol]
        tick_size = s_i["priceFilter"]["tickSize"]
        return float(tick_size)

    def get_round_digits_tick_size(self, symbol: str) -> Tuple[int, float]:
        """
        Calculate the number of digits for price rounding based on the tick size.

        Parameters
        ----------
        symbol : str
            The symbol for which the rounding digits are required.

        Returns
        -------
        tuple
            The number of rounding digits and the tick size.
        """
        tick_size = self.get_tick_size(symbol)
        ts_round_digits_num = max(ceil(-log10(tick_size)), 0)
        return ts_round_digits_num, tick_size

    def get_price(self, symbol: str) -> float:
        """
        Retrieve the current price of a symbol.

        Parameters
        ----------
        symbol : str
            The symbol for which the current price is required.

        Returns
        -------
        float
            The current price of the symbol.
        """
        timestamp_ = self.get_timestamp() - 61

        try:
            price = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=1,
                limit=2,
                from_time=timestamp_,
            )["result"]["list"][0][4]
        except IndexError:
            price = 0
        return float(price)

    def _get_round_digits_qty(self, symbol: str) -> int:
        """
        Calculate the number of digits for trade quantity rounding
        based on the minimum order quantity.

        Parameters
        ----------
        symbol : str
            The symbol for which the rounding digits are required.

        Returns
        -------
        int
            The number of rounding digits for the trade quantity.
        """
        min_qty, _ = self._get_min_trading_qty(symbol)
        round_digits_num = max(int(-log10(min_qty)), 0)
        return round_digits_num

    def get_balance(self, coin: str) -> float:
        """
        Retrieve the available balance for a given coin.

        Parameters
        ----------
        coin : str
            The coin for which the balance is required.

        Returns
        -------
        float
            The available balance of the coin.
        """
        balance = self.client.get_wallet_balance(accountType="CONTRACT", coin=coin)
        balance_list = balance["result"]["list"][0]["coin"]
        free_balance = 0.0
        for balance in balance_list:
            if balance["coin"] == coin:
                free_balance = float(balance["walletBalance"])
        return free_balance

    def _get_quantity(
        self,
        symbol: str,
        prices: List[float],
        take_profits: List[float],
        stop_loss: float,
        side: str,
        divide: int,
    ) -> Tuple[float, str]:
        """
        Calculate the quantity to trade based on available balance, risk,
        and price changes.

        Parameters
        ----------
        symbol : str
            The symbol to trade.
        prices : list of float
            List of entry prices.
        take_profits : list of float
            List of take profit levels.
        stop_loss : float
            Stop loss level.
        side : str
            Trading side ("Buy" or "Sell").
        divide : int
            A factor to divide the risk calculation.

        Returns
        -------
        tuple of float and str
            The calculated quantity and a message with detailed
            information about the trade.
        """
        message = ""
        # find the most unprofitable entry price (depends on trading side)
        # and use it to calculate the quantity
        price = max(prices) if side == "Buy" else min(prices)
        # get number of digits for prices rounding (depends on tick size)
        round_digits_num = self._get_round_digits_qty(symbol)
        # get the free balance
        free_balance = self.get_balance(self.quote_coin)
        # get change of price from entry to SL and multiply it by leverage
        price_change = abs(price - stop_loss) / price * self.leverage
        # use price change to find the amount of balance that will be used in the trade
        quantity = (free_balance * self.risk / price_change) / price
        quote_quantity = (free_balance * self.risk / price_change) * self.leverage
        message += (
            f"Symbol is {symbol}\n"
            f"Min trading quantity is {round(quantity, round_digits_num + 1)}\n"
            f"Free balance is {round(free_balance, 2)} {self.quote_coin}\n"
            f"Risk is {self.risk * 100}% x {len(prices)} x {len(take_profits)} = "
            f"{round(self.risk * divide * 100, 2)}%\n"
            f"{round(quantity * divide * price, 2)} "
            f"{self.quote_coin} will be used as margin "
            f"to {side} {round(quantity * divide * self.leverage, round_digits_num)} "
            f"{symbol[:-4]} at price ${price} "
            f"with leverage {self.leverage}x\n"
        )
        logger.info(message)
        quantity = round(quantity * self.leverage, round_digits_num)
        min_qty, min_quote_qty = self._get_min_trading_qty(symbol)
        if quantity == 0:
            message += (
                f"Minimal available trade quantity is 0. "
                f"Increase trade volume or leverage or both. "
                f"Minimal trading quantity for ticker {symbol} is {min_qty}"
            )
        elif quote_quantity < min_quote_qty:
            message += (
                f"Minimal available quote trade quantity {quote_quantity} "
                "is less than minimal quote trade quantity. "
                "Increase trade volume or leverage or both. "
                f"Minimal quote trade quantity for ticker {symbol} is {min_quote_qty}"
            )
            quantity = 0
        return quantity, message

    def place_conditional_order(
        self,
        symbol: str,
        side: str,
        price: float,
        trigger_direction: int,
        trigger_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
    ) -> str:
        """
        Place a conditional order with a specified trigger price and other parameters.

        Parameters
        ----------
        symbol : str
            The symbol to trade.
        side : str
            Trading side ("Buy" or "Sell").
        price : float
            Limit price for the order.
        trigger_direction : int
            Direction of the trigger (1 for rising price, 2 for falling price).
        trigger_price : float
            The price at which the order is triggered.
        quantity : float
            The amount to trade.
        stop_loss : float
            Stop loss level.
        take_profit : float
            Take profit level.

        Returns
        -------
        str
            Message indicating the success of the conditional order placement.
        """
        if side == "Buy":
            position_idx = 1  # hedge-mode Buy side
        else:
            position_idx = 2  # hedge-mode Sell side

        self.client.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Limit",
            qty=quantity,
            price=price,
            triggerDirection=trigger_direction,
            triggerPrice=trigger_price,
            triggerBy="LastPrice",
            timeInForce="GTC",
            positionIdx=position_idx,
            takeProfit=take_profit,
            stopLoss=stop_loss,
            tpTriggerBy="LastPrice",
            slTriggerBy="LastPrice",
            reduceOnly=False,
            closeOnTrigger=False,
            tpslMode="Partial",
        )
        message = (
            f"\nConditional order is placed, Trigger is {trigger_price}, "
            f"Limit price is {price}, SL is {stop_loss}, TP is {take_profit}\n"
        )
        logger.info(message)
        return message

    def find_open_orders(self) -> None:
        """
        Find and cancel open orders that were not
        triggered within a certain time period.
        """
        ts_now = self.get_timestamp()
        order_ids_to_cancel = []
        orders = self.client.get_open_orders(category="linear", settleCoin=self.quote_coin)[
            "result"
        ]["list"]
        for order in orders:
            symbol = order["symbol"]
            created_time = int(order["createdTime"]) // 1000
            order_id = order["orderId"]
            side = order["side"]
            created_price = float(order["lastPriceOnCreated"])
            trigger_price = float(order["triggerPrice"])
            # check if creation price and trigger price almost do not differ
            # this will tell us that this order is not TP / SL order
            diff = abs(created_price - trigger_price) / created_price
            # check the amount of time passed
            time_span = (ts_now - created_time) // 3600
            if time_span >= self.order_timeout_hours and diff < 0.01:
                order_ids_to_cancel.append((symbol, order_id, side))

        for s_o in order_ids_to_cancel:
            symbol, order_id, side = s_o
            message = f"\nOrder timeout. {side} order for ticker {symbol} is cancelled.\n"
            logger.info(message)
            self.client.cancel_order(category="linear", symbol=symbol, orderId=order_id)

    def check_open_positions(self, symbol: str = "") -> bool:
        """
        Check for open positions and close those that have exceeded the timeout period.

        Parameters
        ----------
        symbol : str, optional
            The symbol to check for open positions (if None, check all symbols).

        Returns
        -------
        bool
            True if there are no open positions or all timed-out positions were closed,
            False otherwise.
        """
        ts_now = self.get_timestamp() // 3600
        positions_to_close = []
        if symbol:
            positions = self.client.get_positions(category="linear", symbol=symbol)["result"][
                "list"
            ]
        else:
            positions = self.client.get_positions(category="linear", settleCoin=self.quote_coin)[
                "result"
            ]["list"]
        for pos in positions:  # for every opened position
            symbol = pos["symbol"]
            side = pos["side"]
            size = pos["size"]
            # get time when this position was opened
            last_similar_trade = self._get_last_similar_trade(symbol, side, size)
            if not last_similar_trade:
                positions_to_close.append((symbol, side, size))
                continue
            created_time = int(last_similar_trade["execTime"]) // (3600 * 1000)
            time_span = ts_now - created_time  # check the amount of time passed
            if (
                time_span >= self.position_timeout_hours
                and float(size) > 0
                and side in ["Buy", "Sell"]
            ):
                # if enough time passed - add this position to the list
                # of positions that should be closed
                positions_to_close.append((symbol, side, size))

        for pos in positions_to_close:
            symbol, side, size = pos
            side = "Sell" if side == "Buy" else "Buy"
            self.close_order(symbol, side, size)

        return len(positions) == 0 or len(positions) == len(positions_to_close)

    def _get_last_similar_trade(self, symbol: str, side: str, size: str) -> Union[dict, None]:
        """
        Retrieve the last similar trade based on side and size for a given symbol.

        Parameters
        ----------
        symbol : str
            The symbol to retrieve trade history for.
        side : str
            Trading side ("Buy" or "Sell").
        size : str
            The size of the trade.

        Returns
        -------
        dict or None
            The last similar trade, or None if no similar trades are found.
        """
        trade_history = self.client.get_executions(
            category="linear", symbol=symbol, execType="Trade"
        )["result"]["list"]
        trade_history = [
            t_h for t_h in trade_history if t_h["side"] == side and t_h["orderQty"] == size
        ]
        if trade_history:
            return trade_history[0]
        return None

    def place_market_order(self, symbol: str, side: str, size: str) -> None:
        """
        Place a market order to open or close a position.

        Parameters
        ----------
        symbol : str
            The symbol to trade.
        side : str
            Trading side ("Buy" or "Sell").
        size : str
            The size of the trade.
        """
        position_idx = 1 if side == "Buy" else 2
        self.client.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=size,
            timeInForce="GTC",
            positionIdx=position_idx,
        )

    def close_order(self, symbol: str, side: str, size: str) -> str:
        """
        Place a market order to close a position.

        Parameters
        ----------
        symbol : str
            The symbol to trade.
        side : str
            Trading side ("Buy" or "Sell").
        size : str
            The size of the position to close.

        Returns
        -------
        str
            Message indicating that the position was closed.
        """
        position_idx = 2 if side == "Buy" else 1
        self.client.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=size,
            timeInForce="GTC",
            positionIdx=position_idx,
            reduceOnly=True,
        )
        message = f"\nPosition timeout. Market {side} order for ticker {symbol} is placed\n"
        logger.info(message)
        return message

    @staticmethod
    def _set_trigger_price(price: float, direction: str, tick_size: float) -> Tuple[int, float]:
        """
        Set trigger price based on the direction of the trade.

        Parameters
        ----------
        price : float
            The base price for the trade.
        direction : str
            Direction of the trade ("Rise" or "Fall").
        tick_size : float
            The tick size of the symbol.

        Returns
        -------
        tuple of int and float
            Trigger direction and the trigger price.
        """
        if direction == "Rise":
            trigger_direction = 1
            trigger_price = price - tick_size
        else:
            trigger_direction = 2
            trigger_price = price + tick_size
        return trigger_direction, trigger_price

    def place_all_conditional_orders(self, symbol: str, side: str) -> Tuple[bool, str]:
        """
        Place all necessary conditional orders for a symbol.

        Parameters
        ----------
        symbol : str
            The symbol to trade.
        side : str
            Trading side ("Buy" or "Sell").

        Returns
        -------
        tuple of bool and str
            Status of order placement and a message with details.
        """
        self.set_settings(symbol)
        ts_round_digits_num, tick_size = self.get_round_digits_tick_size(symbol)
        price = self.get_price(symbol)

        if side == "Buy":
            direction = "Fall"
            prices = [price - 2 * tick_size]
            take_profits = [price * (1 + self.TP)]
            stop_loss = price * (1 - self.SL)
        else:
            direction = "Rise"
            prices = [price + 2 * tick_size]
            take_profits = [price * (1 - self.TP)]
            stop_loss = price * (1 + self.SL)

        quantity, message = self._get_quantity(
            symbol,
            prices,
            take_profits,
            stop_loss,
            side,
            len(take_profits) * len(prices),
        )
        if quantity == 0:
            logger.info(message)
            return False, message

        for _, price in enumerate(prices):
            if quantity > 0:
                trigger_direction, trigger_price = self._set_trigger_price(
                    price, direction, tick_size
                )
                price = round(price, ts_round_digits_num)
                trigger_price = round(trigger_price, ts_round_digits_num)
                stop_loss = round(stop_loss, ts_round_digits_num)
                for take_profit in take_profits:
                    take_profit = round(take_profit, ts_round_digits_num)
                    logger.info(f"Place conditional order for ticker {symbol}")
                    for i in range(self.num_retries):
                        try:
                            message += self.place_conditional_order(
                                symbol,
                                side,
                                price,
                                trigger_direction,
                                trigger_price,
                                quantity,
                                stop_loss,
                                take_profit,
                            )
                        except pybit.exceptions.InvalidRequestError:
                            logger.exception(
                                f"Catch an exception while trying place "
                                f"conditional order for ticker {symbol}"
                            )
                            sleep(0.5)
                            price = self.get_price(symbol)
                            trigger_direction, trigger_price = self._set_trigger_price(
                                price, direction, tick_size
                            )
                            logger.info(
                                f"Attempt number {i+1}, ticker price is {price}, "
                                f"trigger price is {trigger_price}"
                            )
                            continue
                        else:
                            break
                sleep(0.1)
        return True, message
