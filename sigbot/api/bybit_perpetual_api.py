import pandas as pd
from os import environ
from datetime import datetime
from math import log10, ceil
from time import sleep
from api.api_base import ApiBase
from pybit import unified_trading
from pybit.exceptions import InvalidRequestError
from log.log import logger
# logger = None
from config.config import ConfigFactory

# Set environment variable
# environ["ENV"] = "debug"
configs = ConfigFactory.factory(environ).configs


class ByBitPerpetual(ApiBase):
    client = ""

    def __init__(self, api_key="Key", api_secret="Secret"):
        self.global_limit = 1000
        self.api_key = api_key
        self.api_secret = api_secret
        self.connect_to_api(self.api_key, self.api_secret)
        self.symbols_info = dict()
        self.query_symbols_info()
        self.leverage = configs['Trade']['leverage']
        self.risk = configs['Trade']['risk']
        self.quote_coin = configs['Trade']['quote_coin']
        self.currency = configs['Trade']['currency']
        self.one_way_mode = configs['Trade']['one_way_mode']
        self.is_isolated = configs['Trade']['is_isolated']
        self.order_timeout_hours = configs['Trade']['order_timeout_hours']
        self.position_timeout_hours = configs['Trade']['position_timeout_hours']

    def connect_to_api(self, api_key, api_secret):
        # TODO remove this when end debugging
        # if environ['ENV'] == 'debug':
        #     test = True
        # else:
        test = False
        self.client = unified_trading.HTTP(api_key=api_key, api_secret=api_secret, testnet=test)

    def get_ticker_names(self, min_volume) -> (list, list, list):
        tickers = pd.DataFrame(self.client.get_tickers(category="linear")['result']['list'])
        all_tickers = tickers['symbol'].to_list()

        tickers = tickers[(tickers['symbol'].str.endswith('USDT'))]
        tickers['volume24h'] = tickers['volume24h'].astype(float)
        tickers['lastPrice'] = tickers['lastPrice'].astype(float)
        ticker_vol = tickers['volume24h'] * tickers['lastPrice']
        tickers = tickers[ticker_vol >= min_volume // 2]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list(), tickers['volume24h'].to_list(), all_tickers

    def get_klines(self, symbol, interval, limit) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        interval = self.convert_interval(interval)
        tickers = pd.DataFrame(self.client.get_kline(category='linear', symbol=symbol,
                                                     interval=interval, limit=limit)['result']['list'])
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 6: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']][::-1].reset_index(drop=True)
    
    def get_historical_klines(self, symbol: str, interval: str, limit: int, min_time: datetime) -> pd.DataFrame:
        """ Save historical time, price and volume info to CryptoCurrency structure
            for some period (earlier than min_time) """
        interval_secs = self.convert_interval_to_secs(interval)
        interval = self.convert_interval(interval)
        tmp_limit = limit
        prev_time, earliest_time = None, datetime.now()
        ts = int(self.get_timestamp() / 3600) * 3600
        tickers = pd.DataFrame()

        while earliest_time > min_time:
            start = (ts - (tmp_limit * interval_secs)) * 1000
            tmp = pd.DataFrame(self.client.get_kline(category='linear', symbol=symbol,
                                                     interval=interval, start=start, limit=limit)['result']['list'])
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp[0].min()
            earliest_time = self.convert_timstamp_to_time(earliest_time, unit='ms')
            # prevent endless cycle if there are no candles that earlier than min_time
            if prev_time == earliest_time:
                break
            
            # drop duplicated rows
            if tickers.shape[0] > 0:
                tickers = tickers[tickers[0] > tmp[0].max()]
                
            tickers = pd.concat([tickers, tmp])
            tmp_limit += limit

        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 6: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']][::-1].reset_index(drop=True)

    def get_historical_funding_rate(self, symbol: str, limit: int, min_time: datetime) -> pd.DataFrame:
        """ Save historical funding rate info to CryptoCurrency structure
            for some period (earlier than min_time) """
        interval_secs = 8 * 3600 * 1000
        prev_time, earliest_time = None, datetime.now()
        end_time = int(self.get_timestamp() / 3600) * 3600 * 1000
        funding_rates = pd.DataFrame()
        
        while earliest_time > min_time:
            tmp = pd.DataFrame(self.client.get_funding_rate_history(category='linear', symbol=symbol, limit=limit,
                                                                    endTime=end_time)['result']['list'])
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp['fundingRateTimestamp'].min()
            earliest_time = self.convert_timstamp_to_time(earliest_time, unit='ms')
            # prevent endless cycle if there are no candles that earlier than min_time
            if prev_time == earliest_time:
                break

            # drop duplicated rows
            if funding_rates.shape[0] > 0:
                funding_rates = funding_rates[funding_rates['fundingRateTimestamp'] > tmp['fundingRateTimestamp'].max()]

            funding_rates = pd.concat([funding_rates, tmp])
            end_time = (end_time - (limit * interval_secs))
        funding_rates = funding_rates.rename({'fundingRateTimestamp': 'time',
                                              'fundingRate': 'funding_rate'}, axis=1)
        return funding_rates[['time', 'funding_rate']][::-1].reset_index(drop=True)

    # ===== Trading =====
    def pos_mode_switch(self, symbol) -> None:
        """ Change position mode between one-side and hedge mode """
        mode = 0 if self.one_way_mode else 3
        try:
            self.client.switch_position_mode(
                category='linear',
                symbol=symbol,
                mode=mode
            )
        except InvalidRequestError as e:
            if e.status_code == 110025:
                pass
            else:
                raise e

    def set_margin(self, symbol) -> None:
        """ Set margin level for current symbol """
        trade_mode = 1 if self.is_isolated else 0
        try:
            self.client.switch_margin_mode(
                category='linear',
                symbol=symbol,
                tradeMode=trade_mode,
                buyLeverage=str(self.leverage),
                sellLeverage=str(self.leverage)
            )
        except InvalidRequestError as e:
            if e.status_code == 110026:
                pass
            else:
                raise e

    def set_leverage(self, symbol) -> None:
        """ Set leverage for current symbol """
        try:
            self.client.set_leverage(
                category='linear',
                symbol=symbol,
                buy_leverage=str(self.leverage),
                sell_leverage=str(self.leverage),
            )
        except InvalidRequestError as e:
            if e.status_code == 110043:
                pass
            else:
                raise e

    def set_settings(self, symbol: str) -> None:
        """ Set all necessary settings before making order """
        self.pos_mode_switch(symbol)
        self.get_round_digits_qty(symbol)
        self.set_margin(symbol)
        self.set_leverage(symbol)

    def query_symbols_info(self) -> None:
        """ Get information about all symbols """
        info = self.client.get_instruments_info(category="linear")['result']
        for i in info['list']:
            self.symbols_info[i['symbol']] = i

    def get_min_trading_qty(self, symbol) -> float:
        """ Get minimal amount of currency, that can be used for trading """
        s_i = self.symbols_info[symbol]
        qty = s_i['lotSizeFilter']['minOrderQty']
        return qty

    def get_tick_size(self, symbol) -> float:
        """ Get currency's tick size """
        s_i = self.symbols_info[symbol]
        tick_size = s_i['priceFilter']['tickSize']
        return float(tick_size)

    def get_round_digits_tick_size(self, symbol) -> (int, float):
        """ Get number of digits for price rounding """
        tick_size = self.get_tick_size(symbol)
        ts_round_digits_num = max(ceil(-log10(tick_size)), 0)
        return ts_round_digits_num, tick_size

    def get_price(self, symbol) -> float:
        """ Get current price of symbol  """
        ts = self.get_timestamp() - 61

        try:
            price = self.client.get_kline(
                category='linear',
                symbol=symbol,
                interval=1,
                limit=2,
                from_time=ts)['result']['list'][0][4]
        except IndexError:
            price = 0
        return float(price)

    def get_round_digits_qty(self, symbol) -> int:
        """ Get number of digits for trade quantity rounding """
        min_trading_qty = float(self.get_min_trading_qty(symbol))
        round_digits_num = max(int(-log10(min_trading_qty)), 0)
        return round_digits_num

    def get_balance(self, coin) -> float:
        """ Get balance for current coin """
        balance = self.client.get_wallet_balance(accountType='CONTRACT', coin=coin)
        balance_list = balance['result']['list'][0]['coin']
        free_balance = 0
        for balance in balance_list:
            if balance['coin'] == coin:
                free_balance = float(balance['walletBalance'])
        return free_balance

    def get_quantity(self, symbol: str, prices: list, take_profits: list,
                     stop_loss: float, side: str, divide: int) -> (float, str):
        """ Calculate quantity of currency, that will be used for trade, considering available balance, risk value
        and minimal quantity that is needed for trading """
        message = ''
        # find the most unprofitable entry price (depends on trading side) and use it to calculate the quantity
        if side == 'Buy':
            price = max(prices)
        else:
            price = min(prices)
        # get number of digits for prices rounding (depends on tick size)
        round_digits_num = self.get_round_digits_qty(symbol)
        # get the free balance
        free_balance = self.get_balance(self.quote_coin)
        # get change of price from entry to SL and multiply it by leverage
        price_change = abs(price - stop_loss) / price * self.leverage
        # use price change to find the amount of balance that will be used in the trade
        quantity = ((free_balance * self.risk / price_change) / price)
        message += f'Symbol is {symbol}\n' \
                   f'Min trading quantity is {round(quantity, round_digits_num + 1)}\n' \
                   f'Free balance is {round(free_balance, 2)} {self.quote_coin}\n' \
                   f'Risk is {self.risk * 100}% x {len(prices)} x {len(take_profits)} = ' \
                   f'{round(self.risk * divide * 100, 2)}%\n' \
                   f'{round(quantity * divide * price, 2)} {self.quote_coin} will be used as margin ' \
                   f'to {side} {round(quantity * divide * self.leverage, round_digits_num)} '\
                   f'{self.currency} at price ${price} ' \
                   f'with leverage {self.leverage}x\n'
        logger.info(message)
        quantity = round(quantity * self.leverage, round_digits_num)
        if quantity == 0:
            message += f'Minimal available trade quantity is 0. Increase trade volume or leverage or both. ' \
                       f'Minimal trading quantity for ticker {symbol} is {self.get_min_trading_qty(symbol)}'
        return quantity, message

    def place_conditional_order(self, symbol, side, price, trigger_direction, trigger_price, quantity, stop_loss,
                                take_profit) -> str:
        """ Place conditional order """
        self.client.place_order(
            category='linear',
            symbol=symbol,
            side=side,
            orderType='Limit',
            qty=quantity,
            price=price,
            triggerDirection=trigger_direction,
            triggerPrice=trigger_price,
            triggerBy='LastPrice',
            timeInForce='GTC',
            positionIdx=0,
            takeProfit=take_profit,
            stopLoss=stop_loss,
            tpTriggerBy='LastPrice',
            slTriggerBy='MarkPrice',
            reduceOnly=False,
            closeOnTrigger=False,
            tpslMode='Partial'
        )
        message = (f'\nConditional order is placed, Trigger is {trigger_price}, Limit price is {price}, '
                   f'SL is {stop_loss}, TP is {take_profit}\n')
        logger.info(message)
        return message

    def find_open_orders(self) -> None:
        """ Find open orders (not TP / SL) that weren't triggered within a certain time and cancel them """
        ts_now = self.get_timestamp()
        order_ids_to_cancel = list()
        orders = self.client.get_open_orders(category='linear', settleCoin=self.quote_coin)['result']['list']
        for o in orders:
            symbol = o['symbol']
            created_time = int(o['createdTime']) // 1000
            order_id = o['orderId']
            side = o['side']
            created_price = float(o['lastPriceOnCreated'])
            trigger_price = float(o['triggerPrice'])
            # check if creation price and trigger price almost do not differ
            # this will tell us that this order is not TP / SL order
            diff = abs(created_price - trigger_price) / created_price
            # check the amount of time passed
            time_span = (ts_now - created_time) // 3600
            if time_span >= self.order_timeout_hours and diff < 0.01:
                order_ids_to_cancel.append((symbol, order_id, side))

        for s_o in order_ids_to_cancel:
            symbol, order_id, side = s_o
            message = f'\nOrder timeout. {side} order for ticker {symbol} is cancelled.\n'
            logger.info(message)
            self.client.cancel_order(category='linear', symbol=symbol, orderId=order_id)

    def check_open_positions(self, symbol=None) -> bool:
        """ Check if there are open positions. If they are, and we can't cancel them because of position timeout -
            don't open the new one. If there are positions that weren't closed within a certain time - close them """
        ts_now = self.get_timestamp()
        positions_to_close = list()
        if symbol:
            positions = self.client.get_positions(category='linear', symbol=symbol)['result']['list']
        else:
            positions = self.client.get_positions(category='linear', settleCoin=self.quote_coin)['result']['list']
        for p in positions:
            created_time = int(p['updatedTime']) // 1000
            symbol = p['symbol']
            side = p['side']
            size = p['size']
            status = p['positionStatus']
            # check the amount of time passed
            time_span = (ts_now - created_time) // 3600
            if time_span >= self.position_timeout_hours and status == 'Normal':
                positions_to_close.append((symbol, side, size))

        for pos in positions_to_close:
            symbol, side, size = pos
            side = 'Sell' if side == 'Buy' else 'Buy'
            self.place_market_order(symbol, side, size)

        if len(positions) == 0 or len(positions) == len(positions_to_close):
            return True
        return False

    def place_market_order(self, symbol, side, size) -> str:
        """ Place market order to close position """
        self.client.place_order(
            category='linear',
            symbol=symbol,
            side=side,
            orderType='Market',
            qty=size,
            timeInForce='GTC',
            positionIdx=0,
            reduceOnly=False
        )
        message = f'\nPosition timeout. Market {side} order for ticker {symbol} is placed\n'
        logger.info(message)
        return message

    def place_all_conditional_orders(self, symbol, side) -> (bool, str):
        """ Place all necessary conditional orders for symbol """
        if not self.check_open_positions(symbol):
            message = f"There are opened positions for ticker {symbol}, don't open the new one."
            logger.info(message)
            return False, message

        self.set_settings(symbol)
        ts_round_digits_num, tick_size = self.get_round_digits_tick_size(symbol)
        price = self.get_price(symbol)

        if side == 'Buy':
            direction = 'Fall'
            prices = [price - 2 * tick_size]
            take_profits = [price * 1.03]
            stop_loss = price * 0.969
        else:
            direction = 'Rise'
            prices = [price + 2 * tick_size]
            take_profits = [price * 0.97]
            stop_loss = price * 1.031

        quantity, message = self.get_quantity(symbol, prices, take_profits, stop_loss, side,
                                              len(take_profits) * len(prices))
        if quantity == 0:
            logger.info(message)
            return False, message

        for i, price in enumerate(prices):
            if quantity > 0:
                if direction == 'Rise':  # price rises to trigger level, hits stop_px, then price
                    # stop_px > max(market_price, base_price)
                    trigger_direction = 1
                    trigger_price = price - tick_size
                else:  # price falls to trigger level, hits stop_px, then price
                    # stop_px < min(market_price, base_price)
                    trigger_direction = 2
                    trigger_price = price + tick_size
                price = round(price, ts_round_digits_num)
                trigger_price = round(trigger_price, ts_round_digits_num)
                stop_loss = round(stop_loss, ts_round_digits_num)
                for take_profit in take_profits:
                    take_profit = round(take_profit, ts_round_digits_num)
                    print(symbol, side, direction)
                    message += self.place_conditional_order(symbol, side, price, trigger_direction,
                                                            trigger_price, quantity, stop_loss, take_profit)
                sleep(0.1)
        return True, message


if __name__ == '__main__':
    key = ""
    secret = ""
    bybit_api = ByBitPerpetual()
    # tickers = bybit_api.get_ticker_names(500000)
    # print(tickers)
    min_time_ = datetime.now().replace(microsecond=0, second=0, minute=0) - pd.to_timedelta(5 * 5, unit='D')
    # klines_ = bybit_api.get_klines(symbol='BTCUSDT', interval='1h', limit=200)
    klines_ = bybit_api.get_historical_funding_rate('BTCUSDT', 200, min_time_)
    klines_['time'] = pd.to_datetime(klines_['time'], unit='ms')
    klines_['time'] = klines_['time'] + pd.to_timedelta(3, unit='h')
    pass
