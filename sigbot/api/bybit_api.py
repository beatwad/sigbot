from datetime import datetime
import pandas as pd
from api.api_base import ApiBase
from pybit import unified_trading


class ByBit(ApiBase):
    client = ""

    def __init__(self, api_key="Key", api_secret="Secret"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.connect_to_api('', '')

    def connect_to_api(self, api_key, api_secret):
        self.client = unified_trading.HTTP(api_key=api_key, api_secret=api_secret)

    def get_ticker_names(self, min_volume) -> (list, list, list):
        tickers = pd.DataFrame(self.client.get_tickers(category="spot")['result']['list'])
        all_tickers = tickers['symbol'].to_list()

        tickers = tickers[(tickers['symbol'].str.endswith('USDT')) | (tickers['symbol'].str.endswith('USDC'))]
        tickers['volume24h'] = tickers['volume24h'].astype(float)
        tickers['lastPrice'] = tickers['lastPrice'].astype(float)
        ticker_vol = tickers['volume24h'] * tickers['lastPrice']
        tickers = tickers[ticker_vol >= min_volume // 2]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        filtered_symbols = self.delete_duplicate_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list(), tickers['volume24h'].to_list(), all_tickers

    def get_klines(self, symbol, interval, limit) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        interval = self.convert_interval(interval)
        tickers = pd.DataFrame(self.client.get_kline(category='spot', symbol=symbol,
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

        # get historical data in cycle until we reach the min_time
        while earliest_time > min_time:
            start = (ts - (tmp_limit * interval_secs)) * 1000
            tmp = pd.DataFrame(self.client.get_kline(category='spot', symbol=symbol,
                                                     interval=interval, start=start, limit=limit)['result']['list'])
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp[0].min()
            earliest_time = self.convert_timstamp_to_time(earliest_time, unit='ms')
            # prevent endless cycle if there are no candles that earlier than min_time
            if prev_time == earliest_time:
                break
            
            tickers = pd.concat([tickers, tmp])
            tmp_limit += limit
            
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 6: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']][::-1].reset_index(drop=True)


if __name__ == '__main__':
    key = ""
    secret = ""
    bybit_api = ByBit()
    tickers = bybit_api.get_ticker_names(500000)
    kline = bybit_api.get_klines('VINUUSDT', '5m', 1000)
    print(kline)