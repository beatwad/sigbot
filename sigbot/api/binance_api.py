from datetime import datetime
import pandas as pd
from api.api_base import ApiBase
from binance.client import Client


class Binance(ApiBase):
    client = ""

    def __init__(self, api_key="Key", api_secret="Secret"):
        if api_key != "Key" and api_secret != "Secret":
            self.connect_to_api(api_key, api_secret)
        else:
            self.api_key = api_key
            self.api_secret = api_secret

    def connect_to_api(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = Client(self.api_key, api_secret)

    @staticmethod
    def delete_duplicate_symbols(symbols) -> list:
        """ If for pair with USDT exists pair with BUSD - delete it  """
        filtered_symbols = list()
        symbols = symbols.to_list()

        for symbol in symbols:
            if symbol.endswith('BUSD'):
                prefix = symbol[:-4]
                if prefix + 'USDT' not in symbols:
                    filtered_symbols.append(symbol)
            else:
                filtered_symbols.append(symbol)
        return filtered_symbols

    def get_ticker_names(self, min_volume) -> (list, list, list):
        """ Get tickers and their volumes """
        tickers = pd.DataFrame(self.client.get_ticker())
        all_tickers = tickers['symbol'].to_list()

        tickers = tickers[(tickers['symbol'].str.endswith('USDT')) | (tickers['symbol'].str.endswith('BUSD'))]
        tickers.loc[:, 'quoteVolume'] = tickers.loc[:, 'quoteVolume'].astype(float)
        tickers = tickers[tickers['quoteVolume'] >= min_volume // 3]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        filtered_symbols = self.delete_duplicate_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list(), tickers['volume'].to_list(), all_tickers

    def get_klines(self, symbol: str, interval: str, limit: str) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        tickers = pd.DataFrame(self.client.get_klines(symbol=symbol, interval=interval, limit=limit))
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    def get_historical_klines(self, symbol: str, interval: str, limit: int, min_time: datetime) -> pd.DataFrame:
        """ Save historical time, price and volume info to CryptoCurrency structure
            for some period (earlier than min_time) """
        interval_secs = self.convert_interval_to_secs(interval)
        tmp_limit = limit
        prev_time, earliest_time = None, datetime.now()
        ts = int(self.get_timestamp() / 3600) * 3600
        tickers = pd.DataFrame()

        # get historical data in cycle until we reach the min_time
        while earliest_time > min_time:
            start_time = (ts - (tmp_limit * interval_secs)) * 1000
            tmp = pd.DataFrame(self.client.get_klines(symbol=symbol, interval=interval, startTime=start_time, limit=limit))
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp[0].min()
            earliest_time = self.convert_timstamp_to_time(earliest_time, unit='ms')
            # prevent endless cycle if there are no candles than earlier than min_time
            if prev_time == earliest_time:
                break
            tickers = pd.concat([tmp, tickers])
            tmp_limit += limit

        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)


if __name__ == '__main__':
    key = "7arxKITvadhYavxsQr5dZelYK4kzyBGM4rsjDCyJiPzItNlAEdlqOzibV7yVdnNy"
    secret = "3NvopCGubDjCkF4SzqP9vj9kU2UIhE4Qag9ICUdESOBqY16JGAmfoaUIKJLGDTr4"
    binance_api = Binance(key, secret)
    binance_api.get_ticker_names(1e1)
    klines = binance_api.get_klines('OSMOUSDT', '4h', 1000)
    t_list = list()
    # for t in tickers:
    #     t_list.append(t['symbol'])
    # exchange_info = binance_api.client.futures_exchange_info()
    # print('1000SHIBUSDT' in t_list)
