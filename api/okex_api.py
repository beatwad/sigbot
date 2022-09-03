import requests
import pandas as pd
from api.api_base import ApiBase


class Binance(ApiBase):
    client = ""
    URL = 'https://www.okex.com'

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

    def get_ticker_names(self, min_volume) -> list:
        tickers = pd.DataFrame(requests.get(self.URL + '/api/v5/market/tickers?instType=SPOT').json())
        tickers = tickers[(tickers['symbol'].str.endswith('USDT')) | (tickers['symbol'].str.endswith('BUSD'))]
        tickers['volume'] = tickers['volume'].astype(float)
        tickers = tickers[tickers['volume'] >= min_volume]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        filtered_symbols = self.delete_duplicate_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list()

    def get_klines(self, symbol, interval, limit) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        tickers = pd.DataFrame(requests.get(self.URL + '/api/v3/klines', params=params).json())
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']]