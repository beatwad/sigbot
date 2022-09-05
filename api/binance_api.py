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

    def get_ticker_names(self, min_volume) -> list:
        tickers = pd.DataFrame(self.client.get_ticker())
        tickers = tickers[(tickers['symbol'].str.endswith('USDT')) | (tickers['symbol'].str.endswith('BUSD'))]
        tickers['quoteVolume'] = tickers['quoteVolume'].astype(float)
        tickers = tickers[tickers['quoteVolume'] >= min_volume]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        filtered_symbols = self.delete_duplicate_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list()

    def get_klines(self, symbol, interval, limit) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        tickers = pd.DataFrame(self.client.get_klines(symbol=symbol, interval=interval, limit=limit))
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']]
