import pandas as pd
from api.api_base import ApiBase
from pybit import spot


class ByBit(ApiBase):
    client = ""

    def __init__(self, api_key="Key", api_secret="Secret"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.connect_to_api('', '')

    def connect_to_api(self, api_key, api_secret):
        self.client = spot.HTTP(api_key=api_key, api_secret=api_secret)

    @staticmethod
    def delete_duplicate_symbols(symbols) -> list:
        """ If for pair with USDT exists pair with BUSD - delete it  """
        filtered_symbols = list()
        symbols = symbols.to_list()

        for symbol in symbols:
            if symbol.endswith('USDC'):
                prefix = symbol[:-4]
                if prefix + 'USDT' not in symbols:
                    filtered_symbols.append(symbol)
            else:
                filtered_symbols.append(symbol)
        return filtered_symbols

    def get_ticker_names(self, min_volume) -> (list, list):
        tickers = pd.DataFrame(self.client.latest_information_for_symbol()['result'])

        all_tickers = tickers['symbol'].to_list()

        tickers = tickers[(tickers['symbol'].str.endswith('USDT')) | (tickers['symbol'].str.endswith('USDC'))]
        tickers['quoteVolume'] = tickers.loc[:, 'quoteVolume'].astype(float)
        tickers = tickers[tickers['quoteVolume'] >= min_volume]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        filtered_symbols = self.delete_duplicate_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list(), all_tickers

    def get_klines(self, symbol, interval, limit) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        tickers = pd.DataFrame(self.client.query_kline(symbol=symbol, interval=interval, limit=limit)['result'])
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']]


if __name__ == '__main__':
    key = ""
    secret = ""
    bybit_api = ByBit()
    tickers = bybit_api.get_ticker_names(5e5)
    kline = bybit_api.get_klines('BTCUSDT', '15m', 1000)
    # t_list = list()
    # for t in tickers:
    #     t_list.append(t['symbol'])
    # exchange_info = binance_api.client.futures_exchange_info()
    # print('1000SHIBUSDT' in t_list)