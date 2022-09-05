import requests
import pandas as pd
from api.api_base import ApiBase


class OKEX(ApiBase):
    URL = 'https://www.okex.com'

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
        """ Get tickers from spot, futures and swap OKEX exchanges and get tickers with big enough 24h volume """
        tickers_spot = pd.DataFrame(requests.get(self.URL + '/api/v5/market/tickers?instType=SPOT').json()['data'])
        tickers_future = pd.DataFrame(requests.get(self.URL + '/api/v5/market/tickers?instType=FUTURES').json()['data'])
        tickers_swap = pd.DataFrame(requests.get(self.URL + '/api/v5/market/tickers?instType=SWAP').json()['data'])

        tickers = pd.concat([tickers_spot, tickers_future, tickers_swap])
        tickers['symbol'] = tickers['instId'].str.replace('-', '')
        all_tickers = tickers['symbol'].to_list()

        tickers = tickers[(tickers['instId'].str.endswith('USDT')) | (tickers['instId'].str.endswith('USDC'))]
        tickers['volCcy24h'] = tickers['volCcy24h'].astype(float)
        tickers = tickers[tickers['volCcy24h'] >= min_volume]

        filtered_symbols = self.check_symbols(tickers['instId'])
        tickers = tickers[tickers['instId'].isin(filtered_symbols)]
        filtered_symbols = self.delete_duplicate_symbols(tickers['instId'])
        tickers = tickers[tickers['instId'].isin(filtered_symbols)].reset_index(drop=True)
        tickers = tickers.drop_duplicates(subset=['instId'])

        return tickers['instId'].to_list(), all_tickers

    def get_klines(self, symbol, interval, limit=300) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        if not interval.endswith('m'):
            interval = interval.upper()
        params = {'instId': symbol, 'bar': interval, 'limit': limit}
        tickers = pd.DataFrame(requests.get(self.URL + '/api/v5/market/candles', params=params).json()['data'])
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 6: 'volume'}, axis=1)

        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']]


if __name__ == '__main__':
    from datetime import datetime

    okex = OKEX()
    tickers = okex.get_ticker_names(1e5)
    dt1 = datetime.now()

    for ticker in tickers:
        print(ticker)
        klines1 = okex.get_klines(ticker, '5m')
        klines2 = okex.get_klines(ticker, '3m')

    dt2 = datetime.now()
    dtm, dts = divmod((dt2 - dt1).total_seconds(), 60)
    print(f'Time for the cycle (min:sec) - {int(dtm)}:{round(dts, 2)}')

