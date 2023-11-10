import requests
import pandas as pd
from api.api_base import ApiBase


class MEXCFutures(ApiBase):
    URL = 'https://contract.mexc.com/api/v1/contract/'
    interval_dict = {'1m': 'Min1', '5m': 'Min5', '15m': 'Min15', '30m': 'Min30', '1h': 'Min60', '4h': 'Hour4',
                     '8h': 'Hour8', '1d': 'Day1', '1w': 'Week1'}

    def get_ticker_names(self, min_volume) -> (list, list, list):  # ok
        """ Get tickers from spot, futures and swap OKEX exchanges and get tickers with big enough 24h volume """
        tickers = pd.DataFrame(requests.get(self.URL + '/ticker', timeout=3).json()['data'])
        tickers = tickers[tickers['symbol'].str.endswith('USDT')]

        all_tickers = tickers['symbol'].str.replace('_', '').to_list()

        # tickers['volume24'] = tickers['volume24'].astype(float)
        # tickers = tickers[tickers['volume24'] >= min_volume]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list(), tickers['volume24'].to_list(), all_tickers

    def get_klines(self, symbol, interval, limit=300) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        interval_secs = self.convert_interval_to_secs(interval)
        start = (self.get_timestamp() - (limit * interval_secs))
        interval = self.interval_dict[interval]

        params = {'interval': interval, 'start': start}
        tickers = pd.DataFrame(requests.get(self.URL + f'/kline/{symbol}', params=params).json()['data'])
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'asset_volume',
                                  6: 'close_time', 'vol': 'volume'}, axis=1)
        tickers = tickers.sort_values('time', ignore_index=True)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']]


if __name__ == '__main__':
    mexc = MEXCFutures()
    tickers = mexc.get_ticker_names(1e6)[0]

    for ticker in tickers:
        klines1 = mexc.get_klines('LTC_USDT', '5m')
        klines2 = mexc.get_klines('LTC_USDT', '1h')


