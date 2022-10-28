import pandas as pd
from api.api_base import ApiBase
from pybit import usdt_perpetual
from datetime import datetime


class ByBit(ApiBase):
    client = ""

    def __init__(self, api_key="Key", api_secret="Secret"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.mainnet = 'https://api.bybit.com'
        self.testnet = 'https://api-testnet.bybit.com'
        self.connect_to_api('', '')

    def connect_to_api(self, api_key, api_secret):
        self.client = usdt_perpetual.HTTP(self.mainnet, api_key=api_key, api_secret=api_secret)

    @staticmethod
    def get_timestamp():
        today_now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        dt = datetime.strptime(today_now, '%Y-%m-%d %H:%M:%S')
        in_secods_now = int(dt.timestamp())
        return in_secods_now

    def get_ticker_names(self, min_volume) -> (list, list):
        tickers = pd.DataFrame(self.client.latest_information_for_symbol()['result'])

        all_tickers = tickers['symbol'].to_list()

        tickers = tickers[(tickers['symbol'].str.endswith('USDT'))]
        tickers['volume_24h'] = tickers.loc[:, 'volume_24h'].astype(float)
        tickers = tickers[tickers['volume_24h'] >= min_volume]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list(), all_tickers

    def get_klines(self, symbol, interval, limit) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        from_time = (self.get_timestamp() - (limit * int(interval[:-1])) * 60)
        tickers = pd.DataFrame(self.client.query_kline(symbol=symbol, interval=interval[:-1],
                                                       from_time=from_time, limit=limit)['result'])
        tickers = tickers.rename({'open_time': 'time'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']].astype(float)


if __name__ == '__main__':
    key = ""
    secret = ""
    bybit_api = ByBit()
    tickers = bybit_api.get_ticker_names(1e6)
    kline = bybit_api.get_klines('1000BTTUSDT', '5m', 1000)
