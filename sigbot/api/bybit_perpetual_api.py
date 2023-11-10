import pandas as pd
from api.api_base import ApiBase
from pybit import usdt_perpetual


class ByBitPerpetual(ApiBase):
    client = ""

    def __init__(self, api_key="Key", api_secret="Secret"):
        self.global_limit = 1000
        self.api_key = api_key
        self.api_secret = api_secret
        self.mainnet = 'https://api.bybit.com'
        self.testnet = 'https://api-testnet.bybit.com'
        self.connect_to_api('', '')

    def connect_to_api(self, api_key, api_secret):
        self.client = usdt_perpetual.HTTP(self.mainnet, api_key=api_key, api_secret=api_secret)

    def get_ticker_names(self, min_volume) -> (list, list, list):
        tickers = pd.DataFrame(self.client.latest_information_for_symbol()['result'])

        all_tickers = tickers['symbol'].to_list()

        tickers = tickers[(tickers['symbol'].str.endswith('USDT'))]
        # tickers['turnover_24h'] = tickers['turnover_24h'].astype(float)
        # tickers = tickers[tickers['turnover_24h'] >= min_volume]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list(), tickers['volume_24h'].to_list(), all_tickers

    def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        interval_secs = self.convert_interval_to_secs(interval)
        from_time = (self.get_timestamp() - (limit * interval_secs))

        interval = self.convert_interval(interval)
        tickers = pd.DataFrame(self.client.query_kline(symbol=symbol, interval=interval,
                                                       from_time=from_time, limit=limit)['result'])
        tickers = tickers.rename({'open_time': 'time'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']]

    # def get_klines_agg(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    #     """ Save time, price and volume info to CryptoCurrency structure """
    #     limit_mult = math.ceil(self.global_limit / limit)
    #     tickers = None
    #     interval_secs = self.convert_interval_to_secs(interval)
    #     interval = self.convert_interval(interval)
    #     for i in range(limit_mult):
    #         from_time = (self.get_timestamp() - ((self.global_limit - limit * i) * interval_secs))
    #         tmp = pd.DataFrame(self.client.query_kline(symbol=symbol, interval=interval,
    #                                                     from_time=from_time, limit=limit)['result'])
    #         if tickers is None:
    #             tickers = tmp.copy()
    #         else:
    #             tickers = pd.concat([tickers, tmp])
    #     tickers = tickers.rename({'open_time': 'time'}, axis=1)
    #     return tickers[['time', 'open', 'high', 'low', 'close', 'volume']]


if __name__ == '__main__':
    key = ""
    secret = ""
    bybit_api = ByBitPerpetual()
    tickers = bybit_api.get_ticker_names(1)
    print(tickers)
    # kline = bybit_api.get_klines('10000NFTUSDT', '4h', 1000)
    # print(kline)
