from datetime import datetime
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
        tickers['turnover_24h'] = tickers['turnover_24h'].astype(float)
        tickers = tickers[tickers['turnover_24h'] >= min_volume // 3]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list(), tickers['volume_24h'].to_list(), all_tickers

    def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        interval_secs = self.convert_interval_to_secs(interval)
        interval = self.convert_interval(interval)

        from_time = (self.get_timestamp() - (limit * interval_secs))
        tickers = pd.DataFrame(self.client.query_kline(symbol=symbol, interval=interval,
                                                           from_time=from_time, limit=200)['result'])
        # at first time get candles from previous interval to overcome API limit restrictions
        if limit > 100:
            from_time = (self.get_timestamp() - (limit * 2 * interval_secs))
            tmp = pd.DataFrame(self.client.query_kline(symbol=symbol, interval=interval,
                                                       from_time=from_time, limit=limit)['result'])
            if tmp.shape[0] > 0:
                tickers = pd.concat([tmp, tickers])

        tickers = tickers.rename({'open_time': 'time'}, axis=1)

        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)

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
            from_time = (ts - (tmp_limit * interval_secs))
            tmp = pd.DataFrame(self.client.query_kline(symbol=symbol, interval=interval,
                                                       from_time=from_time, limit=limit)['result'])
            prev_time, earliest_time = earliest_time, tmp['open_time'].min()
            earliest_time = self.convert_timstamp_to_time(earliest_time, unit='s')
            # prevent endless cycle if there are no candles that eariler than min_time
            if prev_time == earliest_time:
                break
            
            tickers = pd.concat([tmp, tickers])
            tmp_limit += limit

        tickers = tickers.rename({'open_time': 'time'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)


if __name__ == '__main__':
    key = ""
    secret = ""
    bybit_api = ByBitPerpetual()
    tickers = bybit_api.get_ticker_names(1)
    print(tickers)
    # kline = bybit_api.get_klines('10000NFTUSDT', '4h', 1000)
    # print(kline)
