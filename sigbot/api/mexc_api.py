from datetime import datetime
import requests
import pandas as pd
from api.api_base import ApiBase


class MEXC(ApiBase):
    URL = 'https://api.mexc.com/api/v3'

    def get_ticker_names(self, min_volume) -> (list, list, list):  # ok
        """ Get tickers from spot, futures and swap exchanges and get tickers with big enough 24h volume """
        tickers = pd.DataFrame(requests.get(self.URL + '/ticker/24hr', timeout=3).json())
        tickers = tickers[(tickers['symbol'].str.endswith('USDT')) | (tickers['symbol'].str.endswith('USDC'))]

        all_tickers = tickers['symbol'].to_list()

        tickers['quoteVolume'] = tickers['quoteVolume'].astype(float)
        tickers = tickers[tickers['quoteVolume'] >= min_volume // 2]

        filtered_symbols = self.check_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)]
        filtered_symbols = self.delete_duplicate_symbols(tickers['symbol'])
        tickers = tickers[tickers['symbol'].isin(filtered_symbols)].reset_index(drop=True)

        return tickers['symbol'].to_list(), tickers['volume'].to_list(), all_tickers

    def get_klines(self, symbol, interval, limit=300) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        if interval == '1h':
            interval = '60m'
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        tickers = pd.DataFrame(requests.get(self.URL + '/klines', params=params).json())
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume',
                                  6: 'close_time', 7: 'quote_volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']]

    def get_historical_klines(self, symbol: str, interval: str, limit: int, min_time: datetime) -> pd.DataFrame:
        """ Save historical time, price and volume info to CryptoCurrency structure
            for some period (earlier than min_time) """
        interval_secs = self.convert_interval_to_secs(interval)
        if interval == '1h':
            interval = '60m'
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        tmp_limit = limit
        prev_time, earliest_time = None, datetime.now()
        ts = int(self.get_timestamp() / 3600) * 3600
        tickers = pd.DataFrame()
        
        while earliest_time > min_time:
            start_time = (ts - (tmp_limit * interval_secs)) * 1000
            end_time = (ts - ((tmp_limit - limit) * interval_secs)) * 1000
            params['startTime'] = start_time
            params['endTime'] = end_time
            tmp = pd.DataFrame(requests.get(self.URL + '/klines', params=params).json())
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp[0].min()
            earliest_time = self.convert_timstamp_to_time(earliest_time, unit='ms')
            # prevent endless cycle if there are no candles that earlier than min_time
            if prev_time == earliest_time:
                break
            
            tickers = pd.concat([tmp, tickers])
            tmp_limit += limit

        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'asset_volume',
                                  6: 'close_time', 7: 'volume'}, axis=1)
        tickers = tickers.sort_values('time', ignore_index=True)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)


if __name__ == '__main__':
    mexc = MEXC()
    tickers = mexc.get_ticker_names(5e5)[0]
    min_time = datetime.now().replace(microsecond=0, second=0, minute=0) - pd.to_timedelta(365 * 5, unit='D')
    klines = mexc.get_klines('KWENTAUSDT', '1h', 1000)
    pass


