from datetime import datetime
import requests
import pandas as pd
from api.api_base import ApiBase


class OKEXSwap(ApiBase):
    URL = 'https://www.okex.com'

    def get_ticker_names(self, min_volume) -> (list, list, list):
        """ Get tickers from spot, futures and swap exchanges and get tickers with big enough 24h volume """
        tickers = pd.DataFrame(requests.get(self.URL +
                                            '/api/v5/market/tickers?instType=SWAP', timeout=3).json()['data'])
        tickers['symbol'] = tickers['instId'].str.replace('-', '').str.replace('SWAP', '')
        all_tickers = tickers['symbol'].to_list()

        tickers = tickers[(tickers['instId'].str.endswith('USDT-SWAP')) | (tickers['instId'].str.endswith('USDC-SWAP'))]
        # meaning of vol24h is different between SPOT and SWAP
        tickers['vol24h'] = tickers['vol24h'].astype(float)
        tickers['last'] = tickers['last'].astype(float)
        ticker_vol = tickers['vol24h'] * tickers['last']
        tickers = tickers[ticker_vol >= min_volume // 2]

        filtered_symbols = self.check_symbols(tickers['instId'])
        tickers = tickers[tickers['instId'].isin(filtered_symbols)]
        filtered_symbols = self.delete_duplicate_symbols(tickers['instId'])
        tickers = tickers[tickers['instId'].isin(filtered_symbols)].reset_index(drop=True)
        tickers = tickers.drop_duplicates(subset=['instId'])

        return tickers['instId'].to_list(), tickers['vol24h'].to_list(), all_tickers

    def get_klines(self, symbol, interval, limit=200) -> pd.DataFrame:
        """ Save time, price and volume info to CryptoCurrency structure """
        interval_secs = self.convert_interval_to_secs(interval)

        if not interval.endswith('m'):
            interval = interval.upper()
        params = {'instId': symbol, 'bar': interval, 'limit': limit}
        tickers = pd.DataFrame(requests.get(self.URL + '/api/v5/market/candles', params=params).json()['data'])
        # at first time get candles from previous interval to overcome API limit restrictions
        if limit > 100:
            after = tickers.iloc[0, 0]
            after = int(after) - (limit - 1) * interval_secs * 1000
            params['after'] = str(after)
            tmp = pd.DataFrame(requests.get(self.URL + '/api/v5/market/candles', params=params).json()['data'])
            if tmp.shape[0] > 0:
                tickers = pd.concat([tickers, tmp])

        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 6: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']][::-1].reset_index(drop=True)
    
    def get_historical_klines(self, symbol: str, interval: str, limit: int, min_time: datetime) -> pd.DataFrame:
        """ Save historical time, price and volume info to CryptoCurrency structure
            for some period (earlier than min_time) """
        # maximum limit for this endpoint is 100
        limit = 100
        interval_secs = self.convert_interval_to_secs(interval)
        if not interval.endswith('m'):
            interval = interval.upper()
        params = {'instId': symbol, 'bar': interval, 'limit': limit}
        tmp_limit = 0
        prev_time, earliest_time = None, datetime.now()
        ts = int(self.get_timestamp() / 3600) * 3600
        tickers = pd.DataFrame()
        
        while earliest_time > min_time:
            after = (ts - tmp_limit * interval_secs) * 1000
            params['after'] = str(after)
            tmp = pd.DataFrame(requests.get(self.URL + '/api/v5/market/history-candles', params=params).json()['data'])
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp[0].min()
            earliest_time = self.convert_timstamp_to_time(int(earliest_time), unit='ms')
            # prevent endless cycle if there are no candles that are earlier than min_time
            if prev_time == earliest_time:
                break
            
            tickers = pd.concat([tickers, tmp])
            tmp_limit += limit

        tickers = tickers.sort_values(0, ignore_index=True)
        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 6: 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)

    def get_historical_funding_rate(self, symbol: str, limit: int, min_time: datetime) -> pd.DataFrame:
        """ Save historical funding rate info to CryptoCurrency structure
            for some period (earlier than min_time) """
        interval_secs = 8 * 3600 * 1000
        before = int(self.get_timestamp() / 3600) * 3600 * 1000
        limit = 100
        params = {'instId': symbol, 'limit': limit}
        prev_time, earliest_time = None, datetime.now()
        funding_rates = pd.DataFrame()

        while earliest_time > min_time:
            before = (before - (limit * interval_secs))
            params['before'] = str(before)
            tmp = pd.DataFrame(requests.get(self.URL + '/api/v5/public/funding-rate-history',
                                            params=params).json()['data'])
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp['fundingTime'].min()
            earliest_time = self.convert_timstamp_to_time(earliest_time, unit='ms')
            # prevent endless cycle if there are no candles that earlier than min_time
            if prev_time == earliest_time:
                break

            funding_rates = pd.concat([funding_rates, tmp])
        funding_rates = funding_rates.rename({'fundingTime': 'time', 'fundingRate': 'funding_rate'}, axis=1)
        return funding_rates[['time', 'funding_rate']][::-1].reset_index(drop=True)


if __name__ == '__main__':
    from datetime import datetime
    okex = OKEXSwap()
    tickers = okex.get_ticker_names(1e5)[0]
    min_time = datetime.now().replace(microsecond=0, second=0, minute=0) - pd.to_timedelta(365 * 5, unit='D')
    funding_rates = okex.get_historical_klines(symbol='BTC-USDT-SWAP', interval='1h', limit=150, min_time=min_time)
    funding_rates['time'] = pd.to_datetime(funding_rates['time'], unit='ms')
    funding_rates['time'] = funding_rates['time'] + pd.to_timedelta(3, unit='h')
    pass


