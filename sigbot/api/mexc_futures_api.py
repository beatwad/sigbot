from datetime import datetime
import requests
import pandas as pd
from api.api_base import ApiBase


class MEXCFutures(ApiBase):
    URL = 'https://contract.mexc.com/api/v1/contract/'
    interval_dict = {'1m': 'Min1', '5m': 'Min5', '15m': 'Min15', '30m': 'Min30', '1h': 'Min60', '4h': 'Hour4',
                     '8h': 'Hour8', '1d': 'Day1', '1w': 'Week1'}

    def get_ticker_names(self, min_volume) -> (list, list, list):  # ok
        """ Get tickers from spot, futures and swap exchanges and get tickers with big enough 24h volume """
        tickers = pd.DataFrame(requests.get(self.URL + '/ticker', timeout=3).json()['data'])
        tickers = tickers[tickers['symbol'].str.endswith('USDT')]

        all_tickers = tickers['symbol'].str.replace('_', '').to_list()

        tickers['amount24'] = tickers['amount24'].astype(float)
        tickers = tickers[tickers['amount24'] >= min_volume // 2]

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
        tickers = tickers.rename({'vol': 'volume'}, axis=1)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']]

    def get_historical_klines(self, symbol: str, interval: str, limit: int, min_time: datetime) -> pd.DataFrame:
        """ Save historical time, price and volume info to CryptoCurrency structure
            for some period (earlier than min_time) """
        interval_secs = self.convert_interval_to_secs(interval)
        interval = self.interval_dict[interval]
        params = {'interval': interval, 'limit': limit}
        tmp_limit = limit
        prev_time, earliest_time = None, datetime.now()
        ts = int(self.get_timestamp() / 3600) * 3600
        tickers = pd.DataFrame()
        
        while earliest_time > min_time:
            start_time = (ts - (tmp_limit * interval_secs))
            end_time = (ts - ((tmp_limit - limit) * interval_secs))
            params['start'] = start_time
            params['end'] = end_time
            tmp = pd.DataFrame(requests.get(self.URL + f'/kline/{symbol}', params=params).json()['data'])
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp['time'].min()
            earliest_time = self.convert_timstamp_to_time(earliest_time, unit='s')
            # prevent endless cycle if there are no candles that earlier than min_time
            if prev_time == earliest_time:
                break
            
            tickers = pd.concat([tmp, tickers])
            tmp_limit += limit

        tickers = tickers.rename({0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'asset_volume',
                                  6: 'close_time', 'vol': 'volume'}, axis=1)
        tickers = tickers.sort_values('time', ignore_index=True)
        return tickers[['time', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)

    def get_historical_funding_rate(self, symbol: str, limit: int, min_time: datetime) -> pd.DataFrame:
        """ Save historical funding rate info to CryptoCurrency structure
            for some period (earlier than min_time) """
        params = {'symbol': symbol, 'page_num': 1, 'page_size': 100}
        prev_time, earliest_time = None, datetime.now()
        funding_rates = pd.DataFrame()

        while earliest_time > min_time:
            tmp = pd.DataFrame(requests.get(self.URL + f'/funding_rate/history',
                                            params=params).json()['data']['resultList'])
            if tmp.shape[0] == 0:
                break
            prev_time, earliest_time = earliest_time, tmp['settleTime'].min()
            earliest_time = self.convert_timstamp_to_time(earliest_time, unit='ms')
            # prevent endless cycle if there are no candles that earlier than min_time
            if prev_time == earliest_time:
                break
            funding_rates = pd.concat([funding_rates, tmp])
            params['page_num'] += 1
        funding_rates = funding_rates.rename({'settleTime': 'time', 'fundingRate': 'funding_rate'}, axis=1)
        return funding_rates[['time', 'funding_rate']][::-1].reset_index(drop=True)


if __name__ == '__main__':
    mexc = MEXCFutures()
    min_time = datetime.now().replace(microsecond=0, second=0, minute=0) - pd.to_timedelta(365 * 5, unit='D')
    funding_rates = mexc.get_historical_funding_rate('BTC_USDT', limit=300, min_time=min_time)
    funding_rates['time'] = pd.to_datetime(funding_rates['time'], unit='ms')
    funding_rates['time'] = funding_rates['time'] + pd.to_timedelta(3, unit='h')
    pass


