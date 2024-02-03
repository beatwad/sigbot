import re
import pandas as pd
from abc import ABCMeta
from datetime import datetime


class ApiBase(metaclass=ABCMeta):
    @staticmethod
    def delete_duplicate_symbols(symbols) -> list:
        """ If for pair with USDT exists pair with USDC - delete it  """
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

    @staticmethod
    def check_symbols(symbols: list) -> list:
        """ Check if ticker is not pair with fiat currency or stablecoin or ticker is not a leverage type """
        filtered_symbols = list()
        for symbol in symbols:
            if (symbol.startswith('USD') or symbol.startswith('BUSD') or symbol.startswith('TUSDUS')
                    or symbol.startswith('BTCDOM') or symbol.startswith('BSCYFI')):
                continue
            if (symbol.endswith('USD') and symbol[-4] != 'B') or symbol.endswith('UST'):
                continue
            if re.match(r'.+[23][LS]', symbol) or re.match(r'.+UP-?(BUSD|USD[TC])', symbol) or \
                    re.match(r'.+DOWN-?(BUSD|USD[TC])', symbol):
                continue
            fiat = ['EUR', 'CHF', 'GBP', 'JPY', 'CNY', 'RUB', 'AUD', 'DAI']
            for f in fiat:
                if symbol.startswith(f) and (len(symbol) == 7 or symbol[3] == '-'):
                    break
            else:
                filtered_symbols.append(symbol)
        return filtered_symbols

    @staticmethod
    def get_timestamp():
        today_now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        dt = datetime.strptime(today_now, '%Y-%m-%d %H:%M:%S')
        ts = int(dt.timestamp())
        return ts

    @staticmethod
    def convert_interval_to_secs(interval: str) -> int:
        if interval[-1] == 'h':
            interval = int(interval[:-1]) * 60 * 60
        elif interval[-1] == 'd':
            interval = int(interval[:-1]) * 60 * 60 * 24
        elif interval[-1] == 'w':
            interval = int(interval[:-1]) * 60 * 60 * 24 * 7
        else:
            interval = int(interval[:-1]) * 60
        return interval

    @staticmethod
    def convert_interval(interval: str) -> str:
        if interval[-1] == 'h':
            interval = str(int(interval[:-1]) * 60)
        elif interval[-1] == 'd':
            interval = 'D'
        elif interval[-1] == 'w':
            interval = 'W'
        else:
            interval = interval[:-1]
        return interval
    
    @staticmethod
    def convert_timstamp_to_time(timestamp: int, unit: str):
        time = pd.to_datetime(timestamp, unit=unit)
        time += pd.to_timedelta(3, unit='h')
        return time

    def get_historical_funding_rate(self, symbol: str, limit: int, min_time: datetime) -> pd.DataFrame:
        """ Save historical funding rate info to CryptoCurrency structure
            for some period (earlier than min_time) """
        return pd.DataFrame(columns=['time', 'funding_rate'])


if __name__ == '__main__':
    symbol = 'TRXUPUSDT'
    print(re.match(r'.+UP-?(BUSD|USD[TC])', symbol))
