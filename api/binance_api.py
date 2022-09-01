import re
import pandas as pd
from binance.client import Client
from api.api_base import ApiBase
from models.cryptocurrency import CryptoCurrency


class Binance(ApiBase):
    client = ""

    def __init__(self, api_key="Key", api_secret="Secret"):
        if api_key != "Key" and api_secret != "Secret":
            Binance().connect_to_api(api_key, api_secret)
        else:
            self.api_key = api_key
            self.api_secret = api_secret

    def connect_to_api(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = Client(self.api_key, api_secret)

    def get_api_key(self):
        print("Your Api Key: {}".format(self.api_key))

    def get_api_secret(self):
        print("Your Api Secret: {}".format(self.api_secret))

    def get_exchange_info(self):
        return self.client.get_exchange_info()

    def get_ticker_names(self) -> list:
        """ Get non-stable tickers from Binance exchange that are in pair with USDT or BUSD and have TRADING status """
        ticker_names = list()
        exchange_info = self.client.get_exchange_info()

        for s in exchange_info['symbols']:
            symbol = s['symbol']
            if s['status'] == 'TRADING' and symbol.endswith('USDT'):
                if not (re.match('.?USD', symbol) or re.match('.?UST', symbol)):
                    ticker_names.append(s['symbol'])

        for s in exchange_info['symbols']:
            symbol = s['symbol']
            if s['status'] == 'TRADING' and symbol.endswith('BUSD'):
                if not (re.match('.?USD', symbol) or re.match('.?UST', symbol)):
                    prefix = s['symbol'][:-4]
                    if prefix + 'USDT' not in ticker_names:
                        ticker_names.append(s['symbol'])
        return ticker_names

    def get_ticker_volume(self, ticker_names: list) -> pd.DataFrame:
        """ Get 24h volume for each ticker from ticker list, save to pd.Dataframe """
        tickers = self.client.get_ticker()
        ticker_vol_dict = dict()

        for t in tickers:
            symbol = t['symbol']
            if symbol in ticker_names:
                ticker_vol_dict[symbol] = t['quoteVolume']

        data = {'ticker': list(ticker_vol_dict.keys()), 'volume': list(ticker_vol_dict.values())}
        df = pd.DataFrame(data=data)
        df['volume'] = df['volume'].astype(float)

        return df

    def get_crypto_currency(self, symbol, interval, limit) -> CryptoCurrency:
        """ Save time, price and volume info to CryptoCurrency structure """
        klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        crypto_currency = CryptoCurrency(symbol, interval, limit)
        for kline in klines:
            crypto_currency.time.append(float(kline[0]))
            crypto_currency.open_values.append(float(kline[1]))
            crypto_currency.high_values.append(float(kline[2]))
            crypto_currency.low_values.append(float(kline[3]))
            crypto_currency.close_values.append(float(kline[4]))
            crypto_currency.volume_values.append(float(kline[5]))

        return crypto_currency
