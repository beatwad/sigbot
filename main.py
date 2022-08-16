import numpy as np
from time import sleep
from views.api import API
from config.config import ConfigFactory
from indicators.indicators import RSI, STOCH, MACD

if __name__ == "__main__":
    # Connect to API
    api = API()
    api.connect_to_api("7arxKITvadhYavxsQr5dZelYK4kzyBGM4rsjDCyJiPzItNlAEdlqOzibV7yVdnNy",
                       "3NvopCGubDjCkF4SzqP9vj9kU2UIhE4Qag9ICUdESOBqY16JGAmfoaUIKJLGDTr4")
    configs = ConfigFactory.factory().configs

    i = 1
    while True:
        print(f'Cycle number {i}')
        # Taking Crypto Currency Values
        crypto_currency = api.get_crypto_currency('BTCUSDT', "1d", 100)
        # Get candle and volume data
        open = np.asarray(crypto_currency.open_values)
        close = np.asarray(crypto_currency.close_values)
        high = np.asarray(crypto_currency.high_values)
        low = np.asarray(crypto_currency.low_values)
        volume = np.asarray(crypto_currency.volume_values)

        # To check some information about the cryptocurrency which we got its values
        print("Symbol: {}\nInterval: {}\nLimit: {}".format(crypto_currency.symbol,
                                                           crypto_currency.interval,
                                                           crypto_currency.limit))
        # RSI Scores
        rsi = RSI(configs)
        stoch = STOCH(configs)
        macd = MACD(configs)
        print(rsi.get_indicator(close), stoch.get_indicator(high, low, close), )

        i += 1
        # sleep(10)
        break







