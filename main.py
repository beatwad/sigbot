import numpy as np
import pandas as pd
from time import sleep
from views.api import API
from config.config import ConfigFactory
from indicators.indicators import RSI, STOCH, MACD

if __name__ == "__main__":
    # Connect to API
    cc_df = pd.DataFrame()
    tickers = {'BTCUSDT': ['5m']}
    api = API()
    api.connect_to_api("7arxKITvadhYavxsQr5dZelYK4kzyBGM4rsjDCyJiPzItNlAEdlqOzibV7yVdnNy",
                       "3NvopCGubDjCkF4SzqP9vj9kU2UIhE4Qag9ICUdESOBqY16JGAmfoaUIKJLGDTr4")
    configs = ConfigFactory.factory().configs

    i = 1
    for ticker in tickers:
        print(f'Ticker is {ticker}')
        # Taking Crypto Currency Values
        timeframes = tickers[ticker]
        for timeframe in timeframes:
            crypto_currency = api.get_crypto_currency(ticker, timeframe, 500)
            # Save candle and volume data id Pandas Dataframe
            cc_df[f'{ticker}_{timeframe}_time'] = np.asarray(crypto_currency.time)
            cc_df[f'{ticker}_{timeframe}_open'] = np.asarray(crypto_currency.open_values)
            cc_df[f'{ticker}_{timeframe}_close'] = np.asarray(crypto_currency.close_values)
            cc_df[f'{ticker}_{timeframe}_high'] = np.asarray(crypto_currency.high_values)
            cc_df[f'{ticker}_{timeframe}_low'] = np.asarray(crypto_currency.low_values)
            cc_df[f'{ticker}_{timeframe}_volume'] = np.asarray(crypto_currency.volume_values)

            # To check some information about the cryptocurrency which we got its values
            print("Symbol: {}\nInterval: {}\nLimit: {}".format(crypto_currency.symbol,
                                                               crypto_currency.interval,
                                                               crypto_currency.limit))
            # RSI Scores
            rsi = RSI(configs)
            stoch = STOCH(configs)
            macd = MACD(configs)
            cc_df[f'{ticker}_{timeframe}_rsi'] = rsi.get_indicator(cc_df, ticker, timeframe)
            cc_df[f'{ticker}_{timeframe}_stoch_slowk'], cc_df[f'{ticker}_{timeframe}_stoch_slowd'] =\
                stoch.get_indicator(cc_df, ticker, timeframe)
            cc_df.to_pickle('cc_df.pkl')
            i += 1
            # sleep(10)
            break







