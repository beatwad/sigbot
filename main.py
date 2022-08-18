import pandas as pd
from time import sleep
from os import environ
from data.get_data import DataFactory
from config.config import ConfigFactory
from indicators.indicators import IndicatorFactory

if __name__ == "__main__":
    # Set environment variable
    environ["ENV"] = "development"
    # Set dataframe dict
    cc_dfs = dict()
    # Set list of available exchanges, cryptocurrencies and tickers
    exchanges = {'Binance': {'BTCUSDT': ['5m']}}
    # Get configs
    configs = ConfigFactory.factory(environ).configs

    i = 1

    while True:
        # For every exchange, ticker and timeframe in base get cryptocurrency data and write it to correspond dataframe
        for exchange in exchanges:
            exchange_api = DataFactory.factory(exchange)
            tickers = exchanges[exchange]
            for ticker in tickers:
                timeframes = tickers[ticker]
                for timeframe in timeframes:
                    print(f'Cycle number is {i}, exchange is {exchange}, ticker is {ticker}, timeframe is {timeframe}')
                    # If cryptocurrency dataframe is in dataframe dict - get it, else create new
                    cc_df = cc_dfs.get(f'{ticker}_{timeframe}', pd.DataFrame())
                    # If dataframe is empty - get all available data to fill it,
                    # else - just get necessary for update data
                    if cc_df.shape == (0, 0):
                        interval = configs['Intervals']['creation_interval']
                    else:
                        interval = configs['Intervals']['update_interval']
                    # Write data to the dataframe
                    cc_df = exchange_api.get_data(cc_df, ticker, timeframe, interval)
                    # Create indicator list
                    indicator_list = configs['Indicator_list']
                    indicators = list()
                    for indicator in indicator_list:
                        indicators.append(IndicatorFactory.factory(indicator, configs))
                    # Write indicators to dataframe
                    cc_df = exchange_api.add_indicator_data(cc_df, indicators, ticker, timeframe)
                    # Save dataframe to the disk
                    cc_df.to_pickle('cc_df.pkl')
                    # Update dataframe dict
                    cc_dfs[f'{ticker}_{timeframe}'] = cc_df
                    i += 1
                    sleep(600)
