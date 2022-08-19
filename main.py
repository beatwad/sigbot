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
    dfs = dict()
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
                    # If all cryptocurrencies signal stat is in dataframe dict - get it, else create new
                    stat = dfs.get('stat', pd.DataFrame())
                    # If cryptocurrency dataframe and it's signal stat is in dataframe dict - get it,
                    # else - create the new ones
                    df = dfs.get(f'{ticker}_{timeframe}', pd.DataFrame())
                    # If dataframe is empty - get all available data to fill it,
                    # else - just get necessary for update data
                    if df.shape == (0, 0):
                        interval = configs['Interval']['creation_interval']
                    else:
                        interval = configs['Interval']['update_interval']
                    # Write data to the dataframe
                    df = exchange_api.get_data(df, ticker, timeframe, interval)
                    # Create indicator list
                    indicator_list = configs['Indicator_list']
                    indicators = list()
                    for indicator in indicator_list:
                        indicators.append(IndicatorFactory.factory(indicator, configs))
                    # Write indicators to dataframe
                    df = exchange_api.add_indicator_data(df, indicators, ticker, timeframe)
                    # Save dataframe to the disk
                    df.to_pickle('df.pkl')
                    # Update dataframe dict
                    dfs[f'{ticker}_{timeframe}'] = df
                    i += 1
                    sleep(10)
