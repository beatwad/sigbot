import pandas as pd
from time import sleep
from os import environ
from data.get_data import DataFactory
from config.config import ConfigFactory
from signals.find_signal import FindSignal
from signal_stat.signal_stat import SignalStat
from indicators.indicators import IndicatorFactory

if __name__ == "__main__":
    debug = False
    # Set environment variable
    environ["ENV"] = "development"
    # Set dataframe dict
    dfs = {'stat': {'buy': pd.DataFrame(columns=['time', 'ticker', 'timeframe']),
                    'sell': pd.DataFrame(columns=['time', 'ticker', 'timeframe'])}}
    # Set list of available exchanges, cryptocurrencies and tickers
    exchanges = {'Binance': {'BTCUSDT': ['5m'], 'ETHUSDT': ['5m']}}
    # exchanges = {'Binance': {'BTCUSDT': ['5m']}}
    # Get configs
    configs = ConfigFactory.factory(environ).configs
    # Get dict of exchange APIs
    exchange_apis = dict()
    for exchange in exchanges:
        exchange_apis[exchange] = DataFactory.factory(exchange, **configs)

    i = 1

    while True:
        # For every exchange, ticker and timeframe in base get cryptocurrency data and write it to correspond dataframe
        for exchange, exchange_api in exchange_apis.items():
            tickers = exchanges[exchange]
            for ticker in tickers:
                timeframes = tickers[ticker]
                for timeframe in timeframes:
                    print(f'Cycle number is {i}, exchange is {exchange}, ticker is {ticker}, timeframe is {timeframe}')
                    # If cryptocurrency dataframe is in dataframe dict - get it, else - create the new one
                    if debug:
                        df = pd.read_pickle('BTCUSDT_5m.pkl')
                    else:
                        df = dfs.get(ticker, dict()).get(timeframe, pd.DataFrame())
                        # Write data to the dataframe
                        df = exchange_api.get_data(df, ticker, timeframe)
                    # Create indicator list from search signal patterns list
                    indicators = list()
                    indicator_list = configs['Indicator_list']
                    for indicator in indicator_list:
                        ind_factory = IndicatorFactory.factory(indicator, configs)
                        if ind_factory:
                            indicators.append(ind_factory)
                    # Write indicators to dataframe
                    df = exchange_api.add_indicator_data(df, indicators, ticker, timeframe)
                    # Update dataframe dict
                    if ticker not in dfs:
                        dfs[ticker] = dict()
                    dfs[ticker][timeframe] = df
                    # Get signal
                    fs = FindSignal(configs)
                    points = fs.find_signal(df)
                    # Write statistics
                    ss = SignalStat()
                    dfs = ss.write_stat(dfs, ticker, timeframe, points)
                    # print(ss.calculate_total_stat(dfs, 'buy'))
                    # print(ss.calculate_total_stat(dfs, 'sell'))
                    # print(ss.calculate_ticker_stat(dfs, 'buy', ticker, timeframe))
                    # print(ss.calculate_ticker_stat(dfs, 'sell', ticker, timeframe))
                    # Save dataframe to the disk
                    try:
                        open(f'{ticker}_{timeframe}.pkl', 'w').close()
                    except FileNotFoundError:
                        pass
                    df.to_pickle(f'{ticker}_{timeframe}.pkl')
        i += 1
        sleep(300)
