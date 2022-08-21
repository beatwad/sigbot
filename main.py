import pandas as pd
from time import sleep
from os import environ
from data.get_data import DataFactory
from config.config import ConfigFactory
from signals.find_signal import FindSignal
from signal_stat.signal_stat import SignalStat
from indicators.indicators import IndicatorFactory

if __name__ == "__main__":
    debug = True
    # Set environment variable
    environ["ENV"] = "development"
    # Set dataframe dict
    dfs = {'stat': {'buy': pd.DataFrame(columns=['time', 'ticker', 'timeframe']),
                    'sell': pd.DataFrame(columns=['time', 'ticker', 'timeframe'])}}
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
                    # # If all cryptocurrencies signal stat is in dataframe dict - get it, else create new
                    # stat = dfs.get('stat', pd.DataFrame())
                    # If cryptocurrency dataframe and it's signal stat is in dataframe dict - get it,
                    # else - create the new ones
                    df = dfs.get(ticker, dict()).get(timeframe, pd.DataFrame())
                    # If dataframe is empty - get all available data to fill it,
                    # else - just get necessary for update data
                    if df.shape == (0, 0):
                        interval = configs['Interval']['creation_interval']
                    else:
                        interval = configs['Interval']['update_interval']
                    # Write data to the dataframe
                    if debug:
                        df = pd.read_pickle('debug.pkl')
                    else:
                        df = exchange_api.get_data(df, ticker, timeframe, interval)
                    # Create indicator list
                    indicator_list = configs['Indicator_list']
                    indicators = list()
                    for indicator in indicator_list:
                        indicators.append(IndicatorFactory.factory(indicator, configs))
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
                    ss.calculate_total_stat(dfs, 'buy')
                    # ss.calculate_ticker_stat(dfs, ticker, timeframe)
                    # Save dataframe to the disk
                    try:
                        open('df.pkl', 'w').close()
                    except FileNotFoundError:
                        pass
                    df.to_pickle('df.pkl')
                    i += 1
                    sleep(10)
