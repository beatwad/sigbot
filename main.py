import pandas as pd
from time import sleep
from datetime import datetime
from os import environ
from data.get_data import GetData
from data.get_data import DataFactory
from config.config import ConfigFactory
from signals.find_signal import FindSignal
from signal_stat.signal_stat import SignalStat
from visualizer.visualizer import Visualizer
from indicators.indicators import IndicatorFactory

if __name__ == "__main__":
    debug = True
    # Set environment variable
    environ["ENV"] = "development"
    # Set dataframe dict
    dfs = {'stat': {'buy': pd.DataFrame(columns=['time', 'ticker', 'timeframe']),
                    'sell': pd.DataFrame(columns=['time', 'ticker', 'timeframe'])}}

    # Get configs
    configs = ConfigFactory.factory(environ).configs
    # Set list of available exchanges, cryptocurrencies and tickers
    exchanges = {'Binance': {'API': GetData(**configs), 'tickers': []}}

    # Get timeframes timeframe from which we take levels
    work_timeframe = configs['Timeframes']['work_timeframe']
    higher_timeframe = configs['Timeframes']['higher_timeframe']
    timeframes = [higher_timeframe, work_timeframe]

    # Get API and ticker list for every exchange in list
    for ex in list(exchanges.keys()):
        exchange_api = DataFactory.factory(ex, **configs)
        exchanges[ex]['API'] = exchange_api
        tickers = exchanges[ex]['API'].get_tickers()
        exchanges[ex]['tickers'] = ['BTCUSDT']#tickers

    # Counter
    i = 1
    print(f'Begin, time is {datetime.now()}')

    while True:
        # For every exchange, ticker and timeframe in base get cryptocurrency data and write it to correspond dataframe
        for exchange, exchange_data in exchanges.items():
            exchange_api = exchange_data['API']
            tickers = exchange_data['tickers']
            for ticker in tickers:
                for timeframe in timeframes:
                    print(f'Cycle number is {i}, exchange is {exchange}, ticker is {ticker}, timeframe is {timeframe}')
                    if debug:
                        df = pd.read_pickle('tests/test_ETHUSDT_5m.pkl')
                        new_data_flag = True
                        if i > 1:
                            limit = 10
                        else:
                            limit = 1000
                    else:
                        # If cryptocurrency dataframe is in dataframe dict - get it, else - create the new one
                        df = dfs.get(ticker, dict()).get(timeframe, dict()).get('data', pd.DataFrame())
                        # Write data to the dataframe
                        df, limit, new_data_flag = exchange_api.get_data(df, ticker, timeframe)
                    # Create indicator list from search signal patterns list, if has new data and
                    # data not from higher timeframe, else get only levels
                    if new_data_flag:
                        indicators = list()
                        if timeframe == work_timeframe:
                            indicator_list = configs['Indicator_list']
                        else:
                            indicator_list = ['SUP_RES']
                        for indicator in indicator_list:
                            ind_factory = IndicatorFactory.factory(indicator, configs)
                            if ind_factory:
                                indicators.append(ind_factory)
                        # Write indicators to dataframe, update dataframe dict
                        dfs, df = exchange_api.add_indicator_data(dfs, df, indicators, ticker, timeframe, configs)
                        # Get signals
                        if timeframe == work_timeframe:
                            fs = FindSignal(configs)
                            levels = dfs[ticker][timeframe]['levels']
                            points = fs.find_signal(df, levels, limit)
                            # print(points)
                            # Write statistics
                            ss = SignalStat(**configs)
                            dfs = ss.write_stat(dfs, ticker, timeframe, points)
                            # Generate and save the plot
                            # v = Visualizer(**configs)
                            # filename1 = v.create_plot(dfs, 'ETHUSDT', '5m', points[0], levels)
                            # filename2 = v.create_plot(dfs, 'ETHUSDT', '5m', points[1], levels)
                            # Calculate statistics
                            # print(ss.calculate_total_stat(dfs, 'buy'))
                            # print(ss.calculate_total_stat(dfs, 'sell'))
                            # print(ss.calculate_ticker_stat(dfs, 'buy', ticker, timeframe))
                            # print(ss.calculate_ticker_stat(dfs, 'sell', ticker, timeframe))
                            # Save dataframe to the disk
                        # try:
                        #     open(f'{ticker}_{timeframe}.pkl', 'w').close()
                        # except FileNotFoundError:
                        #     pass
                        # df.to_pickle(f'{ticker}_{timeframe}.pkl')
        i += 1
        sleep(10)
        # print(f'End, time is {datetime.now()}')
        # break
