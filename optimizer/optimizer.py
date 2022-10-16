import sys
import itertools as it

import pandas as pd
from tqdm.auto import tqdm
from os import environ, remove

sys.path.insert(0, '..')

# Set environment variable
environ["ENV"] = "optimize"

from bot.bot import SigBot
from config.config import ConfigFactory

# Get configs
configs = ConfigFactory.factory(environ).configs


# mock class
class Main:
    def __init__(self):
        self.cycle_number = 1


class Optimizer:
    def __init__(self, pattern, optim_dict, **configs):
        self.statistics = dict()
        self.configs = configs
        self.pattern_list = pattern
        self.optim_dict = optim_dict
        self.remove_path = optim_dict
        self.working_timeframe = configs['Timeframes']['work_timeframe']
        # self.buy_stat_path = f'signal_stat/buy_stat_{self.working_timeframe}.pkl'
        # self.sell_stat_path = f'signal_stat/sell_stat_{self.working_timeframe}.pkl'

    # def clean_prev_stat(self):
    #     """ Clean previous statistics files """
    #     try:
    #         remove(self.buy_stat_path)
    #     except FileNotFoundError:
    #         pass
    #     try:
    #         remove(self.sell_stat_path)
    #     except FileNotFoundError:
    #         pass

    def clean_dict(self, dict1):
        res_dict = dict()
        for key, value in dict1.items():
            if key in self.pattern_list:
                res_dict[key] = value['params']
        for key, value in res_dict.items():
            i, vk = 0, list(value.keys())
            while i < len(vk):
                k, v = vk[i], value[vk[i]]
                if type(v) == str:
                    del vk[i]
                    del value[k]
                else:
                    i += 1
        return res_dict

    def merge_dicts(self, dict1, dict2):
        res_dict1 = self.clean_dict(dict1)
        res_dict2 = self.clean_dict(dict2)

        for key, value in res_dict2.items():
            if key not in res_dict1:
                res_dict1[key] = value
            else:
                res_dict1[key] = {**res_dict1[key], **res_dict2[key]}
        return res_dict1

    def get_product_dicts(self):
        res_dict = {k: v for k, v in self.optim_dict.items() if k in self.pattern_list}
        perm_values = list()
        for key, value in res_dict.items():
            keys, values = zip(*value.items())
            perm_dicts = [dict(zip(keys, v)) for v in it.product(*values)]
            perm_values.append(perm_dicts)
        product_dict = [dict(zip(res_dict.keys(), v)) for v in it.product(*perm_values)]
        return product_dict

    def set_configs(self, prod_dict):
        confs = self.configs.copy()

        for key in confs:
            if key == 'Patterns':
                confs[key] = [self.pattern_list]
            elif key == 'Indicator_list':
                confs[key] = self.pattern_list
            elif key in ['Indicator', 'Indicator_signal']:
                for indicator in prod_dict.keys():
                    prod_values = prod_dict[indicator]
                    conf_values = confs[key][indicator]['params']
                    for k, v in conf_values.items():
                        if k in prod_values:
                            conf_values[k] = prod_values[k]
                    if indicator != 'LinearReg':
                        if 'high_bound' in conf_values:
                            conf_values['high_bound'] = 100 - conf_values['low_bound']
                        elif 'high_price_quantile' in conf_values:
                            conf_values['high_price_quantile'] = 100 - conf_values['low_price_quantile']
        return confs

    @staticmethod
    def get_headers_from_dict(prod_dict: dict) -> list:
        headers = list()

        def helper(prod_dict, header):
            for key in prod_dict:
                if type(prod_dict[key]) != dict:
                    headers.append(header + key)
                else:
                    helper(prod_dict[key], header + key + '_')

        helper(prod_dict, '')
        return headers

    @staticmethod
    def get_values_from_dict(prod_dict: dict) -> list:
        headers = list()

        def helper(prod_dict):
            for key in prod_dict:
                if type(prod_dict[key]) != dict:
                    headers.append(prod_dict[key])
                else:
                    helper(prod_dict[key])

        helper(prod_dict)
        return headers

    def optimize(self, pattern, ttype, opt_limit, load):
        main = Main()
        # set pattern string
        pattern = '_'.join(pattern)
        # get list of config dicts with all possible combinations of pattern settings
        product_dicts = self.get_product_dicts()
        print(f'Number of combinations is {len(product_dicts)}')
        # get pattern headers
        headers = self.get_headers_from_dict(product_dicts[0])
        result_statistics = None
        # flag that helps to prevent not necessary exchange data loading
        load_tickers, exchanges, opt_dfs = True, None, None
        # if load flag set to True - load fresh data from exchanges, else get data from dist
        for prod_dict in tqdm(product_dicts):
            # load data
            confs = self.set_configs(prod_dict)
            sb = SigBot(main, load_tickers=load_tickers, **confs)
            # load ticker data from exchange only at first time
            if load_tickers:
                exchanges = sb.exchanges
                opt_dfs = sb.opt_dfs
                load_tickers = False
            else:
                sb.opt_dfs = opt_dfs
                sb.exchanges = exchanges
            # load candle data from exchanges only at first time
            if load:
                sb.save_opt_dataframes(load)
                load = False
            sb.save_opt_statistics(opt_limit)
            # calculate statistic
            rs, fn = sb.stat.calculate_total_stat(sb.database, ttype, pattern)
            # create df to store statistics results
            tmp = pd.DataFrame(columns=['pattern'] + headers +
                                       [f'pct_right_forecast_{lag + 1}' for lag in range(24)] +
                                       [f'pct_price_diff_{lag + 1}' for lag in range(24)] + ['forecasts_num'])
            tmp['pattern'] = [pattern]
            tmp[headers] = self.get_values_from_dict(prod_dict)
            tmp[[f'pct_right_forecast_{lag + 1}' for lag in range(24)]] = [r[0] for r in rs]
            tmp[[f'pct_price_diff_{lag + 1}' for lag in range(24)]] = [r[1] for r in rs]
            tmp['forecasts_num'] = fn
            # add temp df to the result df
            if result_statistics is None:
                result_statistics = tmp.copy()
            else:
                result_statistics = pd.concat([result_statistics, tmp])
            result_statistics = result_statistics.reset_index(drop=True)

        return result_statistics


if __name__ == '__main__':
    ttype = 'buy'
    pattern = ['STOCH', 'RSI', 'LinearReg']
    opt_limit = 100
    load = False

    optim_dict = {'RSI': {'timeperiod': [14], 'low_bound': [25, 30, 35]},
                  'STOCH': {'fastk_period': [9], 'slowk_period': [2, 3, 4],
                            'slowd_period': [3, 5, 7], 'low_bound': [10, 15, 20]},
                  'LinearReg': {'timeperiod': [16, 20, 24, 28, 32], 'low_bound': [0]}}

    opt = Optimizer(pattern, optim_dict, **configs)
    # opt.clean_prev_stat()
    rs = opt.optimize(pattern, ttype, opt_limit, load)
