import sys
import itertools as it

import pandas as pd
from tqdm.auto import tqdm
from os import environ

sys.path.insert(0, '..')

# Set environment variable
environ["ENV"] = "1h_1d"

from bot.bot import SigBot
from config.config import ConfigFactory

# Get configs
configs = ConfigFactory.factory(environ).configs


# mock class
class Main:
    def __init__(self):
        self.cycle_number = 2


class Optimizer:
    def __init__(self, pattern, optim_dict, **configs):
        self.statistics = dict()
        self.configs = configs
        self.pattern = pattern
        self.optim_dict = optim_dict

    def clean_dict(self, dict1):
        res_dict = dict()
        for key, value in dict1.items():
            if key in self.pattern:
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
        res_dict = {k: v for k, v in self.optim_dict.items() if k in self.pattern}
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
            if key in ['Indicator', 'Indicator_signal']:
                for indicator in prod_dict.keys():
                    prod_values = prod_dict[indicator]
                    conf_values = confs[key][indicator]['params']
                    for k, v in conf_values.items():
                        if k in prod_values:
                            conf_values[k] = prod_values[k]
                    if 'high_bound' in conf_values:
                        conf_values['high_bound'] = 100 - conf_values['low_bound']
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

    def optimize(self, pattern, ttype, load):
        main = Main()
        # get list of config dicts with all possible combinations of pattern settings
        product_dicts = self.get_product_dicts()
        print(f'Number of combinations is {len(product_dicts)}')
        # get pattern headers
        headers = self.get_headers_from_dict(product_dicts[0])
        result_statistics = None
        # if load flag set to True - load fresh data from exchanges, else get data from dist
        for prod_dict in tqdm(product_dicts):
            # load data
            confs = self.set_configs(prod_dict)
            sb = SigBot(main, **confs)
            if load:
                sb.save_opt_dataframes(load)
                load = False
            sb.save_opt_statistics()
            # calculate statistic
            tmp_pattern = [[p] + ['()'] for p in pattern]
            rs, fn = sb.stat.calculate_total_stat(sb.database, ttype, tmp_pattern)
            # create df to store statistics results
            tmp = pd.DataFrame(columns=['pattern'] + headers +
                                       [f'pct_right_forecast_{lag + 1}' for lag in range(24)] +
                                       [f'pct_price_diff_{lag + 1}' for lag in range(24)] + ['forecasts_num'])
            tmp['pattern'] = ['_'.join(p for p in pattern)]
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
            result_statistics.to_pickle(f"opt_{'_'.join(p for p in pattern)}_{ttype}.pkl")
        return result_statistics


if __name__ == '__main__':
    pattern = ['STOCH', 'RSI']
    optim_dict = {'RSI': {'timeperiod': [12, 14, 16], 'low_bound': [25, 30, 35]},
                  'STOCH': {'fastk_period': [9, 14], 'slowk_period': [2, 3, 4],
                            'slowd_period': [3, 5, 7], 'low_bound': [10, 15, 20]}}

    opt = Optimizer(pattern, optim_dict, **configs)

    load = True
    rs = opt.optimize(pattern, 'sell', load)




