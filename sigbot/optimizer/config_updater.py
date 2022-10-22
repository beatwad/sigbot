import json
from optimizer import Optimizer


class ConfigUpdater:
    def __init__(self, ttype, timeframe):
        self.ttype = ttype
        self.timeframe = timeframe

    @staticmethod
    def dict_convert(optim_dict):
        def helper(d):
            for key, value in d.items():
                if type(value) == list:
                    d[key] = value[0]
                else:
                    d[key] = helper(value)
            return d

        return helper(optim_dict)

    def config_update(self, optim_dict):
        with open(f'../config/config_{self.timeframe}.json', 'r') as f:
            configs = json.load(f)

        optim_dict = self.dict_convert(optim_dict)

        opt = Optimizer([], optim_dict, **configs)
        confs = opt.save_configs(optim_dict, self.ttype)

        with open(f'../config/config_{self.timeframe}.json', 'w') as f:
            json.dump(confs, f)
