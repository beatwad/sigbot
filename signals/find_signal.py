import numpy as np
from abc import abstractmethod


class SignalFactory(object):
    """ Return indicator according to 'indicator' variable value """
    @staticmethod
    def factory(indicator, params):
        if indicator == 'RSI':
            return RSISignal(**params)
        elif indicator == 'STOCH':
            return STOCHSignal(**params)
        elif indicator == 'MACD':
            return MACDSignal(**params)


class SignalBase:
    type = 'Indicator_signal'
    name = 'Base'

    def __init__(self, params):
        self.params = params[self.type][self.name]['params']

    @abstractmethod
    def find_signal(self, *args, **kwargs):
        pass


class STOCHSignal(SignalBase):
    name = 'STOCH'

    def __init__(self, **params):
        super(STOCHSignal, self).__init__(params)
        self.lower_bound = self.params.get('lower_bound', 20)
        self.upper_bound = self.params.get('lower_bound', 80)

    def find_signal(self, df, ticker, timeframe):
        """ Return signal if RSI is in overbuy/oversell zone, slowk and slowd lines have crossed
            and their direction is down/up """
        points = list()

        slowk = df[f'{ticker}_{timeframe}_stoch_slowk'].values
        slowk_shift = df[f'{ticker}_{timeframe}_stoch_slowk'].shift(1)
        slowd = df[f'{ticker}_{timeframe}_stoch_slowd'].values
        slowd_shift = df[f'{ticker}_{timeframe}_stoch_slowd'].shift(1)

        df['stoch_slowk_dir'] = (slowk - slowk_shift) / np.maximum(np.abs(slowk), np.abs(slowd_shift))
        df['stoch_slowd_dir'] = (slowd - slowd_shift) / np.maximum(np.abs(slowd), np.abs(slowd_shift))
        df['stoch_diff'] = slowk - slowd

        for i, row in df.iterrows():
            if i > 2 and row[f'{ticker}_{timeframe}_stoch_slowk'] <= self.lower_bound \
                    and row[f'{ticker}_{timeframe}_stoch_slowd'] <= self.lower_bound:
                if row['stoch_diff'] > 0 and df.loc[i - 1]['stoch_diff'] > 0 and df.loc[i - 2]['stoch_diff'] > 0:
                    if df.loc[i - 3]['stoch_diff'] < 0 and df.loc[i - 4]['stoch_diff'] < 0 and\
                            df.loc[i - 5]['stoch_diff'] < 0:
                        if row['stoch_slowk_dir'] > 0 and row['stoch_slowd_dir'] > 0:
                            points.append([row[f'{ticker}_{timeframe}_stoch_slowk'], i])
            elif i > 2 and row[f'{ticker}_{timeframe}_stoch_slowk'] >= self.upper_bound \
                    and row[f'{ticker}_{timeframe}_stoch_slowd'] >= self.upper_bound:
                if row['stoch_diff'] < 0 and df.loc[i - 1]['stoch_diff'] < 0 and df.loc[i - 2]['stoch_diff'] < 0:
                    if df.loc[i - 3]['stoch_diff'] > 0 and df.loc[i - 4]['stoch_diff'] > 0 and \
                            df.loc[i - 5]['stoch_diff'] > 0:
                        if row['stoch_slowk_dir'] < 0 and row['stoch_slowd_dir'] < 0:
                            points.append([row[f'{ticker}_{timeframe}_stoch_slowk'], i])
                    points.append([row[f'{ticker}_{timeframe}_stoch_slowk'], i])
        return True


class RSISignal(SignalBase):
    signal_name = "STOCH"
    signal_type = None

    def __init__(self, **params):
        super(RSISignal, self).__init__(params)
        self.lower_bound = self.params.get('lower_bound', 20)
        self.upper_bound = self.params.get('lower_bound', 80)

    def find_signal(self):
        pass


class MACDSignal(SignalBase):
    signal_name = "MACD"
    signal_type = None

    def __init__(self, **params):
        super(MACDSignal, self).__init__(params)
        self.lower_bound = self.params.get('lower_bound', 20)
        self.upper_bound = self.params.get('lower_bound', 80)

    def find_signal(self):
        pass
