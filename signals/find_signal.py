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

    @staticmethod
    def lower_bound(indicator, low_bound):
        """ Returns True if indicator is lower than low_bound param """
        if indicator <= low_bound:
            return True
        return False

    @staticmethod
    def higher_bound(indicator, high_bound):
        """ Returns True if indicator is higher than high_bound param """
        if indicator <= high_bound:
            return True
        return False

    @staticmethod
    def up_direction(indicator_dir):
        """ Returns True if indicator values are moving up """
        if indicator_dir > 0:
            return True
        return False

    @staticmethod
    def down_direction(indicator_dir):
        """ Returns True if indicator values are moving down """
        if indicator_dir < 0:
            return True
        return False


class STOCHSignal(SignalBase):
    name = 'STOCH'

    def __init__(self, **params):
        super(STOCHSignal, self).__init__(params)
        self.low_bound = self.params.get('low_bound', 20)
        self.high_bound = self.params.get('high_bound', 80)

    @staticmethod
    def crossed_lines(df, row, i, up=True):
        """ Returns True if slowk and slowd lines of RSI has crossed """
        if up:
            if row['diff'] < 0 and df.loc[i - 1]['diff'] < 0 and df.loc[i - 2]['diff'] < 0:
                if df.loc[i - 3]['diff'] > 0 and df.loc[i - 4]['diff'] > 0 and \
                        df.loc[i - 5]['diff'] > 0:
                    return True
        else:
            if row['diff'] > 0 and df.loc[i - 1]['diff'] > 0 and df.loc[i - 2]['diff'] > 0:
                if df.loc[i - 3]['diff'] < 0 and df.loc[i - 4]['diff'] < 0 and \
                        df.loc[i - 5]['diff'] < 0:
                    return True
        return False

    def find_signal(self, df, ticker, timeframe):
        """ Return signal if RSI is higher/lower than high/low bound (overbuy/oversell zone),
            slowk and slowd lines have crossed and their direction is down/up """
        points = list()

        slowk = df['stoch_slowk'].values
        slowk_shift = df['stoch_slowk'].shift(1)
        slowd = df['stoch_slowd'].values
        slowd_shift = df['stoch_slowd'].shift(1)

        df['stoch_slowk_dir'] = (slowk - slowk_shift) / np.maximum(np.abs(slowk), np.abs(slowd_shift))
        df['stoch_slowd_dir'] = (slowd - slowd_shift) / np.maximum(np.abs(slowd), np.abs(slowd_shift))
        df['diff'] = slowk - slowd

        for i, row in df.iterrows():
            if i > 2 and self.lower_bound(row['{stoch_slowk'], self.low_bound) \
                    and self.lower_bound(row['stoch_slowd'], self.low_bound):
                if self.crossed_lines(df, row, i, up=False):
                    if self.up_direction(row['stoch_slowk_dir']) and self.up_direction(row['stoch_slowd_dir']):
                        points.append([row['stoch_slowk'], i])  # !!!
            elif i > 2 and self.higher_bound(row['stoch_slowk'], self.high_bound) \
                    and self.higher_bound(row['stoch_slowd'], self.high_bound):
                if self.crossed_lines(df, row, i, up=True):
                    if self.down_direction(row['stoch_slowk_dir']) and self.down_direction(row['stoch_slowd_dir']):
                        points.append([row['stoch_slowk'], i])  # !!!
        return True, points


class RSISignal(SignalBase):
    signal_name = "STOCH"
    signal_type = None

    def __init__(self, **params):
        super(RSISignal, self).__init__(params)
        self.low_bound = self.params.get('low_bound', 25)
        self.high_bound = self.params.get('high_bound', 75)

    def find_signal(self, df, ticker, timeframe):
        """ Return signal if RSI is higher/lower than high/low bound (overbuy/oversell zone),
            slowk and slowd lines have crossed and their direction is down/up """
        points = list()

        for row in df['rsi']:
            if self.lower_bound(row, self.low_bound):
                return True
            elif self.higher_bound(row, self.high_bound):
                return True
        return False


class MACDSignal(SignalBase):
    signal_name = "MACD"
    signal_type = None

    def __init__(self, **params):
        super(MACDSignal, self).__init__(params)
        self.low_bound = self.params.get('low_bound', 20)
        self.high_bound = self.params.get('high_bound', 80)

    def find_signal(self):
        pass
