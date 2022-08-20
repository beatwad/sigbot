import numpy as np
from abc import abstractmethod

import pandas as pd


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
    def lower_bound(indicator: float, low_bound: float) -> bool:
        """ Returns True if indicator is lower than low_bound param """
        if indicator <= low_bound:
            return True
        return False

    @staticmethod
    def higher_bound(indicator: float, high_bound: float) -> bool:
        """ Returns True if indicator is higher than high_bound param """
        if indicator >= high_bound:
            return True
        return False

    @staticmethod
    def up_direction(indicator_dir: float) -> bool:
        """ Returns True if indicator values are moving up """
        if indicator_dir > 0:
            return True
        return False

    @staticmethod
    def down_direction(indicator_dir: float) -> bool:
        """ Returns True if indicator values are moving down """
        if indicator_dir < 0:
            return True
        return False


class STOCHSignal(SignalBase):
    type = 'Indicator_signal'
    name = 'STOCH'

    def __init__(self, **params):
        super(STOCHSignal, self).__init__(params)
        self.low_bound = self.params.get('low_bound', 20)
        self.high_bound = self.params.get('high_bound', 80)

    @staticmethod
    def crossed_lines(df: pd.DataFrame, row: pd.Series, i, up: bool):
        """ Returns True if slowk and slowd lines of RSI has crossed """
        if up:
            if row['diff'] < 0 and df.loc[i - 1]['diff'] > 0:
                return True
        else:
            if row['diff'] > 0 and df.loc[i - 1]['diff'] < 0:
                return True

        return False

    def find_signal(self, row: pd.Series, df: pd.DataFrame, index: int) -> (bool, str):
        """ Return signal if RSI is higher/lower than high/low bound (overbuy/oversell zone),
            slowk and slowd lines have crossed and their direction is down/up """

        # Find STOCH signal
        if index > 2 and self.lower_bound(row['stoch_slowk'], self.low_bound) \
                and self.lower_bound(row['stoch_slowd'], self.low_bound):
            if self.crossed_lines(df, row, index, up=False):
                if self.up_direction(row['stoch_slowk_dir']) and self.up_direction(row['stoch_slowd_dir']):
                    return True, 'buy'
        elif index > 2 and self.higher_bound(row['stoch_slowk'], self.high_bound) \
                and self.higher_bound(row['stoch_slowd'], self.high_bound):
            if self.crossed_lines(df, row, index, up=True):
                if self.down_direction(row['stoch_slowk_dir']) and self.down_direction(row['stoch_slowd_dir']):
                    return True, 'sell'
        return False, ''


class RSISignal(SignalBase):
    type = 'Indicator_signal'
    name = "RSI"

    def __init__(self, **params):
        super(RSISignal, self).__init__(params)
        self.low_bound = self.params.get('low_bound', 25)
        self.high_bound = self.params.get('high_bound', 75)

    def find_signal(self, row: pd.Series, *args) -> (bool, str):
        """ Return signal if RSI is higher/lower than high/low bound (overbuy/oversell zone),
            slowk and slowd lines have crossed and their direction is down/up """

        # Find RSI signal
        if self.lower_bound(row['rsi'], self.low_bound):
            return True, 'buy'
        elif self.higher_bound(row['rsi'], self.high_bound):
            return True, 'sell'
        return False, ''


class MACDSignal(SignalBase):
    type = 'Indicator_signal'
    name = 'MACD'

    def __init__(self, **params):
        super(MACDSignal, self).__init__(params)
        self.low_bound = self.params.get('low_bound', 20)
        self.high_bound = self.params.get('high_bound', 80)

    def find_signal(self):
        pass


class FindSignal:
    """ Class for searching of the indicator combination """
    def __init__(self, configs):
        self.first = True
        self.configs = configs
        self.indicator_list = configs['Indicator_list']

    def prepare_dataframe(self, df):
        """ Add all necessary indicator data to dataframe """
        if 'STOCH' in self.indicator_list:
            df['stoch_slowk_dir'] = df['stoch_slowk'].pct_change().rolling(5).mean()
            df['stoch_slowd_dir'] = df['stoch_slowd'].pct_change().rolling(5).mean()
            df['diff'] = df['stoch_slowk'] - df['stoch_slowd']
            df['diff'] = df['diff'].rolling(5).mean()

        return df

    def prepare_indicator_signals(self) -> list:
        """ Get all indicator signal classes """
        indicator_signals = list()
        for indicator in self.indicator_list:
            indicator_signals.append(SignalFactory.factory(indicator, self.configs))
        return indicator_signals

    def find_signal(self, df: pd.DataFrame) -> list:
        """ Search for the signals through the dataframe, if found - add its index and trade type to the list """
        points = list()

        df = self.prepare_dataframe(df)

        indicator_signals = self.prepare_indicator_signals()
        for index, row in df.iterrows():
            # If we update our signal data, it's not necessary to check it all
            if index > 5 and not self.first:
                break
            for indicator_signal in indicator_signals:
                if not indicator_signal.find_signal(row, df, index)[0]:
                    break
            else:
                points.append((index, indicator_signals[0].find_signal(row, df, index)[1]))

        self.first = False

        return points
