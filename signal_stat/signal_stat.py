import pandas as pd


class TotalSignalStat:
    """ Abstract indicator class """
    type = 'SignalStat'
    name = 'Total'

    def __init__(self, params):
        self.params = params[self.type][self.name]['params']

    """ Get indicator data and write it to the dataframe """
    @abstractmethod
    def get_indicator(self, *args, **kwargs):
        pass
