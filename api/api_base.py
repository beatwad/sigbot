import re
from abc import ABCMeta, abstractmethod


class ApiBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def connect_to_api(self, api_key, api_secret):
        pass

    @staticmethod
    def check_symbol(symbol):
        """ Check if ticker is not pair with fiat currency or stablecoin """
        fiat = ['EUR', 'CHF', 'GBP', 'JPY', 'CNY', 'RUB']
        if re.match('.?USD', symbol) or re.match('.?UST', symbol):
            return False
        for f in fiat:
            if symbol.startswith(f):
                return False
        return True