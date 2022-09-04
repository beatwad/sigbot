import re
from abc import ABCMeta


class ApiBase(metaclass=ABCMeta):
    @staticmethod
    def check_symbols(symbols):
        """ Check if ticker is not pair with fiat currency or stablecoin """
        filtered_symbols = list()
        for symbol in symbols:
            fiat = ['EUR', 'CHF', 'GBP', 'JPY', 'CNY', 'RUB']
            if re.match('.?USD', symbol) or re.match('.?UST', symbol):
                continue
            for f in fiat:
                if symbol.startswith(f):
                    continue
            filtered_symbols.append(symbol)
        return filtered_symbols

    @staticmethod
    def check_symbol(symbols):
        """ Check if ticker is not pair with fiat currency or stablecoin """
        filtered_symbols = list()
        for symbol in symbols:
            fiat = ['EUR', 'CHF', 'GBP', 'JPY', 'CNY', 'RUB']
            if re.match('.?USD', symbol) or re.match('.?UST', symbol):
                continue
            for f in fiat:
                if symbol.startswith(f):
                    continue
            filtered_symbols.append(symbol)
        return filtered_symbols


