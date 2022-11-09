import re
from abc import ABCMeta


class ApiBase(metaclass=ABCMeta):
    @staticmethod
    def check_symbols(symbols: list) -> list:
        """ Check if ticker is not pair with fiat currency or stablecoin or ticker is not a leverage type """
        filtered_symbols = list()
        for symbol in symbols:
            if symbol.endswith('USD') or symbol.endswith('UST'):
                continue
            if re.match(r'.+[23][LS].+', symbol) or re.match(r'.+UP-?(BUSD|USD[TC])].+', symbol) or \
                    re.match(r'.+DOWN-?(BUSD|USD[TC])', symbol):
                continue
            fiat = ['EUR', 'CHF', 'GBP', 'JPY', 'CNY', 'RUB', 'AUD']
            for f in fiat:
                if symbol.startswith(f):
                    break
            else:
                filtered_symbols.append(symbol)
        return filtered_symbols


if __name__ == '__main__':
    symbol = 'XRPDOWN-BUSD'
    print(re.match(r'.+DOWN-?(BUSD|USD[TC])', symbol))
