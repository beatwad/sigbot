class CryptoCurrency:
    def __init__(self, symbol, interval, limit):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.open_values = list()
        self.close_values = list()
        self.high_values = list()
        self.low_values = list()
        self.volume_values = list()
        self.time = list()
