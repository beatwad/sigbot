from signals.signal_base import SignalBase


class STOCHSignal(SignalBase):
    signal_name = "STOCH"
    signal_type = None

    def __init__(self, date_time, c_currency_name, initial_price, time_interval, lowk, slowd):
        self.date_time = date_time
        self.c_currrency_name = c_currency_name
        self.initial_price = initial_price
        self.time_interval = time_interval
        self.lowk = lowk
        self.slowd = slowd

    def create_signal(self):
        pass
        # if 27 <= self.stoch_score <= 33:
        #     self.signal_type = False
        # elif 65 <= self.stoch_score <= 75:
        #     self.signal_type = True
        # else:
        #     self.signal_type = None

    def create_signal_message(self):
        self.create_signal()
        data = {"dateTime": self.date_time,
                "signalName": self.signal_name,
                "coinName": self.c_currrency_name,
                "coinInitialPrice": self.initial_price,
                "timeInterval": self.time_interval}
        return data
