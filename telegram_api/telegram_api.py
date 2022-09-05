import sys
import time
import logging
import functools
from os import environ
from telegram.ext import Updater
from config.config import ConfigFactory

from threading import Thread
from threading import Event


# Set environment variable
environ["ENV"] = "development"
# Get configs
configs = ConfigFactory.factory(environ).configs


def create_logger():
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger("example_logger")
    logger.setLevel(logging.INFO)
    # create the logging file handler
    log_path = configs['Telegram']['params']['log_path']
    fh = logging.FileHandler(log_path)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    # add handler to logger object
    logger.addHandler(fh)
    return logger


# create logger
logger = create_logger()


def exception(function):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        logger = create_logger()
        try:
            return function(*args, **kwargs)
        except:
            # log the exception
            err = "There was an exception in  "
            err += function.__name__
            logger.exception(err)
            # re-raise the exception
            raise
    return wrapper


class TelegramBot(Thread):
    type = 'Telegram'

    # constructor
    def __init__(self, token,  **params):
        # initialize separate thread the Telegram bot, so it can work independently
        Thread.__init__(self)
        # event for stopping bot thread
        self.stopped = Event()
        # event for updating bot thread
        self.update_bot = Event()
        # bot parameters
        self.params = params[self.type]['params']
        self.chat_ids = self.params['chat_ids']
        self.updater = Updater(token=token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        # list of notifications
        self.notification_list = list()

    @exception
    def run(self) -> None:
        """ Until stopped event is set - run bot's thread and update it every second """
        while not self.stopped.wait(1):
            if self.update_bot.is_set():
                self.update_bot.clear()
                self.send_notification()

    @staticmethod
    def process_ticker(ticker: str) -> str:
        """ Bring ticker to more convenient view """
        if '-' in ticker:
            return ticker
        if '/' in ticker:
            ticker = ticker.replace('/', '-')
            return ticker
        ticker = ticker[:-4] + '-' + ticker[-4:]
        return ticker

    def send_notification(self) -> None:
        """ Add notification to notification list and send its items to chat """
        while self.notification_list:
            # get pattern
            _message = self.notification_list.pop(0)
            ticker = self.process_ticker(_message[0])
            timeframe = _message[1]
            df_index = _message[2]
            sig_type = _message[3]
            sig_time = _message[4]
            sig_pattern = [p[0] for p in _message[5]]
            sig_pattern = '_'.join(sig_pattern)
            sig_img_path = _message[6][0]
            sig_exchanges = _message[7]
            chat_id = self.chat_ids[sig_pattern]
            text = f'Новый сигнал\n '
            if sig_type == 'buy':
                text += '• Покупка \n '
            else:
                text += '• Продажа \n '
            text += f'• {ticker} \n '
            text += f'• Таймфрейм {timeframe} \n '
            text += f'Продается на биржах: \n'
            for exchange in sig_exchanges:
                text += f'• {exchange} \n '
            # logger.info(text)
            if sig_img_path:
                self.send_photo(chat_id, sig_img_path, text)
            # self.send_message(chat_id, text)
            time.sleep(5)

    def send_message(self, chat_id, text):
        self.updater.bot.send_message(chat_id=chat_id, text=text)

    def send_photo(self, chat_id, img_path, text):
        self.updater.bot.send_photo(chat_id=chat_id, photo=open(img_path, 'rb'), caption=text)


if __name__ == '__main__':
    telegram_bot = TelegramBot(token='5770186369:AAFrHs_te6bfjlHeD6mZDVgwvxGQ5TatiZA', **configs)
    telegram_bot.start()
    time.sleep(5)
    telegram_bot.notification_list.append(['BTCUSDT', '5m', 10, 'buy', "31.12.11",
                                           ["STOCH", "RSI", "SUP_RES"],
                                           '../visualizer/images/ZBC-USDT_5m_2022-09-05 09:50:00.png',
                                           ["Binance", "OKEX"]])
    telegram_bot.update_bot.set()



