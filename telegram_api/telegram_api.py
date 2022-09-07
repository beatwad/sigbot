import time
import logging
import functools
from os import remove
from os import environ

import pandas as pd
from config.config import ConfigFactory

from threading import Thread
from threading import Event

from telegram import Update
from telegram.ext import Updater, CallbackContext, MessageHandler, Filters


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
        self.prev_sig_limit = self.params.get('prev_sig_limit', 1500)
        self.updater = Updater(token=token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        # list of notifications
        self.notification_list = list()
        # dataframe for storing of notification history
        self.notification_df = pd.DataFrame(columns=['time', 'sig_type', 'ticker', 'timeframe', 'pattern'])
        # set of images to delete
        self.images_to_delete = set()

    @exception
    def run(self) -> None:
        """ Until stopped event is set - run bot's thread and update it every second """
        # on different commands - answer in Telegram
        # self.dispatcher.add_handler(CommandHandler('chat_id', self.get_chat_id))
        # on non command i.e. message - echo the message on Telegram
        if __name__ == '__main__':
            self.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, self.get_chat_id))
            self.updater.start_polling()

        while not self.stopped.wait(1):
            if self.update_bot.is_set():
                self.update_bot.clear()
                self.send_notification()

    def get_chat_id(self, update: Update, context: CallbackContext) -> None:
        """Send a message when the command /start is issued."""
        # update.message.reply_text(update.message.text)
        chat_id = update.effective_chat['id']
        text = f'ID данного чата: {chat_id}'
        self.send_message(chat_id=chat_id, text=text)

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

    def check_previous_notifications(self, sig_time: pd.Timestamp, sig_type: str, ticker: str,
                                     timeframe: str, pattern: str) -> bool:
        """ Check if previous notifications wasn't send short time before """
        tmp = self.notification_df[
                                    (self.notification_df['sig_type'] == sig_type) &
                                    (self.notification_df['ticker'] == ticker) &
                                    (self.notification_df['timeframe'] == timeframe) &
                                    (self.notification_df['pattern'] == pattern)
                                   ]
        if tmp.shape[0] > 0:
            latest_time = tmp['time'].max()
            if sig_time - latest_time < pd.Timedelta(self.prev_sig_limit, "s"):
                return False
        return True

    def add_to_notification_history(self, sig_time: pd.Timestamp, sig_type: str, ticker: str,
                                    timeframe: str, pattern: str) -> None:
        """ Add new notification to notification history """
        tmp = pd.DataFrame()
        tmp['time'] = [sig_time]
        tmp['sig_type'] = [sig_type]
        tmp['ticker'] = [ticker]
        tmp['timeframe'] = [timeframe]
        tmp['pattern'] = [pattern]
        self.notification_df = pd.concat([tmp, self.notification_df])

    def send_notification(self) -> None:
        """ Add notification to notification list and send its items to chat """
        while self.notification_list:
            # Get info from signal
            _message = self.notification_list.pop(0)
            ticker = self.process_ticker(_message[0])
            timeframe = _message[1]
            df_index = _message[2]
            sig_type = _message[3]
            sig_time = _message[4]
            # get patterns
            sig_pattern = [p[0] for p in _message[5]]
            sig_pattern = '_'.join(sig_pattern)
            # get path to image
            sig_img_path = _message[6][0]
            # get list of available exchanges
            sig_exchanges = _message[7]
            # get total and ticker statistics
            result_statistics = _message[8]
            # form message
            if self.check_previous_notifications(sig_time, sig_type, ticker, timeframe, sig_pattern):
                chat_id = self.chat_ids[sig_pattern]
                # Form text message
                text = f'Новый сигнал\n'
                if sig_type == 'buy':
                    text += ' • Покупка \n'
                else:
                    text += ' • Продажа \n'
                text += f' • {ticker} \n'
                text += 'Точность сигнала:\n'
                for i, t in enumerate(range(15, 105, 15)):
                    text += f' • через {t} минут: {result_statistics[0][i][0]}%\n'
                text += 'Cреднее движение:\n'
                for i, t in enumerate(range(15, 105, 15)):
                    text += f' • через {t} минут: {result_statistics[0][i][1]}%\n'
                text += 'Продается на биржах: \n'
                for exchange in sig_exchanges:
                    text += f' • {exchange}\n'
                text += 'Ссылка на TradingView: \n'
                text += f"https://ru.tradingview.com/symbols/{ticker.replace('-', '')}"
                # Send message + signal plot
                if sig_img_path:
                    self.send_photo(chat_id, sig_img_path, text)
                time.sleep(1)
            self.add_to_notification_history(sig_time, sig_type, ticker, timeframe, sig_pattern)
        self.delete_images()

    def send_message(self, chat_id, text):
        self.updater.bot.send_message(chat_id=chat_id, text=text)

    def send_photo(self, chat_id, img_path, text):
        self.updater.bot.send_photo(chat_id=chat_id, photo=open(img_path, 'rb'), caption=text)
        # add image to set of images which we are going to delete
        self.images_to_delete.add(img_path)

    def delete_images(self):
        """ Remove images after we send them, because we don't need them anymore """
        while self.images_to_delete:
            img_path = self.images_to_delete.pop()
            remove(img_path)


if __name__ == '__main__':
    # Set environment variable
    environ["ENV"] = "development"
    # Get configs
    configs = ConfigFactory.factory(environ).configs

    telegram_bot = TelegramBot(token='5770186369:AAFrHs_te6bfjlHeD6mZDVgwvxGQ5TatiZA', **configs)
    telegram_bot.start()
