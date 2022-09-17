import time
from log.log import exception
# from log.log import logger
from time import sleep
from os import remove
from os import environ
import pandas as pd

from config.config import ConfigFactory
from visualizer.visualizer import Visualizer

# import threading
from threading import Thread
from threading import Event

import telegram
from telegram import Update
from telegram.ext import Updater, CallbackContext, MessageHandler, Filters


# Set environment variable
environ["ENV"] = "development"
# Get configs
configs = ConfigFactory.factory(environ).configs


class TelegramBot(Thread):
    type = 'Telegram'

    # constructor
    def __init__(self, token,  database, **params):
        # initialize separate thread the Telegram bot, so it can work independently
        Thread.__init__(self)
        # event for stopping bot thread
        self.stopped = Event()
        # event for updating bot thread
        self.update_bot = Event()
        # ticker database
        self.database = database
        # visualizer class
        self.visualizer = Visualizer(**configs)
        # bot parameters
        self.params = params[self.type]['params']
        self.chat_ids = self.params['chat_ids']
        self.prev_sig_limit = self.params.get('prev_sig_limit', 1500)
        self.max_notifications_in_row = self.params.get('self.max_notifications_in_row', 3)
        self.updater = Updater(token=token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        # list of notifications
        self.notification_list = list()
        # dataframe for storing of notification history
        self.notification_df = pd.DataFrame(columns=['time', 'sig_type', 'ticker', 'timeframe', 'pattern'])
        # set of images to delete
        self.images_to_delete = set()

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
                self.check_notifications()

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

    def add_plot(self, message: list) -> str:
        """ Generate signal plot, save it to file and add this filepath to the signal point data """
        sig_img_path = self.visualizer.create_plot(self.database, message, levels=[])
        # add image to set of images which we are going to delete
        self.images_to_delete.add(sig_img_path)
        return sig_img_path

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

    # @exception
    def check_notifications(self):
        """ Check if we can send each notification separately or there are too many of them,
            so we have to send list of them in one message """
        n_len = len(self.notification_list)
        message_dict = dict()
        # to each pattern corresponds its own chat, so we have to check length of notification list for each pattern
        for pattern in self.chat_ids.keys():
            message_dict[pattern] = list()
        for i in range(n_len):
            message = self.notification_list[i]
            sig_pattern = '_'.join([p[0] for p in message[5]])
            message_dict[sig_pattern].append([i, message])
        for pattern in self.chat_ids.keys():
            # send too long notification list with one message
            if len(message_dict[pattern]) > self.max_notifications_in_row:
                self.send_notifications_in_list(message_dict[pattern], pattern)
            else:
                # send each message from short notification list separately
                for i, message in message_dict[pattern]:
                    self.send_notification(message)
        # clear all sent notifications
        self.notification_list[:n_len] = []

    def send_notifications_in_list(self, message_list: list, pattern: str) -> None:
        """ Send list notifications at once """
        chat_id = self.chat_ids[pattern]
        # Form text message
        text = f'Новые сигналы:\n'
        for _message in message_list:
            i, message = _message
            # Get info from signal
            ticker = self.process_ticker(message[0])
            timeframe = message[1]
            sig_type = message[2]
            sig_time = message[4]
            # get patterns
            sig_pattern = [p[0] for p in message[5]]
            sig_pattern = '_'.join(sig_pattern)
            # add ticker info to notification list
            if self.check_previous_notifications(sig_time, sig_type, ticker, timeframe, sig_pattern):
                text += f' • {ticker}: '
                if sig_type == 'buy':
                    text += 'Покупка \n'
                else:
                    text += 'Продажа \n'
            self.add_to_notification_history(sig_time, sig_type, ticker, timeframe, sig_pattern)
        self.send_message(chat_id, text)
        self.delete_images()
    
    @exception
    def send_notification(self, message) -> None:
        """ Send notification separately """
        # Get info from signal
        ticker = self.process_ticker(message[0])
        timeframe = message[1]
        df_index = message[2]
        sig_type = message[3]
        sig_time = message[4]
        # get patterns
        sig_pattern = [p[0] for p in message[5]]
        sig_pattern = '_'.join(sig_pattern)
        # get path to image
        sig_img_path = self.add_plot(message)
        # get list of available exchanges
        sig_exchanges = message[7]
        # # get total and ticker statistics
        # result_statistics = message[8]
        # form message
        if self.check_previous_notifications(sig_time, sig_type, ticker, timeframe, sig_pattern):
            chat_id = self.chat_ids[sig_pattern]
            # Form text message
            text = f'{ticker} \n'
            if sig_type == 'buy':
                text += 'Покупка \n'
            else:
                text += 'Продажа \n'
            text += 'Продается на биржах: \n'
            for exchange in sig_exchanges:
                text += f' • {exchange}\n'
            text += 'Ссылка на TradingView: \n'
            text += f"https://ru.tradingview.com/symbols/{ticker.replace('-', '').replace('SWAP', '')}"
            # Send message + image
            if sig_img_path:
                self.send_photo(chat_id, sig_img_path, text)
            time.sleep(1)
        self.add_to_notification_history(sig_time, sig_type, ticker, timeframe, sig_pattern)
        self.delete_images()
    
    def send_message(self, chat_id, text):
        try:
            self.updater.bot.send_message(chat_id=chat_id, text=text)
        except (telegram.error.RetryAfter, telegram.error.NetworkError):
            sleep(30)
            self.updater.bot.send_message(chat_id=chat_id, text=text)

    def send_photo(self, chat_id, img_path, text):
        try:
            self.updater.bot.send_photo(chat_id=chat_id, photo=open(img_path, 'rb'), caption=text)
        except (telegram.error.RetryAfter, telegram.error.NetworkError):
            sleep(30)
            self.updater.bot.send_photo(chat_id=chat_id, photo=open(img_path, 'rb'), caption=text)

    def delete_images(self):
        """ Remove images after we send them, because we don't need them anymore """
        while self.images_to_delete:
            img_path = self.images_to_delete.pop()
            try:
                remove(img_path)
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    # Set environment variable
    environ["ENV"] = "development"
    # Get configs
    configs = ConfigFactory.factory(environ).configs

    telegram_bot = TelegramBot(token='5770186369:AAFrHs_te6bfjlHeD6mZDVgwvxGQ5TatiZA', **configs)
    telegram_bot.start()
