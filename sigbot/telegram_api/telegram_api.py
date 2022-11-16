import re
import time
from os import environ, remove

# Get configs
environ["ENV"] = "5m_1h"

from log.log import exception
from time import sleep
import pandas as pd
from threading import Thread, Event

from config.config import ConfigFactory
from visualizer.visualizer import Visualizer

import telegram
from telegram import Update
from telegram.ext import Updater, CallbackContext, MessageHandler, Filters


# Get configs
configs = ConfigFactory.factory(environ).configs


class TelegramBot(Thread):
    type = 'Telegram'

    # constructor
    def __init__(self, token,  database, **configs):
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
        self.configs = configs[self.type]['params']
        self.chat_ids = self.configs['chat_ids']
        self.prev_sig_limit = self.configs.get('prev_sig_minutes_limit', 25)
        self.max_notifications_in_row = self.configs.get('self.max_notifications_in_row', 3)
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
            if sig_time - latest_time < pd.Timedelta(self.prev_sig_limit, "m"):
                return False
        return True

    def check_notifications(self):
        """ Check if we can send each notification separately or there are too many of them,
            so we have to send list of them in one message """
        n_len = len(self.notification_list)
        message_dict = dict()
        # each pattern corresponds to its own chat, so we have to check the length of notification list for each pattern
        for pattern in self.chat_ids.keys():
            message_dict[pattern] = list()
        for i in range(n_len):
            message = self.notification_list[i]
            sig_pattern = message[5]
            message_dict[sig_pattern].append([i, message])
        for pattern in self.chat_ids.keys():
            # send too long notification list in one message
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
        # flag that lets message sending only if there are new tickers
        send_flag = False
        # list to filter duplicate messages
        message_list_tmp = list()
        for _message in message_list:
            _, message = _message
            # Get info from signal
            ticker = self.process_ticker(message[0])
            timeframe = message[1]
            sig_type = message[3]
            sig_time = message[4]
            sig_pattern = message[5]
            # if ticker and trade type aren't in message list already - add them
            if [ticker, sig_type] not in message_list_tmp:
                message_list_tmp.append([ticker, sig_type])
                # add ticker info to notification list
                if self.check_previous_notifications(sig_time, sig_type, ticker, timeframe, sig_pattern):
                    send_flag = True
                    if sig_pattern == 'HighVolume':
                        text += f' • {ticker} \n '
                    elif sig_type == 'buy':
                        text += f' • {ticker}: Покупка \n'
                    else:
                        text += f' • {ticker}: Продажа \n'
                self.add_to_notification_history(sig_time, sig_type, ticker, timeframe, sig_pattern)
        if send_flag:
            self.send_message(chat_id, text)
            self.delete_images()

    @staticmethod
    def clean_ticker(ticker: str) -> str:
        """ Clean ticker of not necessary symbols (2U, 1000, 10000, SWAP, -, etc.) """
        ticker = re.sub(r'\b\d+', '', ticker)
        ticker = re.sub('SWAP', '', ticker)
        ticker = re.sub('-', '', ticker)
        return ticker

    def send_notification(self, message) -> None:
        """ Send notification separately """
        # Get info from signal
        ticker = self.process_ticker(message[0])
        timeframe = message[1]
        sig_type = message[3]
        sig_time = message[4]
        sig_pattern = message[5]
        # get patterns
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
            if sig_pattern == 'HighVolume':
                pass
            elif sig_type == 'buy':
                text += 'Покупка \n'
            else:
                text += 'Продажа \n'
            text += 'Продается на биржах: \n'
            for exchange in sig_exchanges:
                text += f' • {exchange}\n'
            text += 'Ссылка на TradingView: \n'
            clean_ticker = self.clean_ticker(ticker)
            text += f"https://ru.tradingview.com/symbols/{clean_ticker}\n"
            # text += f"https://ru.tradingview.com/chart/?symbol={sig_exchanges[0]}%3A{self.clean_ticker(ticker)}\n"
            if clean_ticker[:-4] != 'BTC':
                text += 'Ссылка на график с BTC: \n'
                text += f"https://ru.tradingview.com/symbols/{clean_ticker[:-4]}BTC"
                # text += f"https://ru.tradingview.com/chart/" \
                #         f"?symbol={sig_exchanges[0]}%3A{self.clean_ticker(ticker)[:-4]}BTC"
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
    configs = ConfigFactory.factory(environ).configs

    telegram_bot = TelegramBot(token='5770186369:AAFrHs_te6bfjlHeD6mZDVgwvxGQ5TatiZA', database=None, **configs)
    telegram_bot.start()
