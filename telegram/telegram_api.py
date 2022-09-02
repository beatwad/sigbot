#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple Bot to reply to Telegram messages.
This program is dedicated to the public domain under the CC0 license.
This Bot uses the Updater class to handle the bot.
First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.
Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import time
import logging
import functools
import threading
from os import environ
from telegram.ext.dispatcher import run_async
from telegram.ext import Updater, CommandHandler

from config.config import ConfigFactory

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
    print(configs.keys())
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


# variable for thread locking
global_lock = threading.Lock()


class Bot:
    # constructor
    def __init__(self, token, language='Русский'):
        self.token = token  # token
        self.bot_is_started = False
        self.sig_bot_is_started = False
        self.language = language
        self.request_kwargs = None
        self.updater = None
        self.dispatcher = None
        self.chat_id = None
        self.notification_list = []
        self.bot = None
        self.proxy_bot = None

    def run(self, sig_bot_status) -> None:
        """ Start Telegram bot """
        self.sig_bot_is_started = sig_bot_status
        self.updater = Updater(self.token, request_kwargs=self.request_kwargs)  # python-telegram-bot updater
        self.dispatcher = self.updater.dispatcher  # python-telegram-bot dispatcher
        self.chat_id = None
        self.bot = self.updater.bot

        # on different commands - answer in Telegram
        self.dispatcher.add_handler(CommandHandler('start', self.start))
        self.dispatcher.add_handler(CommandHandler('help', self.help))
        self.dispatcher.add_handler(CommandHandler('start_bot', self.start_sig_bot))
        self.dispatcher.add_handler(CommandHandler('stop_bot', self.stop_sig_bot))

        self.dispatcher.add_error_handler(_error)

        telegram_name = '@' + str(self.get_me()['username'])
        # if telegram_name:
        #     lbtc_storage.save_item('settings', telegram_name=telegram_name)
        self.bot_is_started = True
        self.updater.start_polling()

    def start(self, bot, update) -> None:
        """Send a message when the command /start is issued."""
        self.chat_id = update.message.chat_id
        if self.language == 'English':
            message = 'Telegram bot is started. Available commands: /start, /help, /start_bot, /stop_bot'
        else:
            message = 'Бот Telegram запущен. Доступные команды: /start, /help, /start_bot, /stop_bot'
        self.bot.send_message(chat_id=update.message.chat_id, text=message)

    def stop(self) -> None:
        """ Stop Telegram bot """
        self.bot_is_started = False
        self.updater.stop()

    def get_me(self):
        return self.bot.getMe()

    def help(self, bot, update) -> None:
        self.chat_id = update.message.chat_id
        if self.language == 'English':
            message = 'Available commands: /help, /start_bot, /stop_bot \n' \
                      '/start_bot - Signal bot start\n'\
                      '/stop_bot - Signal bot stop\n'
        else:
            message = 'Доступные команды: /help, /start_bot, /stop_bot \n' \
                      '/start_bot - Запуск сигнального бота \n' \
                      '/stop_bot - Остановка сигнального бота \n'
        self.bot.send_message(chat_id=update.message.chat_id, text=message)

    def start_sig_bot(self, bot, update) -> None:
        """ Method for LocalBitcoins bot start command handling """
        self.chat_id = update.message.chat_id
        if self.sig_bot_is_started is False:
            self.sig_bot_is_started = True

    def stop_sig_bot(self, bot, update) -> None:
        """ Method for LocalBitcoins bot stop command handling """
        self.chat_id = update.message.chat_id
        if self.sig_bot_is_started is True:
            self.sig_bot_is_started = False

    def send_message(self, message) -> None:
        """ Send message with bot """
        # wait until global lock released
        while global_lock.locked():
            continue

        # acquire lock
        global_lock.acquire()

        if self.chat_id:
            try:
                self.bot.send_message(chat_id=self.chat_id, text=message)
            except Exception as e:
                logger.exception(e)

        # after all operations are done - release the lock
        global_lock.release()

    @run_async
    def send_notification(self, notification) -> None:
        """ Add notification to notification list and send its items to chat """
        self.notification_list.append(notification)
        while self.notification_list:
            time.sleep(1)
            print(f'self.chat_id - {self.chat_id}')
            if self.chat_id:
                _message = self.notification_list.pop()
                self.bot.send_message(chat_id=self.chat_id, text=_message)


def _error(update, error) -> None:
    """Log Errors caused by Updates."""
    logger.warning(f'Update {update} caused error {error}')


def main(token) -> None:
    """Start the bot."""
    # Create the EventHandler and pass it your bot's token.
    tlgrm = Bot(token)
    tlgrm.notification_list.append('test')
    # Start the Bot
    tlgrm.run(False)


if __name__ == '__main__':
    main('5770186369:AAFrHs_te6bfjlHeD6mZDVgwvxGQ5TatiZA')



