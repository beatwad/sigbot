import re
import time
import asyncio
from os import environ, remove

# Get configs
# environ["ENV"] = "5m_1h"

from log.log import exception
from time import sleep
import pandas as pd

from config.config import ConfigFactory
from visualizer.visualizer import Visualizer

import telegram
from telegram import Bot, Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

# Get configs
configs = ConfigFactory.factory(environ).configs


class TelegramBot:
    type = 'Telegram'

    # constructor
    def __init__(self, token,  database, **configs):
        # ticker database
        self.database = database
        # visualizer class
        self.visualizer = Visualizer(**configs)
        # bot parameters
        self.bot = Bot(token=token)
        self.loop = asyncio.new_event_loop()
        self.configs = configs[self.type]['params']
        self.allowed_exchanges = self.configs['allowed_exchanges']
        self.chat_ids = self.configs['chat_ids']
        self.message_thread_ids = self.configs['message_thread_ids']
        self.min_prev_candle_limit = self.configs.get('min_prev_candle_limit', 3)
        self.min_prev_candle_limit_higher = self.configs.get('min_prev_candle_limit_higher', 2)
        self.max_notifications_in_row = self.configs.get('self.max_notifications_in_row', 3)
        # self.dispatcher = self.updater.dispatcher
        # list of notifications
        self.notification_list = list()
        # dataframe for storing of notification history
        self.notification_df = pd.DataFrame(columns=['time', 'sig_type', 'ticker', 'timeframe', 'pattern'])
        # set of images to delete
        self.images_to_delete = set()
        # dictionary that is used to determine too late signals according to current work_timeframe
        self.timeframe_div = configs['Data']['Basic']['params']['timeframe_div']
        # Get working and higher timeframes
        self.higher_tf_patterns = configs['Higher_TF_indicator_list']
        self.work_timeframe = configs['Timeframes']['work_timeframe']
        self.higher_timeframe = configs['Timeframes']['higher_timeframe']

    @staticmethod
    async def get_chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """ Print chat id in response to user message """
        chat_id = update.effective_chat['id']
        message_thread_id = update.message.message_thread_id
        text = f'ID данного чата: {chat_id}, ID данной темы: {message_thread_id}'
        x = await context.bot.send_message(chat_id=chat_id, message_thread_id=message_thread_id, text=text)
        return x

    @exception
    def polling(self) -> None:
        """ Start the bot polling """
        if __name__ == '__main__':
            # Create the Application and pass it your bot's token.
            application = Application.builder().token('5770186369:AAFrHs_te6bfjlHeD6mZDVgwvxGQ5TatiZA').build()
            # on non command i.e. message - echo the message on Telegram
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.get_chat_id))
            # Run the bot until the user presses Ctrl-C
            application.run_polling()

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
            if set(pattern.split('_')).intersection(set(self.higher_tf_patterns)):
                if (sig_time - latest_time).total_seconds() < self.timeframe_div[self.higher_timeframe] * \
                        self.min_prev_candle_limit_higher:
                    return False
            else:
                if (sig_time - latest_time).total_seconds() < self.timeframe_div[self.work_timeframe] * \
                        self.min_prev_candle_limit:
                    return False
        return True

    def check_notifications(self):
        """ Check if we can send each notification separately or there are too many of them,
            so we have to send list of them in one message """
        n_len = len(self.notification_list)
        message_dict = dict()
        # each pattern corresponds to its own chat, so we have to check the length of notification list for each pattern
        for pattern in self.chat_ids.keys():
            message_dict[pattern] = {'buy': [], 'sell': []}
        for i in range(n_len):
            message = self.notification_list[i]
            sig_type = message[3]
            sig_pattern = message[5]
            message_dict[sig_pattern][sig_type].append([i, message])
        for pattern in self.chat_ids.keys():
            for ttype in ['buy', 'sell']:
                # send too long notification list in one message
                if len(message_dict[pattern][ttype]) > self.max_notifications_in_row:
                    self.send_notifications_in_list(message_dict[pattern][ttype], pattern)
                else:
                    # send each message from short notification list separately
                    for i, message in message_dict[pattern][ttype]:
                        self.send_notification(message)
        # clear all sent notifications
        self.notification_list[:n_len] = []

    def send_notifications_in_list(self, message_list: list, pattern: str) -> None:
        """ Send list notifications at once """
        chat_id = self.chat_ids[pattern]
        # Form text message
        text_buy = f'Новые сигналы:\n'
        text_sell = f'Новые сигналы:\n'
        # flag that lets message sending only if there are new tickers
        send_flag_buy = send_flag_sell = False
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
                    if sig_type == 'buy':
                        text_buy += f' • {ticker} \n'
                        send_flag_buy = True
                    else:
                        text_sell += f' • {ticker} \n'
                        send_flag_sell = True
                self.add_to_notification_history(sig_time, sig_type, ticker, timeframe, sig_pattern)

        if send_flag_buy:
            message_thread_id = self.message_thread_ids.get(f'{pattern}_buy', None)
            self.send_message(chat_id, message_thread_id, text_buy)

        if send_flag_buy:
            message_thread_id = self.message_thread_ids.get(f'{pattern}_sell', None)
            self.send_message(chat_id, message_thread_id, text_sell)

        if send_flag_buy or send_flag_sell:
            self.delete_images()

    @staticmethod
    def clean_ticker(ticker: str) -> str:
        """ Clean ticker of not necessary symbols (2U, 1000, 10000, SWAP, -, etc.) """
        if ticker not in ['1INCH']:
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
        # get list of available exchanges
        sig_exchanges = message[7]
        # Check if the same message wasn't send short time ago
        if self.check_previous_notifications(sig_time, sig_type, ticker, timeframe, sig_pattern):
            # if we have a list of exchanges for that notifications are allowed - use it to filter ticker from
            # not necessary exchanges
            if len(self.allowed_exchanges) == 0 or set(sig_exchanges).intersection(set(self.allowed_exchanges)):
                # create image and return path to it
                sig_img_path = self.add_plot(message)
                chat_id = self.chat_ids[sig_pattern]
                message_thread_id = self.message_thread_ids.get(f'{sig_pattern}_{sig_type}', None)
                if message_thread_id is not None:
                    message_thread_id = int(message_thread_id)
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
                    self.send_photo(chat_id, message_thread_id, sig_img_path, text)
                time.sleep(0.5)
        self.add_to_notification_history(sig_time, sig_type, ticker, timeframe, sig_pattern)
        self.delete_images()

    @staticmethod
    async def bot_send_message(bot: Bot, chat_id: str, message_thread_id: int, text: str) -> telegram.Message:
        return await bot.send_message(chat_id=chat_id, message_thread_id=message_thread_id, text=text)

    def send_message(self, chat_id: str, message_thread_id: int, text: str):
        tasks = [self.loop.create_task(self.bot_send_message(self.bot, chat_id, message_thread_id, text))]
        try:
            self.loop.run_until_complete(asyncio.wait(tasks))
        except (telegram.error.RetryAfter, telegram.error.NetworkError, telegram.error.BadRequest):
            sleep(30)
            self.loop.run_until_complete(asyncio.wait(tasks))

    @staticmethod
    async def bot_send_photo(bot: Bot, chat_id: str, message_thread_id: int,
                             img_path: str, text: str) -> telegram.Message:
        return await bot.send_photo(chat_id=chat_id, message_thread_id=message_thread_id,
                                    photo=open(img_path, 'rb'), caption=text)

    def send_photo(self, chat_id: str, message_thread_id: int, img_path: str, text: str):
        tasks = [self.loop.create_task(self.bot_send_photo(self.bot, chat_id, message_thread_id, img_path, text))]
        try:
            self.loop.run_until_complete(asyncio.wait(tasks))
        except (telegram.error.RetryAfter, telegram.error.NetworkError, telegram.error.BadRequest):
            sleep(30)
            self.loop.run_until_complete(asyncio.wait(tasks))

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
    telegram_bot.polling()
