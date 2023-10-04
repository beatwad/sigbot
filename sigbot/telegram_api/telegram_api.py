import re
import time
import asyncio
from os import environ, remove
from constants.constants import telegram_token

# Get configs
# environ["ENV"] = "1h_4h"

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
# Disable SettingWithCopyWarning because it's not necessary here
pd.options.mode.chained_assignment = None


class TelegramBot:
    type = 'Telegram'

    # constructor
    def __init__(self, token,  database, **configs):
        # ticker database
        self.database = database
        # visualizer class
        self.visualizer = Visualizer(**configs)
        # bot parameters
        self.token = token
        self.bot = Bot(token=self.token)
        self.loop = asyncio.new_event_loop()
        self.configs = configs[self.type]['params']
        self.chat_ids = self.configs['chat_ids']
        self.message_thread_ids = self.configs['message_thread_ids']
        # get lists of favorite patterns, excahnges and chats
        self.favorite_exchanges = self.configs['favorite_exchanges']
        self.favorite_patterns = self.configs['favorite_patterns']
        self.favorite_chat_ids = self.configs['favorite_chat_ids']
        self.favorite_message_thread_ids = self.configs['favorite_message_thread_ids']
        # max number of candles after which message still could be sent
        self.min_prev_candle_limit = self.configs.get('min_prev_candle_limit', 3)
        self.min_prev_candle_limit_higher = self.configs.get('min_prev_candle_limit_higher', 2)
        # max number of notifications that could be sent at once
        self.max_notifications_in_row = self.configs.get('self.max_notifications_in_row', 3)
        # list of notifications
        self.notification_list = list()
        # dataframe for storing of notification history
        self.notification_df = pd.DataFrame(columns=['time', 'sig_type', 'ticker', 'timeframe', 'pattern'])
        # set of images to delete
        self.images_to_delete = set()
        # dictionary that is used to determine too late signals according to current work_timeframe
        self.timeframe_div = configs['Data']['Basic']['params']['timeframe_div']
        # get work and higher timeframes
        self.higher_tf_patterns = configs['Higher_TF_indicator_list']
        self.work_timeframe = configs['Timeframes']['work_timeframe']
        self.higher_timeframe = configs['Timeframes']['higher_timeframe']
        # get low and high confident bounds for AI model predictions
        # threshold that filters model predictions with low confidence
        self.pred_buy_thresh = self.configs['pred_buy_thresh']
        self.pred_sell_thresh = self.configs['pred_sell_thresh']

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
            application = Application.builder().token(self.token).build()
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
                                     timeframe: str, sig_pattern: str) -> bool:
        """ Check if previous notifications wasn't send short time before """
        tmp = self.notification_df[
                                    (self.notification_df['sig_type'] == sig_type) &
                                    (self.notification_df['ticker'] == ticker) &
                                    (self.notification_df['timeframe'] == timeframe) &
                                    (self.notification_df['pattern'] == sig_pattern)
                                   ]
        if tmp.shape[0] > 0:
            latest_time = tmp['time'].max()
            if set(sig_pattern.split('_')).intersection(set(self.higher_tf_patterns)):
                if (sig_time - latest_time).total_seconds() < self.timeframe_div[self.higher_timeframe] * \
                        self.min_prev_candle_limit_higher:
                    return False
            else:
                if (sig_time - latest_time).total_seconds() < self.timeframe_div[self.work_timeframe] * \
                        self.min_prev_candle_limit:
                    return False
        return True

    def check_multiple_notifications(self, sig_time: pd.Timestamp, sig_type: str, ticker: str,
                                     sig_pattern: str) -> list:
        """ Check if signals from different patterns appeared not too long time ago """
        if self.notification_df.shape[0] > 0:
            tmp = self.notification_df[
                                        (self.notification_df['sig_type'] == sig_type) &
                                        (self.notification_df['ticker'] == ticker)
                                       ]
            # find time difference between current signal and previous signals
            tmp['time_diff_sec'] = (sig_time - tmp['time']).dt.total_seconds()
            tmp = tmp[(tmp['time_diff_sec'] > 0) &
                      (tmp['time_diff_sec'] <= self.timeframe_div[self.higher_timeframe] * 1.5)]
            if tmp.shape[0] > 0:
                patterns = tmp['pattern'].to_list()
                # if signals from different patterns appeared not so much time ago - send the list of these patterns
                if set(patterns).difference({sig_pattern}):
                    res = sorted(list(set(patterns + [sig_pattern])))
                    if res != ['STOCH_RSI', 'STOCH_RSI_Trend']:
                        return res
        return []

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
                for i, message in message_dict[pattern][ttype]:
                    self.send_notification(message)
        # clear all sent notifications
        self.notification_list[:n_len] = []

    @staticmethod
    def clean_ticker(ticker: str) -> str:
        """ Clean ticker of not necessary symbols (2U, 1000, 10000, SWAP, -, etc.) """
        if not ticker.startswith('1INCH') and not ticker.startswith('3P'):
            ticker = re.sub(r'\b\d+', '', ticker)
        ticker = re.sub('SWAP', '', ticker)
        ticker = re.sub('-', '', ticker)
        ticker = re.sub('_', '', ticker)
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
        # get price prediction of model
        prediction = message[9]
        # get chat and thread ids
        chat_id = self.chat_ids[sig_pattern]
        message_thread_id = self.message_thread_ids.get(f'{sig_pattern}_{sig_type}', None)
        if message_thread_id is not None:
            message_thread_id = int(message_thread_id)
        price = self.database[message[0]][timeframe]['data'][sig_type]['close'].iloc[-1]
        price = self.round_price(price)
        # Check if the same message wasn't send short time ago
        if self.check_previous_notifications(sig_time, sig_type, ticker, timeframe, sig_pattern):
            # create image and return path to it
            sig_img_path = self.add_plot(message)
            # Form text message
            cleaned_ticker = self.clean_ticker(ticker)
            text = f'#{cleaned_ticker[:-4]}\n'
            text += ' + '.join(sig_pattern.split('_')) + '\n'
            text += f'Price / Цена: ${price}\n'
            if sig_pattern == 'HighVolume':
                pass
            elif sig_type == 'buy':
                text += 'Buy / Покупка\n'
            else:
                text += 'Sell / Продажа\n'
            text += 'Exchanges / Биржи:\n'
            for exchange in sig_exchanges:
                text += f' • {exchange}\n'
            text += 'TradingView:\n'
            text += f"https://tradingview.com/symbols/{cleaned_ticker}\n"
            if cleaned_ticker[:-4] != 'BTC':
                text += f'{cleaned_ticker[:-4]}/BTC:\n'
                text += f"https://ru.tradingview.com/symbols/{cleaned_ticker[:-4]}BTC\n"
            # send ML model prediction
            if sig_type == 'buy':
                pred_thresh = self.pred_buy_thresh
                if prediction >= self.pred_buy_thresh:
                    text += 'Buy AI confidence / Уверенность AI:\n'
                    text += f'{round(prediction * 100, 0)}%'
            else:
                pred_thresh = self.pred_sell_thresh
                if prediction >= self.pred_sell_thresh:
                    text += 'AI confidence / Уверенность AI:\n'
                    text += f'{round(prediction * 100, 0)}%'
            # Send message + image
            if sig_img_path:
                # if exchange is in the list of favorite exchanges and pattern is in list of your favorite patterns
                # send the signal to special group
                if set(sig_exchanges).intersection(set(self.favorite_exchanges)) and \
                        sig_pattern in self.favorite_patterns and prediction >= pred_thresh:
                    favorite_chat_id = self.favorite_chat_ids[sig_pattern]
                    favorite_message_thread_id = self.favorite_message_thread_ids.get(f'{sig_pattern}_{sig_type}', None)
                    if favorite_message_thread_id is not None:
                        favorite_message_thread_id = int(favorite_message_thread_id)
                    self.send_photo(favorite_chat_id, favorite_message_thread_id, sig_img_path, text)
                self.send_photo(chat_id, message_thread_id, sig_img_path, text)
            time.sleep(0.5)
        patterns = self.check_multiple_notifications(sig_time, sig_type, ticker, sig_pattern)
        if patterns:
            text = self.send_notification_for_multiple_signals(sig_type, ticker, sig_exchanges, patterns, price)
            chat_id = self.chat_ids['Multiple_Patterns']
            message_thread_id = self.message_thread_ids.get('Multiple_Patterns', None)
            self.send_message(chat_id, message_thread_id, text)
        self.add_to_notification_history(sig_time, sig_type, ticker, timeframe, sig_pattern)
        self.delete_images()

    def round_price(self, price: float) -> float:
        """ Function for price rounding """
        if price > 1:
            price = round(price, 3)
        else:
            price = round(price, 9)
        return price

    def send_notification_for_multiple_signals(self, sig_type: str, ticker: str, sig_exchanges: list, patterns: list,
                                               price: float):
        """ Send notifications in case of multiple signals from different patterns
            but for the same ticker and trade type """
        cleaned_ticker = self.clean_ticker(ticker)
        text = f'#{cleaned_ticker[:-4]}\n'
        text += f'Price / Цена: ${price}\n'
        if sig_type == 'buy':
            text += 'Buy / Покупка\n'
        else:
            text += 'Sell / Продажа\n'
        text += 'Patterns / Паттерны:\n'
        for pattern in patterns:
            text += f' • {pattern}\n'
        text += 'Exchanges / Биржи:\n'
        for exchange in sig_exchanges:
            text += f' • {exchange}\n'
        text += 'TradingView:\n'
        text += f"https://tradingview.com/symbols/{cleaned_ticker}\n"
        if cleaned_ticker[:-4] != 'BTC':
            text += f'{cleaned_ticker[:-4]}/BTC:\n'
            text += f"https://ru.tradingview.com/symbols/{cleaned_ticker[:-4]}BTC"
        return text

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

    telegram_bot = TelegramBot(token=telegram_token, database=None, **configs)
    telegram_bot.polling()
