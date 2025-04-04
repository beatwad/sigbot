"""
This module provides the implementation of a Telegram bot for
sending notifications and managing trade modes.
It utilizes the Python Telegram Bot (PTB) API for interacting
with Telegram and can send messages, photos, and notifications
to various chats and threads.
It also handles visualization of trading signals and their corresponding plots.
"""

import asyncio
import re
import time
from multiprocessing.synchronize import Lock
from os import remove
from typing import Set

import pandas as pd
import telegram
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from loguru import logger
from log.log import exception
from visualizer.visualizer import Visualizer

# Disable SettingWithCopyWarning because it's not necessary here
pd.options.mode.chained_assignment = None


ERROR_CHAT_ID = None


class TelegramBot:
    """
    Bot for sending messages to Telegram

    Attributes
    ----------
    token : str
        The bot's token for authentication.
    database : Union[Dict, None]
        A dictionary representing the ticker database.
    trade_mode : list
        A list containing the current trade mode state
        (bot trade enabled / disabled).
    locker : multiprocessing.Lock
        A lock for managing access to shared resources.
    configs : dict
        Configuration parameters for the bot,
        including chat IDs and other settings.
    """

    type = "Telegram"

    def __init__(
        self,
        token: str,
        database: dict,
        trade_mode: list,
        locker: Lock,
        **configs,
    ):
        # trade mode - if it's activated,
        # bot will use its own signals to trade on exchange
        self.trade_mode = trade_mode
        self.locker = locker
        # load default value of trade mode
        with locker:
            self.trade_mode[0] = configs["Main"]["params"]["trade_mode"]
        # model prediction threshold
        self.pred_thresh = configs["Model"]["params"]["pred_thresh"]
        # ticker database
        self.database = database
        # visualizer class
        self.visualizer = Visualizer(**configs)
        # bot parameters
        self.token = token
        self.bot = Bot(token=self.token)
        self.loop = asyncio.new_event_loop()
        self.configs = configs[self.type]["params"]
        self.chat_ids = self.configs["chat_ids"]
        global ERROR_CHAT_ID  # pylint: disable=W0603
        ERROR_CHAT_ID = self.chat_ids["Errors"]
        self.message_thread_ids = self.configs["message_thread_ids"]
        # get lists of favorite patterns, excahnges and chats
        self.favorite_exchanges = self.configs["favorite_exchanges"]
        self.favorite_patterns = self.configs["favorite_patterns"]
        self.favorite_chat_ids = self.configs["favorite_chat_ids"]
        self.favorite_message_thread_ids = self.configs["favorite_message_thread_ids"]
        # max number of candles after which message still could be sent
        self.min_prev_candle_limit = self.configs.get("min_prev_candle_limit", 3)
        self.min_prev_candle_limit_higher = self.configs.get("min_prev_candle_limit_higher", 2)
        # max number of notifications that could be sent at once
        self.max_notifications_in_row = self.configs.get("self.max_notifications_in_row", 3)
        # set of images to delete
        self.images_to_delete: Set[str] = set()
        # dictionary that is used to determine
        # too late signals according to current work_timeframe
        self.timeframe_div = configs["Data"]["Basic"]["params"]["timeframe_div"]
        # get work and higher timeframes
        self.higher_tf_indicator_set = {
            i for i in configs["Higher_TF_indicator_list"] if i != "Trend"
        }
        self.work_timeframe = configs["Timeframes"]["work_timeframe"]
        self.higher_timeframe = configs["Timeframes"]["higher_timeframe"]

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Enable trade mode."""
        if self.trade_mode[0] == 0:
            with self.locker:
                self.trade_mode[0] = 1
            text = "Trade mode is on"
            await update.message.reply_text(text)
        else:
            text = "Trade mode is already on"
            await update.message.reply_text(text)

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Disable trade mode"""
        if self.trade_mode[0] == 1:
            with self.locker:
                self.trade_mode[0] = 0
            text = "Trade mode is off"
            await update.message.reply_text(text)
        else:
            text = "Trade mode is already off"
            await update.message.reply_text(text)

    @staticmethod
    async def help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Inform user about what this bot can do"""
        text = (
            "Available commands:\n"
            "/start\n"
            "/stop\n"
            "/id\n"
            "Enter any other text to get the status"
        )
        await update.message.reply_text(text)

    @staticmethod
    async def get_chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Print chat id in response to user message"""
        chat_id = update.effective_chat["id"]
        message_thread_id = update.message.message_thread_id
        text = f"ID of this chat: {chat_id}, ID of this topic: {message_thread_id}"
        x = await context.bot.send_message(
            chat_id=chat_id, message_thread_id=message_thread_id, text=text
        )
        return x

    async def get_trade_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Return bot trade state"""
        chat_id = update.effective_chat["id"]
        message_thread_id = update.message.message_thread_id
        if self.trade_mode[0] == 1:
            text = "Trade mode is on"
        else:
            text = "Trade mode is off"
        x = await context.bot.send_message(
            chat_id=chat_id, message_thread_id=message_thread_id, text=text
        )
        return x

    @staticmethod
    async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log the error and send a telegram message to notify the developer."""
        # Log the error before we do anything else,
        # so we can see it even if something breaks.
        logger.error("Exception while handling an update:", exc_info=context.error)

        # Build the message with about what happened.
        message = "An exception occurred while PTB was handling an update"

        # Finally, send the message
        await context.bot.send_message(chat_id=ERROR_CHAT_ID, text=message)

    @exception
    def polling(self) -> None:
        """Start the bot polling"""
        # Create the Application and pass it your bot's token.
        application = (
            Application.builder()
            .token(self.token)
            .read_timeout(60)
            .get_updates_read_timeout(60)
            .build()
        )
        # on non command i.e. message - echo the message on Telegram
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.get_trade_mode)
        )
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("stop", self.stop))
        application.add_handler(CommandHandler("id", self.get_chat_id))
        application.add_handler(CommandHandler("help", self.help))
        # Run the bot until the user presses Ctrl-C
        application.run_polling()

    @staticmethod
    def process_ticker(ticker: str) -> str:
        """
        Bring ticker to a more convenient view.

        Parameters
        ----------
        ticker : str
            The original ticker string.

        Returns
        -------
        str
            The processed ticker string.
        """
        if "-" in ticker:
            return ticker
        if "/" in ticker:
            ticker = ticker.replace("/", "-")
            return ticker
        ticker = ticker[:-4] + "-" + ticker[-4:]
        return ticker

    @staticmethod
    def clean_ticker(ticker: str) -> str:
        """
        Clean ticker from unnecessary symbols (SWAP, -, _, etc.).

        Parameters
        ----------
        ticker : str
            The original ticker string.

        Returns
        -------
        str
            The cleaned ticker string.
        """
        # if ticker has 1000 in its name -
        # apparently it belongs to perpetual futures, add .P to it's name
        if "1000" in ticker:
            ticker = ticker + ".P"
        if not ticker.startswith("SWAP"):
            ticker = re.sub("SWAP", "", ticker)
        ticker = re.sub("-", "", ticker)
        ticker = re.sub("_", "", ticker)
        return ticker

    def add_plot(self, message: list) -> str:
        """
        Generate plot, save it to file and add its filepath to the signal point data.

        Parameters
        ----------
        message : list
            A list containing message details.

        Returns
        -------
        str
            The identifier of the stored plot/message.
        """
        sig_img_path = self.visualizer.create_plot(self.database, message, levels=[])
        # add image to set of images which we are going to delete
        self.images_to_delete.add(sig_img_path)
        return sig_img_path

    def send_notification(self, message: list) -> None:
        """
        Send notification regarding a specific ticker and message.

        Parameters
        ----------
        message : str
            The notification message.
        """
        # Get info from signal
        ticker = self.process_ticker(message[0])
        timeframe, sig_type, sig_pattern, sig_exchanges = (
            message[1],
            message[3],
            message[5],
            message[7],
        )
        # get chat and thread ids
        chat_id = self.chat_ids[sig_pattern]
        message_thread_id = self.message_thread_ids.get(f"{sig_pattern}_{sig_type}", None)
        if message_thread_id is not None:
            message_thread_id = int(message_thread_id)
        # get price and round it
        price = self.database[message[0]][timeframe]["data"][sig_type]["close"].iloc[-1]
        price = self.round_price(price)
        # create image and return path to it
        sig_img_path = self.add_plot(message)
        # Form text message
        cleaned_ticker = self.clean_ticker(ticker)
        if cleaned_ticker.endswith(".P"):
            text = f"#{cleaned_ticker[:-6]}\n"
        else:
            text = f"#{cleaned_ticker[:-4]}\n"
        # remove Volume24 word from pattern name
        cleaned_sig_pattern = sig_pattern[:-9] if sig_pattern.endswith("Volume24") else sig_pattern
        # create message
        text += " + ".join(cleaned_sig_pattern.split("_")) + "\n"
        text += f"Price / Цена: ${price}\n"
        # we do not need Volume24 in pattern's name
        if sig_pattern != "HighVolume":
            if sig_type == "buy":
                text += "Buy / Покупка\n"
            else:
                text += "Sell / Продажа\n"
        text += "Exchanges / Биржи:\n"
        for exchange in sig_exchanges:
            text += f" • {exchange}\n"
        text += "TradingView:\n"
        text += f"https://tradingview.com/symbols/{cleaned_ticker}\n"
        if cleaned_ticker[:-4] != "BTC" and not cleaned_ticker.endswith(".P"):
            text += f"{cleaned_ticker[:-4]}/BTC:\n"
            text += f"https://ru.tradingview.com/symbols/{cleaned_ticker[:-4]}BTC\n"
        # get price prediction of model
        prediction = message[9]
        # send ML model prediction
        if prediction > self.pred_thresh:
            text += "AI confidence / Уверенность AI:\n"
            text += f"{round(prediction * 100, 2)}%"
        # Send message + image
        if sig_img_path:
            # if exchange is in the list of favorite exchanges and
            # pattern is in list of your favorite patterns -
            # send the signal to special group
            logger.info(
                f"Telegram: exchanges - {sig_exchanges}, "
                f"pattern - {sig_pattern}, prediction - {prediction}"
            )
            if (
                set(sig_exchanges).intersection(set(self.favorite_exchanges))
                and sig_pattern in self.favorite_patterns
                and prediction > self.pred_thresh
            ):
                favorite_chat_id = self.favorite_chat_ids[sig_pattern]
                favorite_message_thread_id = self.favorite_message_thread_ids.get(
                    f"{sig_pattern}_{sig_type}", None
                )
                logger.info(
                    f"Telegram: fav chat id - {favorite_chat_id}, "
                    f"fav thread id - {favorite_message_thread_id}"
                )
                if favorite_message_thread_id is not None:
                    favorite_message_thread_id = int(favorite_message_thread_id)
                self.send_photo(favorite_chat_id, favorite_message_thread_id, sig_img_path, text)
            self.send_photo(chat_id, message_thread_id, sig_img_path, text)
        time.sleep(0.5)
        self.delete_images()

    @staticmethod
    def round_price(price: float) -> float:
        """
        Function for price rounding.

        Parameters
        ----------
        price : float
            The ticker price.

        Returns
        -------
        float
            Rounded ticker price.
        """
        if price > 1:
            price = round(price, 3)
        else:
            price = round(price, 9)
        return price

    @staticmethod
    async def bot_send_message(
        bot: Bot, chat_id: str, message_thread_id: int, text: str
    ) -> telegram.Message:
        """
        Send a message via bot.

        Parameters
        ----------
        bot : Bot
            The telegram bot instance.
        chat_id : str
            The chat ID where the message will be sent.
        message_thread_id : int
            The ID of the message thread.
        text : str
            The message text to be sent.

        Returns
        -------
        telegram.Message
            Sent message.
        """
        return await bot.send_message(
            chat_id=chat_id, message_thread_id=message_thread_id, text=text
        )

    @exception
    def send_message(self, chat_id: str, message_thread_id: int, text: str):
        """
        Send a message to a specified chat and thread.

        Parameters
        ----------
        chat_id : str
            The chat ID where the message will be sent.
        message_thread_id : int
            The ID of the message thread.
        text : str
            The message text to be sent.
        """
        tasks = [
            self.loop.create_task(self.bot_send_message(self.bot, chat_id, message_thread_id, text))
        ]
        self.loop.run_until_complete(asyncio.wait(tasks))

    @staticmethod
    async def bot_send_photo(
        bot: Bot, chat_id: str, message_thread_id: int, img_path: str, text: str
    ) -> telegram.Message:
        """Send a photo via bot.

        Parameters
        ----------
        bot : Bot
            The telegram bot instance.
        chat_id : str
            The chat ID where the photo will be sent.
        message_thread_id : int
            The ID of the message thread.
        img_path : str
            The file path to the image to be sent.
        text : str
            The caption text for the photo.

        Returns
        -------
        telegram.Message
            Sent photo message.
        """
        return await bot.send_photo(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            photo=open(img_path, "rb"),
            caption=text,
        )

    @exception
    def send_photo(self, chat_id: str, message_thread_id: int, img_path: str, text: str):
        """
        Send a photo to a specified chat and thread.

        Parameters
        ----------
        chat_id : str
            The chat ID where the photo will be sent.
        message_thread_id : int
            The ID of the message thread.
        img_path : str
            The file path to the image to be sent.
        text : str
            The caption text for the photo.
        """
        tasks = [
            self.loop.create_task(
                self.bot_send_photo(self.bot, chat_id, message_thread_id, img_path, text)
            )
        ]
        self.loop.run_until_complete(asyncio.wait(tasks))

    def delete_images(self):
        """
        Remove images after sending them.
        This method deletes images that are no longer needed from the local storage.
        """
        while self.images_to_delete:
            img_path = self.images_to_delete.pop()
            try:
                remove(img_path)
            except FileNotFoundError:
                pass


# if __name__ == '__main__':
#     from dotenv import load_dotenv, find_dotenv
#     from os import environ
#     # here we load environment variables from .env, must be called before init. class
#     load_dotenv(find_dotenv('../.env'), verbose=True)
#     telegram_token = os.getenv("TELEGRAM_TOKEN")

#     configs = ConfigFactory.factory(environ).configs

#     telegram_bot = TelegramBot(token=telegram_token, database=None,
#                                trade_mode=True, locker=None, **configs)
#     telegram_bot.polling()
