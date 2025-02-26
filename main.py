"""Main module of the program"""

import glob
import sys
from datetime import datetime
from os import environ, remove
from time import sleep
import time
from loguru import logger

from bot.bot import SigBot
from config.config import ConfigFactory
from log.log import format_exception

logger.add("log/log.log")

class Main:
    """
    Initialize the Main class.

    Attributes
    ----------
    load_tickers : bool, optional
        Whether to load tickers at startup (default is True).
    configs : dict
        Configuration parameters for the bot.
    """

    type = "Main"
    error_notification_sent = False

    def __init__(self, load_tickers: bool = True, **configs):
        self.cycle_number = 1
        self.bot_cycle_length = configs[self.type]["params"]["bot_cycle_length_sec"]
        self.time_period = configs[self.type]["params"]["time_period_minutes"]
        self.cycle_length = configs[self.type]["params"]["cycle_length_hours"]
        self.first_cycle_qty_miss = configs[self.type]["params"]["first_cycle_qty_miss"]
        self.sigbot = SigBot(self, load_tickers=load_tickers, **configs)
        # this flag is used to close the bot only after it get new candle data
        self.new_data_flag = False

    @staticmethod
    def check_time(dt: datetime):
        """
        Check if the current time (in minutes) is a multiple of a specific period.

        Parameters
        ----------
        dt : datetime
            The current datetime object.

        Returns
        -------
        bool
            True if the time is a multiple of 60 minutes, otherwise False.
        """
        if (dt.hour * 60 + dt.minute) % 60 == 0:
            return True
        return False

    def cycle(self):
        """
        Main program cycle for processing bot logic.

        This method checks the time, runs the bot's main cycle, and handles
        exceptions such as termination and other errors. It logs the cycle duration
        and controls when to sleep and proceed to the next cycle.
        """
        _dt1 = datetime.now()
        if self.check_time(_dt1) or self.cycle_number == 1:
            print(_dt1, flush=True)
            _dt1 = datetime.now()
            try:
                self.sigbot.main_cycle()
            except (KeyboardInterrupt, SystemExit):
                # delete everything in image directory on exit
                files = glob.glob("visualizer/images/*.png")
                for f in files:
                    remove(f)
                # terminate Telegram bot process
                self.sigbot.telegram_bot_process.terminate()
                self.sigbot.telegram_bot_process.join()
                # exit program
                sys.exit()
            except BaseException:  # noqa
                if not self.error_notification_sent:
                    format_exception()
                    text = f"Catch an exception: {sys.exc_info()[1]}"
                    self.sigbot.telegram_bot.send_message(
                        self.sigbot.telegram_bot.chat_ids["Errors"], None, text
                    )
                    self.error_notification_sent = True
            _dt2 = datetime.now()
            dtm, dts = divmod((_dt2 - _dt1).total_seconds(), 60)
            print(
                f"Cycle is {self.cycle_number}, time for the cycle (min:sec) - "
                f"{int(dtm)}:{round(dts, 2)}",
                flush=True,
            )
            self.cycle_number += 1
            self.new_data_flag = True
            sleep(60)
        else:
            self.new_data_flag = False
            sleep(self.bot_cycle_length)

def main_cycle() -> None:
    # Get configs
    configs_ = ConfigFactory.factory(environ).configs
    print("Start of cycle", flush=True)
    logger.info("Start of cycle")
    dt1 = dt2 = datetime.now()
    # sigbot init
    main = Main(load_tickers=True, **configs_)
    # close the bot after some time (default is 24 hours) and
    # only after it get new candle data
    while (dt2 - dt1).total_seconds() // 3600 <= main.cycle_length or not main.new_data_flag:
        main.cycle()
        dt2 = datetime.now()
    print("End of cycle", flush=True)
    logger.info("End of cycle")
    # terminate Telegram bot process
    main.sigbot.telegram_bot_process.terminate()
    main.sigbot.telegram_bot_process.join()


if __name__ == "__main__":
    while True:
        main_cycle()
        time.sleep(10)
