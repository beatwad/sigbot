import sys
import glob
from time import sleep
from datetime import datetime
from os import environ, remove

# Set environment variable
environ["ENV"] = "debug"

from bot.bot import SigBot
from config.config import ConfigFactory

# Get configs
configs = ConfigFactory.factory(environ).configs


class Main:
    type = 'Main'

    def __init__(self, load_tickers=True, **configs):
        """ Initialize separate thread the Telegram bot, so it can work independently
            event for stopping bot thread """
        self.cycle_number = 1
        self.bot_cycle_length = configs[self.type]['params']['bot_cycle_length_sec']
        self.time_period = configs[self.type]['params']['time_period_minutes']
        self.cycle_length = configs[self.type]['params']['cycle_length_hours']
        self.first_cycle_qty_miss = configs[self.type]['params']['first_cycle_qty_miss']
        self.sigbot = SigBot(self, load_tickers=load_tickers, **configs)

    @staticmethod
    def check_time(dt, time_period):
        """ Check if time in minutes is multiple of a period """
        if dt.minute % time_period == 0:
            return True
        return False

    def cycle(self):
        try:
            dt1 = datetime.now()
            # if self.check_time(dt1, self.time_period) or self.cycle_number == 1:
            self.sigbot.main_cycle()
            dt2 = datetime.now()
            dtm, dts = divmod((dt2 - dt1).total_seconds(), 60)
            print(f'Cycle is {self.cycle_number}, time for the cycle (min:sec) - {int(dtm)}:{round(dts, 2)}')
            self.cycle_number += 1
            sleep(self.bot_cycle_length)
            # else:
            #     sleep(1)
        except (KeyboardInterrupt, SystemExit):
            # stop all exchange monitors
            self.sigbot.stop_monitors()
            # on interruption or exit stop Telegram module thread
            self.sigbot.telegram_bot.stopped.set()
            # delete everything in image directory on exit
            files = glob.glob('visualizer/images/*.png')
            for f in files:
                remove(f)
            # exit program
            sys.exit()


if __name__ == "__main__":
    load_tickers = True
    database, exchanges, tb_bot = None, None, None
    error_notification_sent = False

    while True:
        main = Main(load_tickers, **configs)
        # variables that is used to save some important sigbot object to prevent their recreation
        # and additional memory consumption; their values are saved on the first cycle and then are used
        # on the next cycles
        if load_tickers:
            database = main.sigbot.database
            exchanges = main.sigbot.exchanges
            tb_bot = main.sigbot.telegram_bot
            load_tickers = False
        else:
            main.sigbot.database = database
            main.sigbot.exchanges = exchanges
            main.sigbot.telegram_bot = tb_bot
        # restart the bot every 24 hours
        dt1 = dt2 = datetime.now()
        try:
            while int((dt2 - dt1).total_seconds() / 3600) <= main.cycle_length:
                main.cycle()
                dt2 = datetime.now()
        except (KeyboardInterrupt, SystemExit):
            pass
        except:
            if not error_notification_sent:
                text = f'Catch an exception: {sys.exc_info()[0]}'
                main.sigbot.telegram_bot.send_message(main.sigbot.telegram_bot.chat_ids['Errors'], text)
                error_notification_sent = True
            continue
