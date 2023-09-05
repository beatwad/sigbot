import sys
import glob
from time import sleep
from datetime import datetime
from os import environ, remove

# Set environment variable
environ["ENV"] = "1h_4h"

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
        """ Check if time in minutes is a multiple of a period """
        if (dt.hour * 60 + dt.minute) % 60 == 0:
            return True
        return False

    def cycle(self):
        try:
            dt1 = datetime.now()
            if self.check_time(dt1, self.time_period) or self.cycle_number == 1:
                print(dt1)
                dt1 = datetime.now()
                self.sigbot.main_cycle()
                dt2 = datetime.now()
                dtm, dts = divmod((dt2 - dt1).total_seconds(), 60)
                print(f'Cycle is {self.cycle_number}, time for the cycle (min:sec) - {int(dtm)}:{round(dts, 2)}')
                self.cycle_number += 1
                sleep(60)
            sleep(self.bot_cycle_length)
        except (KeyboardInterrupt, SystemExit):
            # stop all exchange monitors
            self.sigbot.stop_monitors()
            # delete everything in image directory on exit
            files = glob.glob('visualizer/images/*.png')
            for f in files:
                remove(f)
            # exit program
            sys.exit()


if __name__ == "__main__":
    error_notification_sent = False
    dt1 = dt2 = datetime.now()
    # sigbot init
    main = Main(load_tickers=True, **configs)
    # close the bot after some time (default is 24 hours)
    try:
        while int((dt2 - dt1).total_seconds() / 3600) <= main.cycle_length:
            main.cycle()
            dt2 = datetime.now()
    except (KeyboardInterrupt, SystemExit):
        pass
    except:
        if not error_notification_sent:
            text = f'Catch an exception: {sys.exc_info()[1]}'
            main.sigbot.telegram_bot.send_message(main.sigbot.telegram_bot.chat_ids['Errors'], None, text)
            error_notification_sent = True
