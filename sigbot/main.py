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
    error_notification_sent = False

    def __init__(self, load_tickers=True, **configs):
        """ Initialize separate thread the Telegram bot, so it can work independently
            event for stopping bot thread """
        self.cycle_number = 1
        self.bot_cycle_length = configs[self.type]['params']['bot_cycle_length_sec']
        self.time_period = configs[self.type]['params']['time_period_minutes']
        self.cycle_length = configs[self.type]['params']['cycle_length_hours']
        self.first_cycle_qty_miss = configs[self.type]['params']['first_cycle_qty_miss']
        self.sigbot = SigBot(self, load_tickers=load_tickers, **configs)
        # this flag is used to close the bot only after it get new candle data
        self.new_data_flag = False

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
                print(dt1, flush=True)
                dt1 = datetime.now()
                self.sigbot.main_cycle()
                dt2 = datetime.now()
                dtm, dts = divmod((dt2 - dt1).total_seconds(), 60)
                print(f'Cycle is {self.cycle_number}, time for the cycle (min:sec) - {int(dtm)}:{round(dts, 2)}',
                      flush=True)
                self.cycle_number += 1
                self.new_data_flag = True
                sleep(60)
            else:
                self.new_data_flag = False
            sleep(self.bot_cycle_length)
        except (KeyboardInterrupt, SystemExit):
            # delete everything in image directory on exit
            files = glob.glob('visualizer/images/*.png')
            for f in files:
                remove(f)
            # exit program
            sys.exit()
        except:
            if not self.error_notification_sent:
                text = f'Catch an exception: {sys.exc_info()[1]}'
                main.sigbot.telegram_bot.send_message(main.sigbot.telegram_bot.chat_ids['Errors'], None, text)
                self.error_notification_sent = True


if __name__ == "__main__":
    dt1 = dt2 = datetime.now()
    # sigbot init
    main = Main(load_tickers=True, **configs)
    # close the bot after some time (default is 24 hours) and only after it get new candle data
    while int((dt2 - dt1).total_seconds() / 3600) <= main.cycle_length or not main.new_data_flag:
        main.cycle()
        dt2 = datetime.now()
