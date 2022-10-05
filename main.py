import sys
import glob
from bot.bot import SigBot
from time import sleep
from os import environ, remove
from datetime import datetime
from config.config import ConfigFactory

debug = False
# Set environment variable
environ["ENV"] = "development"
# Get configs
configs = ConfigFactory.factory(environ).configs


class Main:
    type = 'Main'

    def __init__(self, **configs):
        # initialize separate thread the Telegram bot, so it can work independently
        # event for stopping bot thread
        self.cycle_number = 1
        self.bot_cycle_length = configs[self.type]['params']['bot_cycle_length_sec']
        self.cycle_length = configs[self.type]['params']['cycle_length_hours']
        self.sigbot = SigBot(self, **configs)

    def cycle(self):
        try:
            dt1 = datetime.now()
            self.sigbot.main_cycle()
            dt2 = datetime.now()
            dtm, dts = divmod((dt2 - dt1).total_seconds(), 60)
            print(f'Cycle is {self.cycle_number}, time for the cycle (min:sec) - {int(dtm)}:{round(dts, 2)}')
            self.cycle_number += 1
            sleep(self.bot_cycle_length)
        except (KeyboardInterrupt, SystemExit):
            # stop all exchange monitors
            self.sigbot.stop_monitors()
            # on interruption or exit stop Telegram module thread
            self.sigbot.telegram_bot.stopped.set()
            # delete everything in image directory on exit
            files = glob.glob('visualizer/images/*')
            for f in files:
                remove(f)
            # exit program
            sys.exit()


if __name__ == "__main__":
    while True:
        main = Main(**configs)
        dt1 = datetime.now()
        dt2 = datetime.now()
        while int((dt2 - dt1).total_seconds() / 3600) <= main.cycle_length:
            main.cycle()
            dt2 = datetime.now()
