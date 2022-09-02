import time
import logging
from telegram.ext import Updater
from telegram import Update
from telegram.ext import CallbackContext
from telegram.ext import CommandHandler

updater = Updater(token='5770186369:AAFrHs_te6bfjlHeD6mZDVgwvxGQ5TatiZA', use_context=True)

dispatcher = updater.dispatcher

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)


def start(update: Update, context: CallbackContext):
    context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")


start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

updater.start_polling()


def send_notification(notification) -> None:
    """ Add notification to notification list and send its items to chat """
    notification_list = ['test1', 'test2', 'test3']
    while notification_list:
        time.sleep(1)
        _message = notification_list.pop(0)
        print(_message)
        updater.bot.send_message(chat_id="-1001702067917", text=_message)

# updater.bot.send_message(chat_id="-1001702067917", text='test')
send_notification('')
