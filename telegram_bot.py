"""
Basic GPT-2 telegram bot
you need to have your own token to create and run one - here it is in my env variables
"""

import logging
import os
import warnings

from cleantext import clean
from telegram.ext import CommandHandler
from telegram.ext import Filters, MessageHandler
from telegram.ext import Updater

# TODO: figure out how to get this to import correctly when it is inside ./telegram-bot
from ai_single_response import query_gpt_peter

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")
model_loc = os.path.join(os.getcwd(), "gpt2_std_gpu_774M_60ksteps")

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text="send me texts and I answer.. after like 30 seconds")


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('idk - there are not any options rn, just send normal texts')


def echo(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=update.message.text)


def ask_gpt(update, context):
    prompt = clean(update.message.text) # clean user input
    prompt = prompt.strip() # get rid of any extra whitespace
    if len(prompt) > 100:
        prompt = prompt[:100]  # truncate
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="ur prompt is quite long, truncating to first 100 characters")
    try:
        firstname = clean(update.message.chat.first_name)
        lastname = clean(update.message.chat.last_name)
        prompt_speaker = firstname + " " + lastname
    except:
        # there was some issue getting that info, whatever
        prompt_speaker = None
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text="... pondering (pls wait) ...")
    resp = query_gpt_peter(folder_path=model_loc, prompt_msg=prompt,
                           speaker=prompt_speaker, )
    bot_resp = resp["out_text"]
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=bot_resp)


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def unknown(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text="m8, I didn't understand that command.")


if __name__ == "__main__":
    # get token
    env_var = os.environ
    my_vars = dict(env_var)
    my_token = my_vars["GPTPETER_BOT"]
    updater = Updater(token=my_token, use_context=True)

    dispatcher = updater.dispatcher
    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    help_handler = CommandHandler('help', help)
    dispatcher.add_handler(help_handler)

    gpt_handler = MessageHandler(Filters.text & (~Filters.command), ask_gpt)
    dispatcher.add_handler(gpt_handler)

    unknown_handler = MessageHandler(Filters.command, unknown)
    dispatcher.add_handler(unknown_handler)

    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()
