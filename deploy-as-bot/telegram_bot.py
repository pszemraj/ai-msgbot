"""
Basic GPT-2 telegram bot
you need to have your own token to create and run one - here it is in my env variables
"""
import sys
sys.path.append("..")

import logging
import os
import time
import warnings

from cleantext import clean
from symspellpy import SymSpell
from telegram.ext import CommandHandler
from telegram.ext import Filters, MessageHandler
from telegram.ext import Updater
from transformers import pipeline

from ai_single_response import query_gpt_peter

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")
model_loc = os.path.join(os.getcwd(), "../gp2_DDandPeterTexts_774M_73Ksteps")
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def symspell_correct(speller, qphrase: str):
    suggestions = speller.lookup_compound(
        clean(qphrase), max_edit_distance=2, ignore_non_words=True
    )
    if len(suggestions) < 1:
        return qphrase
    else:
        first_result = suggestions[0]
        return first_result._term


def gramformer_correct(corrector, qphrase: str):
    try:
        corrected = corrector(
            clean(qphrase), return_text=True, clean_up_tokenization_spaces=True
        )
        return corrected[0]["generated_text"]
    except:
        print("NOTE - failed to correct with gramformer")
        return clean(qphrase)


def start(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="send me texts and I answer.. after like 30-45 seconds",
    )


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text(
        "idk - there are not any options rn, just send normal texts"
    )


def echo(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)


def ask_gpt(update, context):
    st = time.time()
    prompt = clean(update.message.text)  # clean user input
    prompt = prompt.strip()  # get rid of any extra whitespace
    if len(prompt) > 100:
        prompt = prompt[:100]  # truncate
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="ur prompt is quite long, truncating to first 100 characters",
        )
    try:
        firstname = clean(update.message.chat.first_name)
        lastname = clean(update.message.chat.last_name)
        prompt_speaker = firstname + " " + lastname
    except:
        # there was some issue getting that info, whatever
        prompt_speaker = None
    context.bot.send_message(
        chat_id=update.effective_chat.id, text="... neurons are working ..."
    )
    resp = query_gpt_peter(
        folder_path=model_loc,
        prompt_msg=prompt,
        speaker=prompt_speaker,
        kparam=125,
        temp=0.75,
        top_p=0.65,  # latest hyperparam search results 21-oct
    )
    if use_gramformer:
        bot_resp = gramformer_correct(corrector, qphrase=resp["out_text"])
    else:
        bot_resp = symspell_correct(sym_spell, qphrase=resp["out_text"])
    rt = round(time.time() - st, 2)
    print(f"took {rt} sec to respond")
    context.bot.send_message(chat_id=update.effective_chat.id, text=bot_resp)


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def unknown(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id, text="m8, I didn't understand that command."
    )


use_gramformer = True  # TODO change this to a default argument and use argparse
gram_model = "prithivida/grammar_error_correcter_v1"
dictionary_path = r"../symspell_rsc/frequency_dictionary_en_82_765.txt"  # from repo root
bigram_path = (
    r"symspell_rsc/frequency_bigramdictionary_en_243_342.txt"  # from repo root
)

if __name__ == "__main__":
    # get token
    env_var = os.environ
    my_vars = dict(env_var)
    my_token = my_vars["GPTPETER_BOT"]

    # load on bot start so does not have to reload
    if use_gramformer:
        print("using gramformer..")
        corrector = pipeline("text2text-generation", model=gram_model, device=-1)
    else:
        print("using default SymSpell..")
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    updater = Updater(token=my_token, use_context=True)

    dispatcher = updater.dispatcher
    start_handler = CommandHandler("start", start)
    dispatcher.add_handler(start_handler)

    help_handler = CommandHandler("help", help)
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
