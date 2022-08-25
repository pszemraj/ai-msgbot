"""
Basic GPT-2 telegram bot

you need to have your own token to create and run one - this script loads and reads the user's environmental variables. This script assumes the API token is stored under "GPTFRIEND_BOT"

creating a bot: https://www.codementor.io/@karandeepbatra/part-1-how-to-create-a-telegram-bot-in-python-in-under-10-minutes-19yfdv4wrq

# TODO: add conversation flow + context from previous messages
"""
import argparse
import os
import sys
from os.path import dirname

sys.path.append(dirname(dirname(os.path.abspath(__file__))))

import logging
import time
import warnings

from symspellpy import SymSpell
from telegram.ext import CommandHandler
from telegram.ext import Filters, MessageHandler
from telegram.ext import Updater
from pathlib import Path
from transformers import pipeline

from ai_single_response import query_gpt_model
from utils import remove_trailing_punctuation, DisableLogger

with DisableLogger():
    from cleantext import clean

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")
cwd = Path.cwd()
my_cwd = str(cwd.resolve())  # string so it can be passed to os.path() objects

logging.basicConfig(
    filename=f"LOGFILE-{Path(__file__).stem}.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def symspell_correct(speller, qphrase: str):
    """
    symspell_correct corrects a string using symspellpy

    Args:
        speller (symspellpy.SymSpell): [symspellpy SymSpell class object, already instantiated]
        qphrase (str): [text to be corrected]

    Returns:
        [str]: [corrected text]
    """
    suggestions = speller.lookup_compound(
        clean(qphrase), max_edit_distance=2, ignore_non_words=True
    )
    if len(suggestions) < 1:
        return qphrase
    else:
        first_result = suggestions[0]
        return first_result._term


def gramformer_correct(corrector, qphrase: str):
    """
    gramformer_correct - correct a string using a text2textgen pipeline model from transformers

    Args:
        corrector (transformers.pipeline): [transformers pipeline object, already created w/ relevant model]
        qphrase (str): [text to be corrected]

    Returns:
        [str]: [corrected text]
    """

    try:
        corrected = corrector(
            clean(qphrase), return_text=True, clean_up_tokenization_spaces=True
        )
        return corrected[0]["generated_text"]
    except:
        print("NOTE - failed to correct with gramformer")
        return clean(qphrase)


def start(update, context):
    """instantiates telegram bot"""
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="A GPT chatbot model - send it messages as if to a friend.",
    )


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text(
        "There are no options at the moment, just send normal texts to the Bot. Note: 1) only messages w/ text are supported 2) if bot does not respond, it may be offline. Contact the bot manager."
    )


def echo(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)


def ask_gpt(update, context):
    """
    ask_gpt - queries the relevant gpt2 model and interfaces with Telegram

    Args:
        update (telegram class obj): [description]
        context (telegram class obj): [description]
    """
    st = time.time()
    prompt = clean(update.message.text)  # clean user input
    prompt = prompt.strip()  # get rid of any extra whitespace
    if len(prompt) > 100:
        prompt = prompt[:100]  # truncate
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="INFO: prompt is too long, truncating to first 100 chars",
        )
    try:
        firstname = clean(update.message.chat.first_name)
        lastname = clean(update.message.chat.last_name)
        prompt_speaker = firstname + " " + lastname
    except:
        # there was some issue getting that info, whatever
        prompt_speaker = None
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="... neurons are working ...",  # confirms receipt / running to user
    )
    resp = query_gpt_model(
        folder_path=model_loc,
        prompt_msg=prompt,
        speaker=prompt_speaker,
        kparam=125,
        temp=0.75,
        top_p=0.65,  # can be changed based on hyperparam desires
    )
    # now, actually respond from model
    if use_gramformer:
        bot_resp = gramformer_correct(corrector, qphrase=resp["out_text"])
    else:
        bot_resp = symspell_correct(sym_spell, qphrase=resp["out_text"])
    bot_resp = remove_trailing_punctuation(
        bot_resp
    )  # remove trailing punctuation to seem more natural
    rt = round(time.time() - st, 2)
    print(f"took {rt} sec to respond")
    context.bot.send_message(chat_id=update.effective_chat.id, text=bot_resp)


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def unknown(update, context):
    """Responds to unknown command"""

    context.bot.send_message(
        chat_id=update.effective_chat.id, text="Command not understood, sorry!"
    )


dictionary_path = (
    r"../symspell_rsc/frequency_dictionary_en_82_765.txt"  # from repo root
)
bigram_path = (
    r"symspell_rsc/frequency_bigramdictionary_en_243_342.txt"  # from repo root
)


def get_parser():
    """
    get_parser - a helper function for the argparse module

    Returns:
        [argparse.ArgumentParser]: [the argparser relevant for this script]
    """

    parser = argparse.ArgumentParser(
        description="submit a message and have a 774M parameter GPT model respond"
    )
    parser.add_argument(
        "--model",
        required=False,
        type=str,
        default="GPT2_trivNatQAdailydia_774M_175Ksteps",
        help="folder - with respect to git directory of your repo that has the model files in it (pytorch.bin + "
        "config.json). No models? Run the script download_models.py",
    )

    parser.add_argument(
        "--use-gramformer",
        default=True,
        action="store_false",
        help="passing this argument DEACTIVATES gramformer and switches to symspell",
    )
    parser.add_argument(
        "--gram-model",
        required=False,
        type=str,
        default="prithivida/grammar_error_correcter_v1",
        help="text2text generation model ID from huggingface for the model to correct grammar",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    default_model = str(args.model)
    model_loc = cwd.parent / default_model
    model_loc = str(model_loc.resolve())
    gram_model = args.gram_model
    print(f"using model stored here: \n {model_loc} \n")
    # get token
    env_var = os.environ
    my_vars = dict(env_var)
    my_token = my_vars["GPTFRIEND_BOT"]

    # load on bot start so does not have to reload
    use_gramformer = args.use_gramformer

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
