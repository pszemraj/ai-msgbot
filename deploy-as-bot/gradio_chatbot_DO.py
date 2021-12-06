"""

deploy-as-bot\gradio_chatbot.py

A system, method for deploying to Gradio. Gradio is a basic "deploy" interface which allows for other users to test your model from a web URL. It also enables some basic functionality like user flagging for weird responses.
Note that the URL is displayed once the script is run. 

Set the working directory to */deploy-as-bot in terminal before running.

"""
import os
import sys
from os.path import dirname

sys.path.append(dirname(dirname(os.path.abspath(__file__))))

import gradio as gr
import logging
import argparse
import time
import warnings
from pathlib import Path
from cleantext import clean
from transformers import pipeline
from datetime import datetime
from ai_single_response import query_gpt_model
#from gradio.networking import get_state, set_state
from flask import Flask, request, session, jsonify, abort, send_file, render_template, redirect

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")

logging.basicConfig()
cwd = Path.cwd()
my_cwd = str(cwd.resolve())  # string so it can be passed to os.path() objects


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


def ask_gpt(message: str, sender: str = ""):
    """
    ask_gpt - queries the relevant model with a prompt message and (optional) speaker name

    Args:
        message (str): prompt message to respond to
        sender (str, optional): speaker aka who said the message. Defaults to "".

    Returns:
        [str]: [model response as a string]
    """
    st = time.time()
    prompt = clean(message)  # clean user input
    prompt = prompt.strip()  # get rid of any extra whitespace
    if len(prompt) > 200:
        prompt = prompt[-200:]  # truncate
    sender = clean(sender.strip())
    if len(sender) > 2:
        try:
            prompt_speaker = clean(sender)
        except:
            # there was some issue getting that info, whatever
            prompt_speaker = None
    else:
        prompt_speaker = None

    resp = query_gpt_model(
        folder_path=model_loc,
        prompt_msg=prompt,
        speaker=prompt_speaker,
        kparam=150,
        temp=0.75,
        top_p=0.65,  # optimize this with hyperparam search
    )
    bot_resp = gramformer_correct(corrector, qphrase=resp["out_text"])
    rt = round(time.time() - st, 2)
    print(f"took {rt} sec to respond")

    return bot_resp


def chat(first_and_last_name, message):
    """
    chat - helper function that makes the whole gradio thing work.

    Args:
        first_and_last_name (str or None): [speaker of the prompt, if provided]
        message (str): [description]

    Returns:
        [str]: [returns an html string to display]
    """
    history = session.get("my_state") or []
    response = ask_gpt(message, sender=first_and_last_name)
    history.append((f"{first_and_last_name}: " + message, " GPT-Model: " + response)) #+ " [end] "))
    session["my_state"] = history
    session.modified = True
    html = "<div class='chatbot'>"
    for user_msg, resp_msg in history:
        html += f"<div class='user_msg'>{user_msg}</div>"
        html += f"<div class='resp_msg' style='color: black'>{resp_msg}</div>"
    html += "</div>"
    return html


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
        # "gp2_DDandPeterTexts_774M_73Ksteps", - from GPT-Peter
        default="GPT2_trivNatQAdailydia_774M_175Ksteps",
        help="folder - with respect to git directory of your repo that has the model files in it (pytorch.bin + "
        "config.json). No models? Run the script download_models.py",
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
    corrector = pipeline("text2text-generation", model=gram_model, device=-1)
    print("Finished loading the gramformer model - ", datetime.now())
    iface = gr.Interface(
        chat,
        inputs=["text", "text"],
        outputs="html",
        title="Real-Impact English Chat Demo 英语聊天演示",
        description="A basic interface with a 774M parameter model trained on general Q&A and conversation. Treat it like a friend!",
        article="**Important Notes & About:**\n"
        "1. the model can take up to 200 seconds to respond sometimes, patience is a virtue.\n"
        "2. entering a username is completely optional.\n"
        "3. the model was trained on several different datasets. Anything it says should be fact-checked before being regarded as a true statement.\n ",
        css="""
        .chatbox {display:flex;flex-direction:column}
        .user_msg, .resp_msg {padding:4px;margin-bottom:4px;border-radius:4px;width:80%}
        .user_msg {background-color:cornflowerblue;color:white;align-self:start}
        .resp_msg {background-color:lightgray;align-self:self-end}
    """,
        allow_screenshot=True,
        allow_flagging=False,
        flagging_dir="gradio_data",
        flagging_options=[
            "great response",
            "doesn't make sense",
            "bad/offensive response",
        ],
        enable_queue=True,  # allows for dealing with multiple users simultaneously
        theme="darkhuggingface",
        server_name="0.0.0.0",
    )
    iface.launch(share=True)
