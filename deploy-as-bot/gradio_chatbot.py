"""

deploy-as-bot\gradio_chatbot.py

A system, method for deploying to Gradio. Gradio is a basic "deploy" interface which allows for other users to test your model from a web URL. It also enables some basic functionality like user flagging for weird responses.
Note that the URL is displayed once the script is run. 

Set the working directory to */deploy-as-bot in terminal before running.

"""
import os
import sys
from os.path import dirname

# add the path to the script to the sys.path
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
        return clean(
            qphrase
        )  # fallback is to return the cleaned up version of the message


def ask_gpt(message: str, sender: str = ""):
    """
    ask_gpt - queries the relevant model with a prompt message and (optional) speaker name.
    nnote this version is modified w.r.t gradio local server deploy

    Args:
        message (str): prompt message to respond to
        sender (str, optional): speaker aka who said the message. Defaults to "".

    Returns:
        [str]: [model response as a string]
    """
    st = time.time()
    prompt = clean(message)  # clean user input
    prompt = prompt.strip()  # get rid of any extra whitespace
    if len(prompt) > 100:
        prompt = prompt[:100]  # truncate
    sender = clean(sender.strip())
    if len(sender) > 2:
        try:
            prompt_speaker = clean(sender)
        except:
            prompt_speaker = None  # fallback
    else:
        prompt_speaker = None  # fallback

    resp = query_gpt_model(
        folder_path=model_loc,
        prompt_msg=prompt,
        speaker=prompt_speaker,
        kparam=150,  # top k responses
        temp=0.75,  # temperature
        top_p=0.65,  # nucleus sampling
    )
    bot_resp = gramformer_correct(
        corrector, qphrase=resp["out_text"]
    )  # correct grammar
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
    history = gr.get_state() or []
    response = ask_gpt(message, sender=first_and_last_name)
    history.append(("You: " + message, " GPT-Model: " + response + " [end] "))
    gr.set_state(history)  # save the history
    html = ""
    for user_msg, resp_msg in history:
        html += f"{user_msg}"
        html += f"{resp_msg}"
    html += ""
    return html


def get_parser():
    """
    get_parser - a helper function for the argparse module

    Returns:
        [argparse.ArgumentParser]: [the argparser relevant for this script]
    """

    parser = argparse.ArgumentParser(
        description="host a chatbot on gradio",
    )
    parser.add_argument(
        "--model",
        required=False,
        type=str,
        default="GPT2_trivNatQAdailydia_774M_175Ksteps",  # folder name of model
        help="folder - with respect to git directory of your repo that has the model files in it (pytorch.bin + "
        "config.json). No models? Run the script download_models.py",
    )

    parser.add_argument(
        "--gram-model",
        required=False,
        type=str,
        default="prithivida/grammar_error_correcter_v1",  # huggingface model
        help="text2text generation model ID from huggingface for the model to correct grammar",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    default_model = str(args.model)
    model_loc = cwd.parent / default_model
    model_loc = str(model_loc.resolve())
    gram_model = args.gram_model

    # init items for the pipeline
    iface = gr.Interface(
        chat,
        inputs=["text", "text"],
        outputs="html",
        title=f"GPT-Chatbot Demo: {default_model} Model",
        description=f"A basic interface with a GPT2-based model, specifically {default_model}. Treat it like a friend!",
        article="**Important Notes & About:**\n"
        "1. the model can take up to 60 seconds to respond sometimes, patience is a virtue.\n"
        "2. entering a username is completely optional.\n"
        "3. the model started from a pretrained checkpoint, and was trained on several different datasets. Anything it says sshould be fact-checked before being regarded as a true statement.\n ",
        css="""
        .chatbox {display:flex;flex-direction:column}
        .user_msg, .resp_msg {padding:4px;margin-bottom:4px;border-radius:4px;width:80%}
        .user_msg {background-color:cornflowerblue;color:white;align-self:start}
        .resp_msg {background-color:lightgray;align-self:self-end}
    """,
        allow_screenshot=True,
        allow_flagging=True,  # allow users to flag responses as inappropriate
        flagging_dir="gradio_data",
        flagging_options=[
            "great response",
            "doesn't make sense",
            "bad/offensive response",
        ],
        enable_queue=True,  # allows for dealing with multiple users simultaneously
        theme="darkhuggingface",
    )

    corrector = pipeline("text2text-generation", model=gram_model, device=-1)
    print("Finished loading the gramformer model - ", datetime.now())
    print(f"using model stored here: \n {model_loc} \n")

    # launch the gradio interface and start the server
    iface.launch(share=True)
