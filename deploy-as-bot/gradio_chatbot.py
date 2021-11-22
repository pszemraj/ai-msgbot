"""

deploy-as-bot\gradio_chatbot.py

A system, method for deploying to Gradio. Gradio is a basic "deploy" interface which allows for other users to test your model from a web URL. It also enables some basic functionality like user flagging for weird responses.
Note that the URL is displayed once the script is run. 

"""
import os
import sys
from os.path import dirname

sys.path.append(dirname(dirname(os.path.abspath(__file__))))

import gradio as gr
import logging
import time
import warnings
from pathlib import Path
from cleantext import clean
from transformers import pipeline
from datetime import datetime
from ai_single_response import query_gpt_model

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")

logging.basicConfig()
# default_model = "gp2_DDandPeterTexts_774M_73Ksteps"
cwd = Path.cwd()
# model_loc = cwd.parent / default_model
# model_loc = str(model_loc.resolve())
# print(f"using model stored here: \n {model_loc} \n")
my_cwd = str(cwd.resolve())  # string so it can be passed to os.path() objects
gram_model = "prithivida/grammar_error_correcter_v1"


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
    if len(prompt) > 100:
        prompt = prompt[:100]  # truncate
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
        top_p=0.65,  # latest hyperparam search results 21-oct
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
    history = gr.get_state() or []
    response = ask_gpt(message, sender=first_and_last_name)
    history.append(("You: " + message, " GPT-Peter: " + response + " [end] "))
    gr.set_state(history)
    html = ""
    for user_msg, resp_msg in history:
        html += f"{user_msg}"
        html += f"{resp_msg}"
    html += ""
    return html


if __name__ == "__main__":
    default_model = "gp2_DDandPeterTexts_774M_73Ksteps"
    model_loc = cwd.parent / default_model
    model_loc = str(model_loc.resolve())
    print(f"using model stored here: \n {model_loc} \n")
    corrector = pipeline("text2text-generation", model=gram_model, device=-1)
    print("Finished loading the gramformer model - ", datetime.now())
    iface = gr.Interface(
        chat,
        inputs=["text", "text"],
        outputs="html",
        title="GPT-Peter: 774M Parameter Model",
        description="A basic interface with a 774M parameter model trained on general Q&A and conversation. Treat it like a friend!",
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
        allow_flagging=True,
        flagging_dir="gradio_data",
        flagging_options=["great response", "doesn't make sense", "bad/offensive response"],
        enable_queue=True,  # allows for dealing with multiple users simultaneously
        theme="darkhuggingface",
    )
    iface.launch(share=True)
