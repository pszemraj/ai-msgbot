import os
import sys
from os.path import dirname, join

sys.path.append(dirname(dirname(os.path.abspath(__file__))))

import gradio as gr
import logging
import time
import warnings

from cleantext import clean
from transformers import pipeline
from datetime import datetime
from ai_single_response import query_gpt_peter

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")

logging.basicConfig()
gpt_peter_model = "gp2_DDandPeterTexts_gpu_774M_175Ksteps"
gram_model = "prithivida/grammar_error_correcter_v1"
model_loc = join(dirname(os.getcwd()), gpt_peter_model)


def gramformer_correct(corrector, qphrase: str):
    try:
        corrected = corrector(
            clean(qphrase), return_text=True, clean_up_tokenization_spaces=True
        )
        return corrected[0]["generated_text"]
    except:
        print("NOTE - failed to correct with gramformer")
        return clean(qphrase)


def ask_gpt(message: str, sender: str = ""):
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

    resp = query_gpt_peter(
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
    corrector = pipeline("text2text-generation", model=gram_model, device=-1)
    print("Finished loading the gramformer model - ", datetime.now())
    iface = gr.Interface(
        chat,
        inputs=["text", "text"],
        outputs="html",
        title="GPT-Peter: 774M Parameter Model",
        description="A basic interface with a 774M parameter model of the best and most "
        "humble human to grace the earth. You can view / screenshot your chat history on the right, and feel free to "
        "'flag' anything either amusing or nonsensical",
        article="**Important Notes & About:**\n"
        "1. the model can take up to 60 seconds to respond sometimes, patience is a virtue.\n"
        "2. entering your name is completely optional, but might get you a more personalized response if you "
        "have messaged me in the past.\n"
        "3. the model started from a pretrained checkpoint, **and in addition was trained on other datasets** "
        "before Peter's messages says should not be interpreted as a past message or an absolutely true "
        "statement.\n "
        "_You can learn more about the model architecture and training process [here]("
        "https://youtu.be/dQw4w9WgXcQ)._",
        css="""
        .chatbox {display:flex;flex-direction:column}
        .user_msg, .resp_msg {padding:4px;margin-bottom:4px;border-radius:4px;width:80%}
        .user_msg {background-color:cornflowerblue;color:white;align-self:start}
        .resp_msg {background-color:lightgray;align-self:self-end}
    """,
        allow_screenshot=True,
        allow_flagging=True,
        flagging_dir="gradio_data",
        flagging_options=["amusing", "I actually laughed", "bad/useless response"],
        enable_queue=True,
        theme="darkhuggingface",
    )
    iface.launch(share=True)
