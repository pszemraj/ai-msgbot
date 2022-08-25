#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_single_response.py - a script to generate a response to a prompt from a pretrained GPT model

example:
*\gpt2_chatbot> python ai_single_response.py --model "GPT2_conversational_355M_WoW10k" --prompt "hey, what's up?" --time

query_gpt_model is used throughout the code, and is the "fundamental" building block of the bot and how everything works. I would recommend testing this function with a few different models.

"""
import argparse
import pprint as pp
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
import logging

from cleantext import clean

from utils import clear_loggers, print_spacer, remove_trailing_punctuation

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")
warnings.filterwarnings(action="ignore", message=".*the GPL-licensed package `unidecode` is not installed*")

logging.basicConfig(
    filename=f"LOGFILE-{Path(__file__).stem}.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

from aitextgen import aitextgen


def extract_response(full_resp: list, plist: list, verbose: bool = False):
    """
    extract_response - helper fn for ai_single_response.py. By default aitextgen returns the prompt and the response, we just want the response

    Args:
        full_resp (list): the full response from aitextgen
        plist (list): the prompt list
        verbose (bool, optional): Defaults to False.

    Returns:
        response (str): the response, without the prompt
    """
    bot_response = []
    for line in full_resp:
        if line.lower() in plist and len(bot_response) < len(plist):
            first_loc = plist.index(line)
            del plist[first_loc]
            continue
        bot_response.append(line)
    full_resp = [clean(ele, lower=False) for ele in bot_response]

    if verbose:
        print("the isolated responses are:\n")
        pp.pprint(full_resp)
        print_spacer()
        print("the input prompt was:\n")
        pp.pprint(plist)
        print_spacer()
    return full_resp  # list of only the model generated responses


def get_bot_response(
    name_resp: str, model_resp: list, name_spk: str, verbose: bool = False
):
    """
    get_bot_response - gets the bot response to a prompt, checking to ensure that additional statements by the "speaker" are not included in the response.

    Args:
        name_resp (str): the name of the responder
        model_resp (list): the model response
        name_spk (str): the name of the speaker
        verbose (bool, optional): Defaults to False.

    Returns:
        bot_response (str): the bot response, isolated down to just text without the "name tokens" or further messages from the speaker.
    """

    fn_resp = []

    name_counter = 0
    break_safe = False
    for resline in model_resp:
        if name_resp.lower() in resline.lower():
            name_counter += 1
            break_safe = True
            continue
        if ":" in resline and name_resp.lower() not in resline.lower():
            break
        if name_spk.lower() in resline.lower() and not break_safe:
            break
        else:
            fn_resp.append(resline)
    if verbose:
        print("the full response is:\n")
        print("\n".join(fn_resp))

    return fn_resp


def query_gpt_model(
    folder_path: str or Path,
    prompt_msg: str,
    conversation_history: list = None,
    speaker: str = None,
    responder: str = None,
    resp_length: int = 48,
    kparam: int = 40,
    temp: float = 0.7,
    top_p: float = 0.9,
    aitextgen_obj=None,
    verbose: bool = False,
    use_gpu: bool = False,
):
    """
    query_gpt_model - queries the GPT model and returns the first response by <responder>

    Args:
        folder_path (str or Path): the path to the model folder
        prompt_msg (str): the prompt message
        conversation_history (list, optional): the conversation history. Defaults to None.
        speaker (str, optional): the name of the speaker. Defaults to None.
        responder (str, optional): the name of the responder. Defaults to None.
        resp_length (int, optional): the length of the response in tokens. Defaults to 48.
        kparam (int, optional): the k parameter for the top_k. Defaults to 40.
        temp (float, optional): the temperature for the softmax. Defaults to 0.7.
        top_p (float, optional): the top_p parameter for nucleus sampling. Defaults to 0.9.
        aitextgen_obj (_type_, optional): a pre-loaded aitextgen object. Defaults to None.
        verbose (bool, optional): Defaults to False.
        use_gpu (bool, optional): Defaults to False.

    Returns:
        model_resp (dict): the model response, as a dict with the following keys: out_text (str) the generated text and full_conv (dict) the conversation history
    """

    try:
        ai = (
        aitextgen_obj
        if aitextgen_obj
        else aitextgen(
            model_folder=folder_path,
            to_gpu=use_gpu,
        )
    )
    except Exception as e:
        print(f"Unable to initialize aitextgen model: {e}")
        print(f"Check model folder: {folder_path}, run the download_models.py script to download the model files")
        sys.exit(1)

    mpath = Path(folder_path)
    mpath_base = (
        mpath.stem
    )  # only want the base name of the model folder for check below
    # these models used person alpha and person beta in training
    mod_ids = ["natqa", "dd", "trivqa", "wow", "conversational"]
    if any(substring in str(mpath_base).lower() for substring in mod_ids):
        speaker = "person alpha" if speaker is None else speaker
        responder = "person beta" if responder is None else responder
    else:
        if verbose:
            print("speaker and responder not set - using default")
        speaker = "person" if speaker is None else speaker
        responder = "george robot" if responder is None else responder

    prompt_list = (
        conversation_history if conversation_history is not None else []
    )  # track conversation
    prompt_list.append(speaker.lower() + ":" + "\n")
    prompt_list.append(prompt_msg.lower() + "\n")
    prompt_list.append("\n")
    prompt_list.append(responder.lower() + ":" + "\n")
    this_prompt = "".join(prompt_list)
    pr_len = len(this_prompt)
    if verbose:
        print("overall prompt:\n")
        pp.pprint(prompt_list)
    # call the model
    print("\n... generating...")
    this_result = ai.generate(
        n=1,
        top_k=kparam,
        batch_size=128,
        # the prompt input counts for text length constraints
        max_length=resp_length + pr_len,
        min_length=16 + pr_len,
        prompt=this_prompt,
        temperature=temp,
        top_p=top_p,
        do_sample=True,
        return_as_list=True,
        use_cache=True,
    )
    if verbose:
        print("\n... generated:\n")
        pp.pprint(this_result)  # for debugging
    # process the full result to get the ~bot response~ piece
    this_result = str(this_result[0]).split("\n")
    input_prompt = this_prompt.split("\n")

    diff_list = extract_response(
        this_result, input_prompt, verbose=verbose
    )  # isolate the responses from the prompts
    # extract the bot response from the model generated text
    bot_dialogue = get_bot_response(
        name_resp=responder, model_resp=diff_list, name_spk=speaker, verbose=verbose
    )
    bot_resp = ", ".join(bot_dialogue)
    bot_resp = remove_trailing_punctuation(
        bot_resp.strip()
    )  # remove trailing punctuation to seem more natural
    if verbose:
        print("\n... bot response:\n")
        pp.pprint(bot_resp)
    prompt_list.append(bot_resp + "\n")
    prompt_list.append("\n")
    conv_history = {}
    for i, line in enumerate(prompt_list):
        if i not in conv_history.keys():
            conv_history[i] = line
    if verbose:
        print("\n... conversation history:\n")
        pp.pprint(conv_history)
    print("\nfinished!")

    # return the bot response and the full conversation
    return {"out_text": bot_resp, "full_conv": conv_history}


# Set up the parsing of command-line arguments
def get_parser():
    """
    get_parser [a helper function for the argparse module]

    Returns: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="submit a message and have a pretrained GPT model respond"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        required=True,  # MUST HAVE A PROMPT
        type=str,
        help="the message the bot is supposed to respond to. Prompt is said by speaker, answered by responder.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=False,
        type=str,
        default="distilgpt2-tiny-conversational",
        help="folder - with respect to git directory of your repo that has the model files in it (pytorch.bin + "
        "config.json). You can also pass the huggingface model name (e.g. distilgpt2)",
    )

    parser.add_argument(
        "-s",
        "--speaker",
        required=False,
        default=None,
        help="Who the prompt is from (to the bot). Primarily relevant to bots trained on multi-individual chat data",
    )
    parser.add_argument(
        "-r",
        "--responder",
        required=False,
        default="person beta",
        help="who the responder is. Primarily relevant to bots trained on multi-individual chat data",
    )

    parser.add_argument(
        "--topk",
        required=False,
        type=int,
        default=150,
        help="how many responses to sample (positive integer). lower = more random responses",
    )

    parser.add_argument(
        "--temp",
        required=False,
        type=float,
        default=0.75,
        help="specify temperature hyperparam (0-1). roughly considered as 'model creativity'",
    )

    parser.add_argument(
        "--topp",
        required=False,
        type=float,
        default=0.65,
        help="nucleus sampling frac (0-1). aka: what fraction of possible options are considered?",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="pass this argument if you want all the printouts",
    )

    parser.add_argument(
        "-rt",
        "--time",
        default=False,
        action="store_true",
        help="pass this argument if you want to know runtime",
    )

    parser.add_argument(
        "--use_gpu",
        required=False,
        action="store_true",
        help="use gpu if available",
    )

    return parser


if __name__ == "__main__":
    # parse the command line arguments
    args = get_parser().parse_args()
    query = args.prompt
    model_dir = str(args.model)
    model_loc = Path.cwd() / model_dir if "/" not in model_dir else model_dir
    spkr = args.speaker
    rspndr = args.responder
    k_results = args.topk
    my_temp = args.temp
    my_top_p = args.topp
    want_verbose = args.verbose
    want_rt = args.time
    use_gpu = args.use_gpu

    st = time.perf_counter()

    resp = query_gpt_model(
        folder_path=model_loc,
        prompt_msg=query,
        speaker=spkr,
        responder=rspndr,
        kparam=k_results,
        temp=my_temp,
        top_p=my_top_p,
        verbose=want_verbose,
        use_gpu=use_gpu,
    )

    output = resp["out_text"]
    pp.pprint(output, indent=4)

    rt = round(time.perf_counter() - st, 1)

    if want_rt:
        print("took {runtime} seconds to generate. \n".format(runtime=rt))

    if want_verbose:
        print("finished - ", datetime.now())
        p_list = resp["full_conv"]
        print("A transcript of your chat is as follows: \n")
        p_list = [item.strip() for item in p_list]
        pp.pprint(p_list)
