#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_single_response.py

An executable way to call the model. example:
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
from cleantext import clean
from utils import print_spacer, remove_trailing_punctuation

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")

from aitextgen import aitextgen


def extract_response(full_resp: list, plist: list, verbose: bool = False):
    """
    extract_response - helper fn for ai_single_response.py. By default aitextgen returns the prompt and the response, we just want the response

    Args:
        full_resp (list): a list of strings, each string is a response
        plist (list): a list of strings, each string is a prompt

        verbose (bool, optional): 4 debug. Defaults to False.
    """
    full_resp = [clean(ele) for ele in full_resp]
    plist = [clean(pr) for pr in plist]
    p_len = len(plist)
    assert (
        len(full_resp) >= p_len
    ), "model output should have as many lines or longer as the input."

    if set(plist).issubset(full_resp):

        del full_resp[:p_len]  # remove the prompts from the responses
    else:
        print("the isolated responses are:\n")
        pp.pprint(full_resp)
        print_spacer()
        print("the input prompt was:\n")
        pp.pprint(plist)
        print_spacer()
        sys.exit("Exiting: some prompts not found in the responses")
    if verbose:
        print("the isolated responses are:\n")
        pp.pprint(full_resp)
        print_spacer()
        print("the input prompt was:\n")
        pp.pprint(plist)
        print_spacer()
    return full_resp  # list of only the model generated responses


def get_bot_response(
    name_resp: str, model_resp: str, name_spk: str, verbose: bool = False
):

    """

    get_bot_response  - from the model response, extract the bot response. This is needed because depending on the generation length the model may return more than one response.

    Args:   name_resp (str): the name of the responder
    model_resp (str): the model response
    verbose (bool, optional): 4 debug. Defaults to False.

    returns: fn_resp (list of str)
    """

    fn_resp = []

    name_counter = 0
    break_safe = False
    for resline in model_resp:
        if resline.startswith(name_resp):
            name_counter += 1
            break_safe = True  # know the line is from bot as this line starts with the name of the bot
            continue
        if name_spk is not None and name_spk.lower() in resline.lower():
            # TODO: fix this
            break
        if ":" in resline and name_counter > 0:
            if break_safe:
                # we know this is a response from the bot even tho ':' is in the line
                fn_resp.append(resline)
                break_safe = False
            else:
                # we do not know this is a response from the bot. could be name of another person.. bot is "finished" response
                break
        else:
            fn_resp.append(resline)
            break_safe = False
    if verbose:
        print("the full response is:\n")
        print("\n".join(fn_resp))

    return fn_resp


def query_gpt_model(
    folder_path,
    prompt_msg: str,
    speaker=None,
    responder=None,
    resp_length=128,
    kparam=150,
    temp=0.75,
    top_p=0.65,
    verbose=False,
    use_gpu=False,
):
    """
    query_gpt_model [pass a prompt in to model, get a response. Does NOT "remember" past conversation]

    Args:
        folder_path ([type]): [description]
        prompt_msg (str): [description]
        speaker ([type], optional): [description]. Defaults to None.
        responder (str, optional): [description]. Defaults to None.
        resp_length (int, optional): [description]. Defaults to 128.
        kparam (int, optional): [description]. Defaults to 50.
        temp (float, optional): [description]. Defaults to 0.75.
        top_p (float, optional): [description]. Defaults to 0.65.
        verbose (bool, optional): [description]. Defaults to False.
        use_gpu (bool, optional): [description]. Defaults to False.

    Returns:
        [dict]: [returns a dict with A) just model response as str B) total conversation]
    """
    ai = aitextgen(
        model_folder=folder_path,
        to_gpu=use_gpu,
    )

    mpath = Path(folder_path)
    mpath_base = (
        mpath.stem
    )  # only want the base name of the model folder for check below
    # these models used person alpha and person beta in training
    mod_ids = ["natqa", "dd", "trivqa", "wow"]
    if any(substring in str(mpath_base).lower() for substring in mod_ids):
        speaker = "person alpha" if speaker is None else speaker
        responder = "person beta" if responder is None else responder
    else:
        if verbose:
            print("speaker and responder not set - using default")
        speaker = "person" if speaker is None else speaker
        responder = "george robot" if responder is None else responder

    p_list = []  # track conversation
    p_list.append(speaker.lower() + ":" + "\n")
    p_list.append(prompt_msg.lower() + "\n")
    p_list.append("\n")
    p_list.append(responder.lower() + ":" + "\n")
    this_prompt = "".join(p_list)
    pr_len = len(this_prompt)
    if verbose:
        print("overall prompt:\n")
        pp.pprint(this_prompt, indent=4)
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
    this_result = str(this_result[0]).split(
        "\n"
    )  # TODO: adjust hardcoded value for index to dynamic (if n>1)
    og_res = this_result.copy()
    og_prompt = p_list.copy()
    diff_list = extract_response(
        this_result, p_list, verbose=verbose
    )  # isolate the responses from the prompts
    # extract the bot response from the model generated text
    bot_dialogue = get_bot_response(
        name_resp=responder, model_resp=diff_list, name_spk=speaker, verbose=verbose
    )
    bot_resp = ", ".join(bot_dialogue)
    bot_resp = remove_trailing_punctuation(bot_resp.strip()) # remove trailing punctuation to seem more natural
    # remove the last ',' '.' chars
    bot_resp = bot_resp[:-1] if bot_resp.endswith(".") else bot_resp
    bot_resp = bot_resp[:-1] if bot_resp.endswith(",") else bot_resp
    if verbose:
        print("\n... bot response:\n")
        pp.pprint(bot_resp)
    og_prompt.append(bot_resp + "\n")
    og_prompt.append("\n")

    print("\nfinished!")
    # return the bot response and the full conversation

    return {"out_text": bot_resp, "full_conv": og_prompt}  # model responses


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
        "--prompt",
        required=True,  # MUST HAVE A PROMPT
        type=str,
        help="the message the bot is supposed to respond to. Prompt is said by speaker, answered by responder.",
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
        "--speaker",
        required=False,
        default=None,
        help="Who the prompt is from (to the bot). Primarily relevant to bots trained on multi-individual chat data",
    )
    parser.add_argument(
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
        "--verbose",
        default=False,
        action="store_true",
        help="pass this argument if you want all the printouts",
    )
    parser.add_argument(
        "--time",
        default=False,
        action="store_true",
        help="pass this argument if you want to know runtime",
    )
    return parser


if __name__ == "__main__":
    # parse the command line arguments
    args = get_parser().parse_args()
    query = args.prompt
    model_dir = str(args.model)
    model_loc = Path.cwd() / model_dir
    spkr = args.speaker
    rspndr = args.responder
    k_results = args.topk
    my_temp = args.temp
    my_top_p = args.topp
    want_verbose = args.verbose
    want_rt = args.time

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
        use_gpu=False,
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
