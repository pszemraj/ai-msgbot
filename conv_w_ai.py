#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conv_w_ai.py - a script to run a conversation with a GPT-2 model

Insteading of taking a prompt, pass it in, get a response, and return that response + end, the querying of the model is done in a while loop. Similar to how results are handled in the ai_single_respose.py script, the results are returned as a list of strings. Instead of these strings being discarded, they are appended to a list, and the next prompt/response pair is appended to the list, etc. This then allows the transformer model to "see" the conversational context. from earlier in the conversation.

"""

import argparse
import pprint as pp
import time
import warnings
from pathlib import Path

from aitextgen import aitextgen

from ai_single_response import extract_response, get_bot_response, query_gpt_model
from utils import get_timestamp, remove_trailing_punctuation

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")


def converse_w_ai(
    folder_path,
    start_msg: str,
    speaker=None,
    responder=None,
    resp_length=128,
    kparam=150,
    temp=0.75,
    top_p=0.65,
    verbose=False,
    use_gpu=False,
):
    # initialise pre while-loop variables

    if verbose:
        print(f"initializing conversation... {get_timestamp()}")
    start_msg = (
        str(input("enter a message to start the conversation: "))
        if start_msg is None
        else start_msg
    )
    mpath = Path(folder_path)
    mpath_base = (
        mpath.stem
    )  # only want the base name of the model folder for check below
    mod_ids = [
        "natqa",
        "dd",
        "trivqa",
        "wow",
        "conversational",
    ]  # these models used person alpha and person beta in training
    if any(substring in str(mpath_base).lower() for substring in mod_ids):
        speaker = "person alpha" if speaker is None else speaker
        responder = "person beta" if responder is None else responder
    else:
        if verbose:
            print("speaker and responder not set - using default")
        speaker = "person" if speaker is None else speaker
        responder = "person" if responder is None else responder

    ai = aitextgen(
        model_folder=folder_path,
        to_gpu=use_gpu,
    )
    prompt_msg = start_msg if start_msg is not None else None
    conversation = {}
    # start conversation
    print(
        f"Entering chat room with GPT Model {mpath_base}. CTRL+C to exit, or type 'exit' to end conversation"
    )

    while True:

        if prompt_msg is not None:
            # inherit prompt from argparse
            pp.pprint(f"You started off with: {prompt_msg}")
        else:
            prompt_msg = str(input("enter a message to start/continue the chat: "))
        if prompt_msg.lower().strip() == "exit":
            print(
                f"exiting conversation loop based on {prompt_msg} input - {get_timestamp()}"
            )
            break
        # # TODO: add safeguard vs. max input length / token length for specific models

        model_outputs = query_gpt_model(
            folder_path=folder_path,
            prompt_msg=prompt_msg,
            conversation_history=list(conversation.values()) if len(conversation) > 0 else None,
            speaker=speaker,
            responder=responder,
            resp_length=resp_length,
            kparam=kparam,
            temp=temp,
            top_p=top_p,
            aitextgen_obj=ai,
            verbose=verbose,
            use_gpu=use_gpu,
        )
        bot_resp = model_outputs['out_text']
        conversation = model_outputs['full_conv']
        pp.pprint(bot_resp, indent=4)

        prompt_msg = None

    return list(conversation.values())

# Set up the parsing of command-line arguments
def get_parser():
    """
    get_parser [a helper function for the argparse module]

    Returns:
        [argparse.ArgumentParser]: [the argparser relevant for this script]
    """

    parser = argparse.ArgumentParser(
        description="submit a message and have a custom fine-tuned GPT model respond"
    )
    parser.add_argument(
        "--prompt",
        required=False,
        default=None,
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

    # run the chat
    my_conv = converse_w_ai(
        folder_path=model_loc,
        start_msg=query,
        speaker=spkr,
        responder=rspndr,
        kparam=k_results,
        temp=my_temp,
        top_p=my_top_p,
        verbose=want_verbose,
        use_gpu=False,
    )

    # print the runtime / transcript
    print("\nA transcript of the conversation:")
    pp.pprint(my_conv, indent=4)

    rt = round(time.perf_counter() - st, 1)

    if want_rt:
        print("The chat took a total of {runtime} seconds. \n".format(runtime=rt))
