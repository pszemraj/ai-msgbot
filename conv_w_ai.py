#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conv_w_ai.py - a script to run a conversation with a GPT-2 model

Instead of taking a prompt, pass it in, get a response, and return that response + end, the querying of the model is done in a while loop. Similar to how results are handled in the ai_single_respose.py script, the results are returned as a list of strings. Instead of these strings being discarded, they are appended to a list, and the next prompt/response pair is appended to the list, etc. This then allows the transformer model to "see" the conversational context. from earlier in the conversation.

"""

import argparse
import pprint as pp
import time
import warnings
from pathlib import Path

from aitextgen import aitextgen

from ai_single_response import query_gpt_model
from utils import get_timestamp, shorten_list

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")


def converse_w_ai(
    folder_path: str,
    start_msg: str,
    speaker: str = None,
    responder: str = None,
    resp_length: int = 48,
    max_context_length: int = 512,
    kparam: int = 40,
    temp: float = 0.7,
    top_p: float = 0.9,
    verbose: bool = False,
    use_gpu: bool = False,
):
    """
    converse_w_ai - a helper function for the aitextgen module calling query_gpt_model

    Args:
        folder_path (str): the path to the folder containing the model files
        start_msg (str): the message the bot is supposed to respond to. Prompt is said by speaker, answered by responder.
        speaker (str, optional): Who the prompt is from (to the bot). Primarily relevant to bots trained on multi-individual chat data. Defaults to None.
        responder (str, optional): who the responder is. Primarily relevant to bots trained on multi-individual chat data. Defaults to "person beta".
        resp_length (int, optional): the length of the response in tokens. Defaults to 48.
        max_context_length (int, optional): the maximum length of the context _in characters_. Defaults to 512.
        kparam (int, optional): the k parameter for the top_k. Defaults to 40.
        temp (float, optional): the temperature for the softmax. Defaults to 0.7.
        top_p (float, optional): the top_p parameter for nucleus sampling. Defaults to 0.9.
        verbose (bool, optional): Defaults to False.
        use_gpu (bool, optional): Defaults to False.

    Returns:
        [list]: [a list of strings, each string is a response]
    """

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
        # safeguard against max_input_length (ai-textgen does not support this)
        current_history = list(conversation.values())
        conversation_history = shorten_list(current_history, max_length=max_context_length)
        model_outputs = query_gpt_model(
            folder_path=folder_path,
            prompt_msg=prompt_msg,
            conversation_history=conversation_history if len(conversation_history) > 0 else None,
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
        bot_resp = model_outputs["out_text"]
        conversation = model_outputs["full_conv"]
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
        "-p",
        "--prompt",
        required=False,
        default=None,
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
        "config.json). No models? Run the script download_models.py",
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
        "--use_gpu",
        required=False,
        action="store_true",
        help="use gpu if available",
    )

    parser.add_argument(
        "--max-context-length",
        required=False,
        type=int,
        default=512,
        help="the maximum length of the context _in characters_. Defaults to 512",
    )
    parser.add_argument(
        "--resp-length",
        required=False,
        type=int,
        default=48,
        help="the length of the response in tokens. Defaults to 48",
    )

    parser.add_argument(
        "-rt",
        "--time",
        default=False,
        action="store_true",
        help="pass this argument if you want to know runtime",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="pass this argument if you want all the printouts",
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
    use_gpu = args.use_gpu
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
        use_gpu=use_gpu,
    )

    # print the runtime / transcript
    print("\nA transcript of the conversation:")
    pp.pprint(my_conv, indent=4)

    rt = round(time.perf_counter() - st, 1)

    if want_rt:
        print("The chat took a total of {runtime} seconds. \n".format(runtime=rt))
