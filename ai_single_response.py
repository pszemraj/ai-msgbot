#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_single_response.py

An executable way to call the model. example:
*\gpt2_chatbot> python .\ai_single_response.py --prompt "where is the grocery store?" --time

this will return a response to the prompt.

"""
import argparse
import pprint as pp
import time
import warnings
from datetime import datetime
from pathlib import Path
from cleantext import clean

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")

from aitextgen import aitextgen

def extract_response(full_resp:list, plist:list, verbose:bool=False):
    """
    extract_response - helper fn for ai_single_response.py. By default aitextgen returns the prompt and the response, we just want the response
    
    Args:
        full_resp (list): a list of strings, each string is a response
        plist (list): a list of strings, each string is a prompt
        
        verbose (bool, optional): 4 debug. Defaults to False.
    """
    
    plist = [ele for ele in plist]

    iso_resp = []
    for line in full_resp:
        iso_resp.append(line) if line not in plist else None
        # if line in plist:
        #     continue
        # else:
        #     iso_resp.append(line)
    if verbose:
        print("the isolated responses are:\n")
        print("\n".join(iso_resp))
        print("the input prompt was:\n")
        print("\n".join(plist))
    return iso_resp # list of only the model gnerated responses

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
        responder = "person" if responder is None else responder

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
        pp.pprint(this_result)  # for debugging
    try:

        this_result = str(this_result[0]).split(
            "\n"
        )  # TODO: adjust hardcoded value for index to dynamic (if n>1)
        res_out = [clean(ele) for ele in this_result]
        p_out = [clean(ele) for ele in p_list]
        if verbose:

            pp.pprint(res_out)  # for debugging
            print("the original prompt:\n")
            pp.pprint(p_out)  # for debugging

        diff_list = []
        name_counter = 0
        break_safe = False
        for resline in res_out:

            if (responder + ":") in resline:
                name_counter += 1
                break_safe = True  # next line a response from bot
                continue
            if ":" in resline and name_counter > 0:
                if break_safe:
                    diff_list.append(resline)
                    break_safe = False
                else:
                    break
            if resline in p_out:
                break_safe = False
                continue

            else:
                diff_list.append(resline)
                break_safe = False

        if verbose:
            print("------------------------diff list: ")
            pp.pprint(
                diff_list
            )  # where diff_list is only the text generated by the model
            print("---------------------------------")

        output = ", ".join(diff_list)

    except Exception:
        output = "oops, there was an error. try again"

    p_list.append(output + "\n")
    p_list.append("\n")

    print("\nfinished!")

    return {"out_text": output, "full_conv": p_list}  # model responses


# Set up the parsing of command-line arguments
def get_parser():
    """
    get_parser [a helper function for the argparse module]

    Returns: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="submit a message and have a 774M parameter GPT model respond"
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
