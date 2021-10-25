"""

An executable way to call the model. example:
*\gpt2_chatbot> python .\ai_single_response.py --prompt "do you love me?" --speaker "luz"

"""
import argparse
import gc
import os
import pprint as pp
import time
import warnings
from datetime import datetime
from os.path import join

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")

from aitextgen import aitextgen


def query_gpt_peter(
    folder_path,
    prompt_msg: str,
    speaker=None,
    responder="peter szemraj",
    kparam=125,
    temp=0.75,
    top_p=0.65,
    verbose=False,
    use_gpu=False,
):
    ai = aitextgen(
        model_folder=folder_path,
        to_gpu=use_gpu,
    )
    p_list = []
    if speaker is not None:
        p_list.append(speaker.lower() + ":" + "\n")  # write prompt as the speaker
    p_list.append(prompt_msg.lower() + "\n")
    p_list.append("\n")
    p_list.append(responder.lower() + ":" + "\n")
    this_prompt = "".join(p_list)
    if verbose:
        print("overall prompt:\n")
        pp.pprint(this_prompt, indent=4)
    print("\n... generating... \n")
    this_result = ai.generate(
        n=1,
        top_k=kparam,
        batch_size=512,
        max_length=128,
        min_length=16,
        prompt=this_prompt,
        temperature=temp,
        top_p=top_p,
        do_sample=True,
        return_as_list=True,
    )

    try:
        this_result = str(this_result[0]).split("\n")
        res_out = [str(ele).strip() for ele in this_result]
        p_out = [str(ele).strip() for ele in p_list]
        diff_list = list(
            set(res_out).difference(p_out)
        )  # remove prior prompts for the response
        this_result = [
            str(msg)
            for msg in diff_list
            if (":" not in str(msg))
            and ("szemr" not in str(msg))
            and ("peter" not in str(msg))
        ]  # remove all names
        if not isinstance(this_result, list):
            list(this_result)
        output = str(this_result[0]).strip()
        # add second line of output if first is too short (subjective)
        if len(output) < 15 and len(this_result) > 1:
            output = output + " " + str(this_result[1]).strip()
    except:
        output = "bro, there was an error. try again"

    p_list.append(output + "\n")
    p_list.append("\n")

    model_responses = {"out_text": output, "full_conv": p_list}
    print("finished!\n")

    return model_responses


# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(
    description="submit a message and have a 335M parameter GPT model respond"
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
    default="gp2_DDandPeterTexts_41kPeter-774M",
    help="folder - with respect to git directory of your repo that has the model files in it (pytorch.bin + "
    "config.json)",
)

parser.add_argument(
    "--speaker",
    required=False,
    default=None,
    help="who the prompt is from (to the bot). Note this does not help if you do not text me often",
)
parser.add_argument(
    "--responder",
    required=False,
    default="peter szemraj",
    help="who the responder is. default = peter szemraj",
)

parser.add_argument(
    "--topk",
    required=False,
    type=int,
    default=125,
    help="how many responses to sample. lower = more random responses",
)

parser.add_argument(
    "--temp",
    required=False,
    type=float,
    default=0.75,
    help="roughly considered as 'model creativity'",
)

parser.add_argument(
    "--topp",
    required=False,
    type=float,
    default=0.65,
    help="nucleus sampling frac - aka: what fraction of possible options are considered?",
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

if __name__ == "__main__":
    args = parser.parse_args()
    query = args.prompt
    model_dir = args.model
    model_loc = join(os.getcwd(), model_dir)
    spkr = args.speaker
    rspndr = args.responder
    k_results = args.topk
    my_temp = args.temp
    my_top_p = args.topp
    want_verbose = args.verbose
    want_rt = args.time

    # force-update the speaker+responder params for the generic model case
    if "dailydialogue" in model_dir.lower():
        spkr = "john smith"
        rspndr = "nancy sellers"
        # ^ fake people I made up when parsing Daily Dialogue dataset    # force-update the speaker+responder params for the generic model case
    if "trivnatqa" in model_dir.lower():
        spkr = "person alpha"
        rspndr = "person beta"
        # ^ fake people I made up when parsing Daily Dialogue dataset

    st = time.time()

    resp = query_gpt_peter(
        folder_path=model_loc,
        prompt_msg=query,
        speaker=spkr,
        responder=rspndr,
        kparam=k_results,
        temp=my_temp,
        top_p=my_top_p,
        verbose=False,
        use_gpu=False,
    )

    output = resp["out_text"]
    pp.pprint(output, indent=4)

    # pp.pprint(this_result[3].strip(), indent=4)
    rt = round(time.time() - st, 1)
    gc.collect()

    if want_rt:
        print("took {runtime} seconds to generate. \n".format(runtime=rt))

    if want_verbose:
        print("finished - ", datetime.now())
    if want_verbose:
        p_list = resp["full_conv"]
        print("A transcript of your chat is as follows: \n")
        p_list = [item.strip() for item in p_list]
        pp.pprint(p_list)
