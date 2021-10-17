import argparse
import gc
import os
import pprint as pp
import time
from datetime import datetime
from os.path import join
import warnings

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")

from aitextgen import aitextgen

folder_path = join(os.getcwd(), "gpt2_275k_checkpoint")
verbose = False
# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(
    description="submit a message and have the 335M parameter peter szemraj model respond"
)
parser.add_argument(
    "--prompt", required=True, help="the message the bot is supposed to respond to"
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
    prompt_msg = args.prompt
    speaker = args.speaker
    responder = args.responder
    verbose = args.verbose
    want_rt = args.time

    ai = aitextgen(
        model_folder=folder_path,
        to_gpu=False,
    )

    p_list = []
    st = time.time()
    if speaker is not None:
        p_list.append(speaker.lower() + ":" + "\n")  # write prompt as the speaker
    p_list.append(prompt_msg.lower() + "\n")
    p_list.append("\n")
    p_list.append(responder.lower() + "\n")
    this_prompt = "".join(p_list)
    if verbose:
        print("overall prompt:\n")
        pp.pprint(this_prompt, indent=4)
    print("\n... generating... \n")
    this_result = ai.generate(
        n=1,
        top_k=50,
        batch_size=512,
        max_length=128,
        min_length=16,
        prompt=this_prompt,
        temperature=0.7,
        top_p=0.9,
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
            output = output + str(this_result[1]).strip()
    except:
        output = "bro, there was an error. try again"

    pp.pprint(output, indent=4)
    p_list.append(output + "\n")
    p_list.append("\n")
    # pp.pprint(this_result[3].strip(), indent=4)
    rt = round(time.time() - st, 1)
    gc.collect()

    if want_rt:
        print("took {runtime} seconds to generate. \n".format(runtime=rt))

    if verbose:
        print("finished - ", datetime.now())
    if verbose:
        print("A transcript of your chat is as follows: \n")
        p_list = [item.strip() for item in p_list]
        pp.pprint(p_list)
