"""
WIP

convert ai_single_response to now respond to the whole conversation

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


def query_gpt_model(
    folder_path,
    prompt_msg: str,
    speaker=None,
    responder="person beta",
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
        responder (str, optional): [description]. Defaults to "person beta".
        kparam (int, optional): [description]. Defaults to 125.
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
    p_list = []
    if "natqa" in str(folder_path).lower():
        speaker = "person alpha"  # manual correction
        responder = "person beta"
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
        use_cache=True,
    )
    if verbose:
        pp.pprint(this_result)  # to see what is going on
    try:
        this_result = str(this_result[0]).split("\n")
        res_out = [clean(ele) for ele in this_result]
        p_out = [clean(ele) for ele in p_list]
        if verbose:
            pp.pprint(res_out)  # to see what is going on
            pp.pprint(p_out)  # to see what is going on

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
            pp.pprint(diff_list)  # to see what is going on
            print("---------------------------------")

        output = ", ".join(diff_list)

    except:
        output = "oops, there was an error. try again"

    p_list.append(output + "\n")
    p_list.append("\n")

    model_responses = {"out_text": output, "full_conv": p_list}
    print("finished!\n")

    return model_responses


# Set up the parsing of command-line arguments
def get_parser():
    """
    get_parser [a helper function for the argparse module]

    Returns:
        [argparse.ArgumentParser]: [the argparser relevant for this script]
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
        # "gp2_DDandPeterTexts_774M_73Ksteps", - from GPT-Peter
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

    # force-update the speaker+responder params for the generic model case
    if "dailydialogue" in model_dir.lower():
        spkr = "john smith"
        rspndr = "nancy sellers"
        # ^ arbitrary people created when parsing Daily Dialogue dataset
        # # force-update the speaker+responder params
        # for the generic model case
    if "natqa" in model_dir.lower():
        spkr = "person alpha"
        rspndr = "person beta"
        # ^ arbitrary people created when parsing NatQA + TriviaQA + Daily Dialogue datasets

    st = time.time()

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

    # pp.pprint(this_result[3].strip(), indent=4)
    rt = round(time.time() - st, 1)

    if want_rt:
        print("took {runtime} seconds to generate. \n".format(runtime=rt))

    if want_verbose:
        print("finished - ", datetime.now())
    if want_verbose:
        p_list = resp["full_conv"]
        print("A transcript of your chat is as follows: \n")
        p_list = [item.strip() for item in p_list]
        pp.pprint(p_list)
