"""
NOTE- WIP

conv_w_ai.py

Instead of calling the model in one instance, call it in a loop.
"""
from aitextgen import aitextgen
import argparse
import pprint as pp
import time
import warnings
from pathlib import Path
from utils import get_timestamp
from ai_single_response import extract_response, get_bot_response
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
    ]  # these models used person alpha and person beta in training
    if any(substring in str(mpath_base).lower() for substring in mod_ids):
        speaker = "person alpha" if speaker is None else speaker
        responder = "person beta" if responder is None else responder
    else:
        if verbose:
            print("speaker and responder not set - using default")
        speaker = "person" if speaker is None else speaker
        responder = "person" if responder is None else responder
    p_list = []  # track conversation

    ai = aitextgen(
        model_folder=folder_path,
        to_gpu=use_gpu,
    )
    prompt_msg = start_msg if start_msg is not None else None

    # start conversation
    print(
        f"Entering chat room with GPT Model {mpath_base}. CTRL+C to exit, or type 'exit' to end conversation"
    )

    while True:

        if prompt_msg is not None:
            # inherit prompt from argparse
            pp.pprint(f'You started off with: {prompt_msg}')
        else:
            prompt_msg = str(
                input("enter a message to start/continue the chat: "))
        if prompt_msg.lower().strip() == "exit":
            print(f"exiting conversation loop based on {prompt_msg} input - {get_timestamp()}")
            break
        p_list.append(speaker.lower() + ":" + "\n")
        p_list.append(prompt_msg.lower() + "\n")
        p_list.append("\n")
        p_list.append(responder.lower() + ":" + "\n")
        this_prompt = "".join(p_list)
        # TODO: add safeguard vs. max input length / token length
        pr_len = len(this_prompt)

        # query loaded model
        if verbose:
            print("overall prompt:\n")
            pp.pprint(this_prompt, indent=4)
        print("\n... generating response...")


        chat_resp = ai.generate(
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

        # process the full result to get the ~bot response~ piece
        chat_resp = str(chat_resp[0]).split(
            "\n"
        )  # TODO: adjust hardcoded value for index to dynamic (if n>1)

        if verbose:
            print("chat response:\n")
            pp.pprint(chat_resp, indent=4)
        resp = chat_resp.copy()
        list_p = p_list.copy()
        # isolate the responses from the prompts
        diff_list = extract_response(resp, list_p, verbose=verbose)
        # extract the bot response from the model generated text
        bot_dialogue = get_bot_response(
            name_resp=responder, model_resp=diff_list, name_spk=speaker, verbose=verbose)
        bot_resp = ", ".join(bot_dialogue)
        pp.pprint(bot_resp, indent=4)
        p_list.append(bot_resp + "\n")
        p_list.append("\n")

        prompt_msg = None

    return p_list # note that here it is exported as a list of strings


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
    print("\nA transcript of the conversation:")
    pp.pprint(my_conv, indent=4)

    rt = round(time.perf_counter() - st, 1)

    if want_rt:
        print("The chat took a total of {runtime} seconds. \n".format(runtime=rt))
