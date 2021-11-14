"""

downloads the model files needed to run things in this repo. If things change just update the folder name and the links
All model files (that have been generated by Peter) are in this dropbox folder:

https://www.dropbox.com/sh/e2hbxkzu1e4vtte/AACdUHz-J735F5Cn-KV4udlka?dl=0
"""

import argparse
import os
import utils
from pathlib import Path

model_links = {
    "gpt335M_275ks_checkpoint": "https://www.dropbox.com/sh/7kyoo9462lykfhp/AACbtz0FpwEvD24J04n53LGca?dl=1",
    "gpt335M_325ks_checkpoint": "https://www.dropbox.com/sh/5qhujccnpr9b8ba/AABTU9V3N87iYy7qwWEDVfnsa?dl=1",
    "gpt-neo-125M_150ks_checkpoint": "https://www.dropbox.com/sh/e2hbxkzu1e4vtte/AACdUHz-J735F5Cn-KV4udlka?dl=1",
    "gpt2_std_gpu_774M_120ksteps": "https://www.dropbox.com/sh/f8pocv18n0bohng/AACVMXcWR9Kn_CQsZKqpF1xoa?dl=1",
    "gpt2_dailydialogue_355M_75Ksteps": "https://www.dropbox.com/sh/ahx3teywshods41/AACrGhc_Qntw6GuX7ww-3pbBa?dl=1",
    "GPT2_dailydialogue_355M_150Ksteps": "https://www.dropbox.com/sh/nzcgavha8i11mvw/AACZXMoJuSfI3d3vGRrT_cp5a?dl=1",
    "GPT2_trivNatQAdailydia_774M_175Ksteps": "https://www.dropbox.com/sh/vs848vw311l04ah/AAAuQCyuTEfjaLKo7ipybEZRa?dl=1",
    "gp2_DDandPeterTexts_774M_73Ksteps": "https://www.dropbox.com/sh/bnrwpqqh34s2bea/AAAfuPTJ0A5FgHeOJ0cMlUFha?dl=1",
}

# Set up the parsing of command-line arguments
def get_parser():
    parser = argparse.ArgumentParser(
        description="downloads model files if not found in local working directory"
    )
    parser.add_argument(
        "--download-all",
        default=False,
        action="store_true",
        help="pass this argument if you want all the model files instead of just the 'primary' ones",
    )
    
    return parser

if __name__ == "__main__":
    
    args = get_parser().parse_args()
    get_all = args.download_all
    cwd = Path.cwd()
    my_cwd = str(cwd.resolve()) #string so it can be passed to os.path() objects 
    
    folder_names = [dir for dir in os.listdir(my_cwd) if os.path.isdir(dir)]
    if get_all:
        # download model files not as useful (skipped by default)
        if "gpt2_325k_checkpoint" not in folder_names:
            # standard GPT-2 trained in a mediocre way up to 325,000 steps on my whats app data
            utils.get_zip_URL(
                model_links["gpt335M_325ks_checkpoint"], extract_loc=my_cwd
            )
        if "gpt-neo_datasetv2" not in folder_names:
            # the GPT-Neo small model by EleutherAI trained on my whatsapp data (this was the first one I made.. might suck
            utils.get_zip_URL(
                model_links["gpt-neo-125M_150ks_checkpoint"], extract_loc=my_cwd
            )

        if "GPT2_dailydialogue_355M_150Ksteps" not in folder_names:
            # "DailyDialogues 355M parameter model - to be trained further with custom data or used directly
            utils.get_zip_URL(
                model_links["GPT2_dailydialogue_355M_150Ksteps"], extract_loc=my_cwd
            )


    if "GPT2_trivNatQAdailydia_774M_175Ksteps" not in folder_names:
        # base "advanced" 774M param GPT-2 model trained on: Trivia, Natural Questions, Dialy Dialogues
        utils.get_zip_URL(
            model_links["GPT2_trivNatQAdailydia_774M_175Ksteps"],
            extract_loc=my_cwd,
        )

    if "gp2_DDandPeterTexts_774M_73Ksteps" not in folder_names:
        # GPT-Peter: trained on 73,000 steps of Peter's messages in addition to same items as GPT2_trivNatQAdailydia_774M_175Ksteps
        utils.get_zip_URL(
            model_links["gp2_DDandPeterTexts_774M_73Ksteps"], extract_loc=my_cwd
        )
