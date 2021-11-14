import os
import sys
from os.path import dirname, join, basename

sys.path.append(dirname(dirname(os.path.abspath(__file__))))

import argparse
import pprint as pp
import re
from datetime import date, datetime

from cleantext import clean
from tqdm import tqdm
from utils import load_dir_files

def parse_whatsapp(text_path:str, verbose:bool=False):
    """
    parse_whatsapp - main function to parse a single conversation exported with whatsapp

    Args:
        text_path (str): [description]
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        textlines = f.readlines()

    textlines = [clean(line) for line in textlines]

    re_string = "\[([0-9]+(\.[0-9]+)+), ([0-9]+(:[0-9]+)+)\] "
    sub_textlines = [re.sub(re_string, "", line) for line in textlines]

    fin_text = []

    for line in sub_textlines:
        line = str(line)
        if "omitted" in line:
            continue # this line just reports an attachment (that is not present)
        else:
            parts = line.split(": ")
            if len(parts) == 2 and isinstance(parts, list):
                fin_text.append(parts[0] + ":\n")
                fin_text.append(parts[1] + "\n")
                fin_text.append("\n")
            elif len(parts) > 2:
                fin_text.append(parts[0] + ":\n")
                fin_text.append(" ".join(parts[1:]) + "\n")
                fin_text.append("\n")
            else:
                continue

    if verbose:
        print("exiting the function have {} lines".format(len(fin_text)))
    return fin_text


# Set up the parsing of command-line arguments
def get_parser():
    parser = argparse.ArgumentParser(
        description="convert whatsapp chat exports to GPT-2 input"
    )
    parser.add_argument(
        "--datadir",
        required=True,
        help="Path to input directory containing txt whatsapp exports",
    )
    parser.add_argument(
        "--outdir",
        required=False,
        default=os.getcwd(),
        help="Path to the output directory, where the output file will be created",
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    input_path = args.datadir
    output_path = args.outdir

    if not os.path.isdir(input_path):
        print("The path specified does not exist")
        sys.exit()

    txt_files = load_dir_files(input_path, verbose=True)

    if len(txt_files) < 1:
        print("Did not find any text files in: \n {}".format(input_path))
        sys.exit()

    train_data = []

    for txtf in tqdm(txt_files, total=len(txt_files), desc="parsing whatsapp files.."):
        reformed = parse_whatsapp(txtf)
        train_data.extend(reformed)

    print("parsed {} lines of text data".format(len(train_data)))

    today_string = date.today().strftime("%b-%d-%Y")
    comp_data_name = "compiled_whatsapp_data_{}.txt".format(today_string)
    f_out_path = join(output_path, comp_data_name)

    with open(f_out_path, "w", encoding="utf-8", errors="ignore") as fo:
        fo.writelines(train_data)

    print("finished - ", datetime.now())
    print("the output file can be found at: \n {}".format(f_out_path))
