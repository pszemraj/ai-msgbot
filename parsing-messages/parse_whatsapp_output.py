import argparse
import os
import pprint as pp
import re
import sys
from datetime import date, datetime
from os.path import basename, join

from cleantext import clean
from natsort import natsorted
from tqdm import tqdm


def load_dir_files(directory, req_extension=".txt", return_type="list", verbose=False):
    appr_files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(directory):
        for prefile in f:
            if prefile.endswith(req_extension):
                fullpath = os.path.join(r, prefile)
                appr_files.append(fullpath)

    appr_files = natsorted(appr_files)

    if verbose:
        print("A list of files in the {} directory are: \n".format(directory))
        if len(appr_files) < 10:
            pp.pprint(appr_files)
        else:
            pp.pprint(appr_files[:10])
            print("\n and more. There are a total of {} files".format(len(appr_files)))

    if return_type.lower() == "list":
        return appr_files
    else:
        if verbose:
            print("returning dictionary")

        appr_file_dict = {}
        for this_file in appr_files:
            appr_file_dict[basename(this_file)] = this_file

        return appr_file_dict


def parse_whatsapp(text_path, verbose=False):
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        textlines = f.readlines()

    textlines = [clean(line) for line in textlines]

    re_string = "\[([0-9]+(\.[0-9]+)+), ([0-9]+(:[0-9]+)+)\] "
    sub_textlines = [re.sub(re_string, "", line) for line in textlines]

    fin_text = []

    for line in sub_textlines:
        line = str(line)
        if "omitted" in line:
            continue
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

if __name__ == "__main__":
    args = parser.parse_args()
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
