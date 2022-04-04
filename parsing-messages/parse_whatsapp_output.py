"""
parsing-messages\parse_whatsapp_output.py

parse messages exported from WhatsApp via the standard "whatsapp export" process. Assumes that all whatsapp exports are stored somewhere in the user-provided directory, and that they remain in the default export structure of <messages with X contact>/<poorly_labeled_textfile_of_messages.txt>
"""

import os
import sys
from os.path import dirname, join, basename
import warnings

sys.path.append(dirname(dirname(os.path.abspath(__file__))))

import argparse
import pprint as pp
import re
from datetime import date, datetime

from cleantext import clean
from tqdm import tqdm
from utils import load_dir_files
from pathlib import Path


def clean_message_whatsapp(text: str, lang: str = "en"):
    """
    clean_message_whatsapp - clean the message text of any non-ascii characters

    Args:
        text (str): [the message text to be cleaned]
        lang (str, optional): [the language of the message]. Defaults to "en". set to 'de' for German special handling

    Returns:
        [str]: [the cleaned message text]
    """

    clean_text = clean(
        text,
        fix_unicode=True,  # fix various unicode errors
        to_ascii=True,  # transliterate to closest ASCII representation
        lower=True,  # lowercase text
        no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
        no_urls=True,  # replace all URLs with a special token
        no_emails=True,  # replace all email addresses with a special token
        no_phone_numbers=True,  # replace all phone numbers with a special token
        no_numbers=False,  # replace all numbers with a special token
        no_digits=False,  # replace all digits with a special token
        no_currency_symbols=False,  # replace all currency symbols with a special token
        no_punct=False,  # remove punctuations
        replace_with_punct="",  # instead of removing punctuations you may replace them
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang=lang,
    )
    return clean_text


def get_omission_criteria(**args):
    """
    get_omission_criteria - get the list of strings that indicate a message should be omitted

    Args:
        **args: [keyword arguments]

    Returns:
        [list]: [a list of strings, each corresponding to a line in the overall script]
    """
    default_msg = "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them."
    deleted_message_a = "deleted this message"
    deleted_message_b = "This message was deleted"
    omission_criteria = ["omitted", default_msg, deleted_message_a, deleted_message_b]

    # add any additional criteria to the list
    if args is not None:
        omission_criteria.extend(args)
    return omission_criteria


def parse_whatsapp(text_path: str, lang: str = "en", verbose: bool = False):
    """
    parse_whatsapp - main function to parse a single conversation exported with whatsapp

    Args:
        text_path (str): [path to a text file]
        lang (str, optional): [the language of the message]. Defaults to "en". set to 'de' for German special handling
        verbose (bool, optional): [print additional outputs for debugging]. Defaults to False.

    Returns:
        [list]: [a list of strings, each corresponding to a line in the overall script]
    """
    omission_criteria = get_omission_criteria()

    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        textlines = f.readlines()

    re_string = "\[([0-9]+(\.[0-9]+)+), ([0-9]+(:[0-9]+)+)\] "
    no_time_textlines = [re.sub(re_string, "", line) for line in textlines]
    sub_textlines = [
        clean_message_whatsapp(line, lang=lang) for line in no_time_textlines
    ]
    if verbose:
        print(f"The first 2 processed lines are:\n{sub_textlines[:2]}")
    fin_text = []

    for line in sub_textlines:
        line = line.strip() if isinstance(line, str) else line[0].strip()
        if any(x.lower() in line.lower() for x in omission_criteria):
            continue  # omit this line
        else:
            # split the line into two parts, before and after the first colon
            message_parts = line.split(":", 1)
            if len(message_parts) > 2:
                warnings.warn(f"There are more than one colon in this line: {line}")
            if isinstance(message_parts, list) and len(message_parts) == 2:
                part1 = message_parts[0].strip()
                part2 = message_parts[1].strip()
                if len(part2) < 2:
                    continue  # this line is just a timestamp
                fin_text.append(part1 + ":\n")
                fin_text.append(part2 + "\n")
                fin_text.append("\n")
            elif isinstance(message_parts, str) and len(message_parts) > 4:
                fin_text.append(message_parts + "\n")
                fin_text.append("\n")
            else:
                continue

    if verbose:
        print("exiting the function have {} lines".format(len(fin_text)))
    return fin_text


# Set up the parsing of command-line arguments
def get_parser():
    parser = argparse.ArgumentParser(
        description="convert whatsapp chat exports to a single text file",
    )
    parser.add_argument(
        "-i",
        "--datadir",
        required=True,
        help="Path to input directory containing txt whatsapp exports",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        required=False,
        default=str(Path.cwd().resolve()),
        help="Path to the output directory, where the output file will be created",
    )
    parser.add_argument(
        "-l",
        "--lang",
        required=False,
        default="en",
        help="The language of the messages. Defaults to 'en'",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        action="store_true",
        help="Print additional outputs for debugging",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    input_path = args.datadir
    output_path = args.outdir
    verbose = args.verbose
    lang = args.lang
    if not os.path.isdir(input_path):
        print("The path specified does not exist")
        sys.exit()

    txt_files = load_dir_files(input_path, verbose=True)

    if len(txt_files) < 1:
        print("Did not find any text files in: \n {}".format(input_path))
        sys.exit()
    if verbose:
        print("Found {} text files in: \n {}".format(len(txt_files), input_path))

    train_data = []

    for txtf in tqdm(txt_files, total=len(txt_files), desc="parsing whatsapp files.."):
        reformed = parse_whatsapp(txtf, verbose=verbose, lang=lang)
        train_data.extend(reformed)

    print("parsed {} lines of text data".format(len(train_data)))

    today_string = date.today().strftime("%b-%d-%Y")
    comp_data_name = "compiled_whatsapp_data_{}.txt".format(today_string)
    f_out_path = join(output_path, comp_data_name)

    with open(f_out_path, "w", encoding="utf-8", errors="ignore") as fo:
        fo.writelines(train_data)

    print("finished - ", datetime.now())
    print("the output file can be found at: \n {}".format(f_out_path))
