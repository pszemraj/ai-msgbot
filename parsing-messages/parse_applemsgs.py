"""
parsing-messages\parse_applemsgs.py

Similar to the whatsapp parser, but this parses output CSVs of iphone / apple texts.

Note that this script was geared towards data in the iMazing export format/structure, as my messaage data was exported in this way. if message data is exported in another way, it's likely that some manipulation of the columns is required (which is doable! just fyi).

"""

import os
import random
import sys
from os.path import basename, dirname, join

sys.path.append(dirname(dirname(os.path.abspath(__file__))))

import argparse
import pprint as pp
from datetime import date, datetime
from os.path import basename, join

import pandas as pd
from cleantext import clean
from natsort import natsort_keygen
from tqdm import tqdm
from utils import load_dir_files


def clean_message_apple(text: str, lang: str = "en"):
    """
    clean_message_apple - clean the message text of any non-ascii characters

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
        no_phone_numbers=False,  # replace all phone numbers with a special token
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


def parse_apple_msg(
    csv_path: str, lang: str = "en", sender_name: str = "peter szemraj", verbose=False
):
    """
    parse_apple_msg - parses a csv of messages exported from a device, in Apple format (i.e. has specific apple columns and/or artifacts in messages)

    Args:
        csv_path (str): [path to a CSV file containing messages and other relevant info]
        verbose (bool, optional): [debug printouts]. Defaults to False.

    Returns:
        [list]: [returns a list of strings, each representing a line in the dialogue "script"]
    """

    df = pd.read_csv(csv_path).convert_dtypes()

    clean_df = df[df.Text.notnull()]
    clean_df = clean_df[~clean_df["Text"].str.contains('\n"*."', na=False, regex=False)]
    clean_df = clean_df[
        ~clean_df["Text"].str.contains("an image", na=False, regex=False)
    ]
    emote_words = ["Liked", "Disliked", "Loved", "Emphasized"]
    del_rows = []
    for index, row in clean_df.iterrows():

        if len(row["Text"].split(" ")) > 0:
            first_word = row["Text"].split(" ")[0]
            if any(substring in first_word for substring in emote_words):
                del_rows.append(index)

    clean_df.drop(del_rows, axis=0, inplace=True)
    clean_df.reset_index(drop=True, inplace=True)

    if verbose:
        pp.pprint(clean_df.info(verbose=True))

    srt_df = clean_df.copy()
    srt_df["Message Date"] = pd.to_datetime(
        srt_df["Message Date"], infer_datetime_format=True
    )

    srt_df.sort_values(
        by="Message Date", key=natsort_keygen(), inplace=True, ascending=True
    )

    srt_df.reset_index(drop=True, inplace=True)
    conv_words = []

    for index, row in srt_df.iterrows():

        row_text = clean_message_apple(str(row["Text"]), lang=lang).strip()

        if len(row_text) < 2:
            continue
        else:
            if row["Type"] == "Outgoing":
                conv_words.append(f"{sender_name}:" + "\n")
            elif pd.notna(row["Sender Name"]):
                conv_words.append(str(row["Sender Name"]) + ":\n")
            else:
                conv_words.append(str(row["Sender ID"]) + ":\n")

            conv_words.append(row_text + "\n")
            conv_words.append("\n")

    if verbose:
        print("exiting the function, have {} lines of text".format(len(conv_words)))
    return conv_words


# Set up the parsing of command-line arguments
def get_parser():
    """
    get_parser - helper function for argparse
    """
    parser = argparse.ArgumentParser(
        description="convert csv apple message exports to GPT-2 input"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Path to input directory containing .csv apple message exports",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=False,
        default=os.getcwd(),
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
        action="store_true",
        help="Print out debug information",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    input_path = args.input_dir
    output_path = args.output_dir
    lang = args.lang
    verbose = args.verbose
    if not os.path.isdir(input_path):
        print("The path specified does not exist")
        sys.exit()

    csv_files = load_dir_files(input_path, req_extension=".csv", verbose=verbose)

    if len(csv_files) < 1:
        print("Did not find any CSV files in: \n {}".format(input_path))
        sys.exit()
    random.shuffle(csv_files)
    train_data = []

    for txtf in tqdm(csv_files, total=len(csv_files), desc="parsing msg .CSV files.."):
        reformed = parse_apple_msg(txtf, lang=lang, verbose=verbose)
        train_data.extend(reformed)

    print("parsed {} lines of text data".format(len(train_data)))

    today_string = date.today().strftime("%b-%d-%Y")
    comp_data_name = "compiled_apple_msg_data_{}.txt".format(today_string)
    f_out_path = join(output_path, comp_data_name)

    with open(f_out_path, "w", encoding="utf-8", errors="ignore") as fo:
        fo.writelines(train_data)

    print("finished - ", datetime.now())
    print("the output file can be found at: \n {}".format(f_out_path))
