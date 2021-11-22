"""
parsing-messages\parse_applemsgs.py

Similar to the whatsapp parser, but this parses output CSVs of iphone / apple texts.

Note that this script was geared towards data in the iMazing export format/structure, as my messaage data was exported in this way. if message data is exported in another way, it's likely that some manipulation of the columns is required (which is doable! just fyi). 

"""

import os
import sys
from os.path import dirname, join, basename

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


def parse_apple_msg(csv_path: str, verbose=False):
    """
    parse_apple_msg [parses a csv of messages exported from a device, in Apple format (i.e. has specific apple columns and/or artifacts in messages)]

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
    conv_words = []

    for index, row in srt_df.iterrows():

        if row["Type"] == "Outgoing":
            conv_words.append("peter szemraj:" + "\n")
        elif pd.notna(row["Sender Name"]):
            conv_words.append(str(row["Sender Name"]) + ":\n")
        else:
            conv_words.append(str(row["Sender ID"]) + ":\n")

        conv_words.append(clean(str(row["Text"])) + "\n")
        conv_words.append("\n")

    if verbose:
        print("exiting the function, have {} lines of text".format(len(conv_words)))
    return conv_words


# Set up the parsing of command-line arguments
def get_parser():
    """
    get_parser - helper function for argparse

    Returns:
        [argparse.ArgumentParser]: [parsed arguments from user]
    """
    parser = argparse.ArgumentParser(
        description="convert csv apple message exports to GPT-2 input"
    )
    parser.add_argument(
        "--datadir",
        required=True,
        help="Path to input directory containing .csv apple message exports",
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

    csv_files = load_dir_files(input_path, req_extension=".csv", verbose=True)

    if len(csv_files) < 1:
        print("Did not find any CSV files in: \n {}".format(input_path))
        sys.exit()

    train_data = []

    for txtf in tqdm(csv_files, total=len(csv_files), desc="parsing msg .CSV files.."):
        reformed = parse_apple_msg(txtf)
        train_data.extend(reformed)

    print("parsed {} lines of text data".format(len(train_data)))

    today_string = date.today().strftime("%b-%d-%Y")
    comp_data_name = "compiled_apple_msg_data_{}.txt".format(today_string)
    f_out_path = join(output_path, comp_data_name)

    with open(f_out_path, "w", encoding="utf-8", errors="ignore") as fo:
        fo.writelines(train_data)

    print("finished - ", datetime.now())
    print("the output file can be found at: \n {}".format(f_out_path))
