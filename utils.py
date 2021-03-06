"""
general utility functions for loading, saving, etc
"""
import os
from pathlib import Path
import pprint as pp
import re
import shutil  # zipfile formats
from datetime import datetime
from os.path import basename
from os.path import getsize, join

import requests
from cleantext import clean
from natsort import natsorted
from symspellpy import SymSpell
import pandas as pd
from tqdm.auto import tqdm


def get_timestamp():
    return datetime.now().strftime("%b-%d-%Y_t-%H")


def correct_phrase_load(my_string: str):
    """
    correct_phrase_load [basic / unoptimized implementation of SymSpell to correct a string]

    Args:
        my_string (str): [text to be corrected]

    Returns:
        [type]: [description]
    """
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

    dictionary_path = (
        r"symspell_rsc/frequency_dictionary_en_82_765.txt"  # from repo root
    )
    bigram_path = (
        r"symspell_rsc/frequency_bigramdictionary_en_243_342.txt"  # from repo root
    )
    # term_index is the column of the term and count_index is the
    # column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    # max edit distance per lookup (per single word, not per whole input string)
    suggestions = sym_spell.lookup_compound(
        clean(my_string), max_edit_distance=2, ignore_non_words=True
    )
    if len(suggestions) < 1:
        return my_string
    else:
        first_result = suggestions[0]
        return first_result._term


def fast_scandir(dirname: str):
    """
    fast_scandir [an os.path-based means to return all subfolders in a given filepath]

    Args:
        dirname (str): [description]

    Returns:
        [list]: [description]
    """

    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders  # list


def create_folder(directory: str):

    os.makedirs(directory, exist_ok=True)


def chunks(lst: list, n: int):
    """
    chunks   -  Yield successive n-sized chunks from lst
    Args:
        lst (list): [description]
        n (int): [description]

    Yields:
        [type]: [description]
    """

    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def chunky_pandas(my_df, num_chunks: int = 4):
    """
    chunky_pandas [split dataframe into `num_chunks` equal chunks, return each inside a list]

    Args:
        my_df (pd.DataFrame): [description]
        num_chunks (int, optional): [description]. Defaults to 4.

    Returns:
        [type]: [description]
    """
    n = int(len(my_df) // num_chunks)
    list_df = [my_df[i : i + n] for i in range(0, my_df.shape[0], n)]

    return list_df


def load_dir_files(
    directory: str, req_extension=".txt", return_type="list", verbose=False
):
    """
    load_dir_files - an os.path based method of returning all files with extension `req_extension` in a given directory and subdirectories

    Args:
        directory (str): [description]
        req_extension (str, optional): [description]. Defaults to ".txt".
        return_type (str, optional): [description]. Defaults to "list".
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
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


def URL_string_filter(text):
    """
    URL_string_filter - filter out nonstandard "text" characters

    Args:
        text ([type]): [description]

    Returns:
        [str]: [description]
    """
    custom_printable = (
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ._"
    )

    filtered = "".join((filter(lambda i: i in custom_printable, text)))

    return filtered


def getFilename_fromCd(cd):
    if not cd:
        return None
    fname = re.findall("filename=(.+)", cd)
    if len(fname) > 0:
        output = fname[0]
    elif cd.find("/"):
        possible_fname = cd.rsplit("/", 1)[1]
        output = URL_string_filter(possible_fname)
    else:
        output = None
    return output


def get_zip_URL(
    URLtoget: str,
    extract_loc: str = None,
    file_header: str = "dropboxexport_",
    verbose: bool = False,
):
    """
    get_zip_URL [summary]

    Args:
        URLtoget (str): [description]
        extract_loc (str, optional): [description]. Defaults to None.
        file_header (str, optional): [description]. Defaults to "dropboxexport_".
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    r = requests.get(URLtoget, allow_redirects=True)
    names = getFilename_fromCd(r.headers.get("content-disposition"))
    fixed_fnames = names.split(";")  # split the multiple results
    this_filename = file_header + URL_string_filter(fixed_fnames[0])

    # define paths and save the zip file
    if extract_loc is None:
        extract_loc = "dropbox_dl"
    dl_place = join(os.getcwd(), extract_loc)
    create_folder(dl_place)
    save_loc = join(os.getcwd(), this_filename)
    open(save_loc, "wb").write(r.content)
    if verbose:
        print("downloaded file size was {} MB".format(getsize(save_loc) / 1000000))

    # unpack the archive
    shutil.unpack_archive(save_loc, extract_dir=dl_place)
    if verbose:
        print("extracted zip file - ", datetime.now())
        x = load_dir_files(dl_place, req_extension="", verbose=verbose)

    # remove original
    try:
        os.remove(save_loc)
        del save_loc
    except:
        print("unable to delete original zipfile - check if exists", datetime.now())

    print("finished extracting zip - ", datetime.now())

    return dl_place


def merge_dataframes(data_dir: str, ext=".xlsx", verbose=False):
    """
    merge_dataframes - given a filepath, loads and attempts to merge all files as dataframes

    Args:
        data_dir (str): [root directory to search in]
        ext (str, optional): [anticipate file extension for the dataframes ]. Defaults to '.xlsx'.

    Returns:
        pd.DataFrame(): merged dataframe
    """

    src = Path(data_dir)
    src_str = str(src.resolve())
    mrg_df = pd.DataFrame()

    all_reports = load_dir_files(directory=src_str, req_extension=ext, verbose=verbose)

    failed = []

    for df_path in tqdm(all_reports, total=len(all_reports), desc="joining data..."):

        try:
            this_df = pd.read_excel(df_path).convert_dtypes()

            mrg_df = pd.concat([mrg_df, this_df], axis=0)
        except:
            short_p = os.path.basename(df_path)
            print(
                f"WARNING - file with extension {ext} and name {short_p} could not be read."
            )
            failed.append(short_p)

    if len(failed) > 0:
        print("failed to merge {} files, investigate as needed")

    if verbose:
        pp.pprint(mrg_df.info(True))

    return mrg_df
