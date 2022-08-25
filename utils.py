"""
general utility functions for loading, saving, and manipulating data
"""

import os
import logging
import pprint as pp
import re
import shutil  # zipfile formats
import warnings
from datetime import datetime
from os.path import basename, getsize, join
from pathlib import Path
import logging

import pandas as pd
import requests
from cleantext import clean
from natsort import natsorted
from symspellpy import SymSpell
from tqdm.auto import tqdm

import warnings

warnings.filterwarnings(action="ignore", message=".*the GPL-licensed package `unidecode` is not installed*") # cleantext GPL-licensed package reminder is annoying

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    errors="replace",
    level=logging.ERROR,
)

def get_timestamp():
    return datetime.now().strftime("%b-%d-%Y_t-%H")


def print_spacer(n=1):
    """print_spacer - print a spacer line"""
    print("\n   --------    " * n)


def remove_trailing_punctuation(text: str):
    """
    remove_trailing_punctuation - remove trailing punctuation from a string

    Args:
        text (str): [string to be cleaned]

    Returns:
        [str]: [cleaned string]
    """
    return text.strip("?!.,;:")


def correct_phrase_load(my_string: str):
    """
    correct_phrase_load [basic / unoptimized implementation of SymSpell to correct a string]

    Args:
        my_string (str): [text to be corrected]

    Returns:
        str: the corrected string
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
    Args:   lst (list): list to be chunked
    n (int): size of chunks

    """

    for i in range(0, len(lst), n):
        yield lst[i : i + n]



def shorten_list(
    list_of_strings: list, max_chars: int = 512, no_blanks=True, verbose=False
):
    """a helper function that iterates through a list backwards, adding to a new list.

        When <max_chars> is met, that list entry is not added.
    Args:
        list_of_strings (list): list of strings to be shortened
        max_chars (int, optional): maximum number of characters in a the list in total. Defaults to 512.
        no_blanks (bool, optional): if True, blank strings are not added to the new list. Defaults to True.
        verbose (bool, optional): if True, print the list of strings before and after the shorten. Defaults to False.
    """
    list_of_strings = [str(x) for x in list_of_strings] # convert to strings if not already
    shortened_list = []
    total_len = 0
    for i, string in enumerate(list_of_strings[::-1], start=1):

        if len(string.strip()) == 0 and no_blanks:
            continue
        if len(string) + total_len >= max_chars:
            logging.info(f"string # {i} puts total over limit, breaking ")
            break
        total_len += len(string)
        shortened_list.insert(0, string)
    if len(shortened_list) == 0:
        logging.info(f"shortened list with max_chars={max_chars} has no entries")
    if verbose:
        print(f"total length of list is {total_len} chars")
    return shortened_list




def chunky_pandas(my_df, num_chunks: int = 4):
    """
    chunky_pandas [split dataframe into `num_chunks` equal chunks, return each inside a list]

    Args:
        my_df (pd.DataFrame)
        num_chunks (int, optional): Defaults to 4.

    Returns:
        list: a list of dataframes
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


    Returns:
        list or dict: an iterable of filepaths or a dict of filepaths and their respective filenames
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

    """
    custom_printable = (
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ._"
    )

    filtered = "".join((filter(lambda i: i in custom_printable, text)))

    return filtered


def getFilename_fromCd(cd):
    """getFilename_fromCd - get the filename from a given cd str"""
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
    """get_zip_URL - download a zip file from a given URL and extract it to a given location"""

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
    except Exception:
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
        pd.DataFrame(): merged dataframe of all files
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
        except Exception:
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


def download_URL(url: str, file=None, dlpath=None, verbose=False):
    """
    download_URL - download a file from a URL and show progress bar

    Parameters
    ----------
    url : str,        URL to download
    file : str, optional, default None, name of file to save to. If None, will use the filename from the URL
    dlpath : str, optional, default None, path to save the file to. If None, will save to the current working directory
    verbose : bool, optional, default False, print progress bar

    Returns
    -------
    str - path to the downloaded file
    """

    if file is None:
        if "?dl=" in url:
            # is a dropbox link
            prefile = url.split("/")[-1]
            filename = str(prefile).split("?dl=")[0]
        else:
            filename = url.split("/")[-1]

        file = clean(filename)
    if dlpath is None:
        dlpath = Path.cwd()  # save to current working directory
    else:
        dlpath = Path(dlpath)  # make a path object

    r = requests.get(url, stream=True, allow_redirects=True)
    total_size = int(r.headers.get("content-length"))
    initial_pos = 0
    dl_loc = dlpath / file
    with open(str(dl_loc.resolve()), "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=file,
            initial=initial_pos,
            ascii=True,
        ) as pbar:
            for ch in r.iter_content(chunk_size=1024):
                if ch:
                    f.write(ch)
                    pbar.update(len(ch))

    if verbose:
        print(f"\ndownloaded {file} to {dlpath}\n")

    return str(dl_loc.resolve())


def dl_extract_zip(
    URLtoget: str,
    extract_loc: str = None,
    file_header: str = "TEMP_archive_dl_",
    verbose: bool = False,
):
    """
    dl_extract_zip - generic function to download a zip file and extract it

    Parameters
    ----------
    URLtoget : str, zip file URL to download
    extract_loc : str, optional, default None, path to save the zip file to. If None, will save to the current working directory
    file_header : str, optional, default 'TEMP_archive_dl_', prefix for the zip file name
    verbose : bool, optional, default False, print progress bar

    Returns
    -------
    str - path to the downloaded and extracted folder
    """

    extract_loc = Path(extract_loc)
    extract_loc.mkdir(parents=True, exist_ok=True)

    save_loc = download_URL(
        url=URLtoget, file=f"{file_header}.zip", dlpath=None, verbose=verbose
    )

    shutil.unpack_archive(save_loc, extract_dir=extract_loc)

    if verbose:
        print("extracted zip file - ", datetime.now())
        x = load_dir_files(extract_loc, req_extension="", verbose=verbose)

    # remove original
    try:
        os.remove(save_loc)
        del save_loc
    except Exception as e:
        warnings.warn(message=f"unable to delete original zipfile due to {e}")
    if verbose:
        print("finished extracting zip - ", datetime.now())

    return extract_loc

