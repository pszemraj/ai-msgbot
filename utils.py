import os
import pprint as pp
import re
import shutil  # zipfile formats
from datetime import datetime
from os.path import basename
from os.path import getsize, join

import requests
from natsort import natsorted


def fast_scandir(dirname):
    # return all subfolders in a given filepath

    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders  # list


def create_folder(directory):
    os.makedirs(directory, exist_ok=True)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def chunky_pandas(my_df, num_chunks=4):
    n = int(len(my_df) // num_chunks)
    list_df = [my_df[i : i + n] for i in range(0, my_df.shape[0], n)]

    return list_df


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


def URL_string_filter(text):
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
    URLtoget, extract_loc=None, file_header="dropboxexport_", verbose=False
):
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
