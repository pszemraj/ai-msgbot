#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A cli helper script to take a model and prepare it for uploading to huggingface.
"""
import os
import sys
from os.path import dirname

# add the path to the script to the sys.path
sys.path.append(dirname(dirname(os.path.abspath(__file__))))
import warnings

warnings.filterwarnings(action="ignore", message=".*gradient_checkpointing*")

import argparse
from pathlib import Path

from aitextgen import aitextgen

from utils import get_timestamp

# Set up the parsing of command-line arguments
def get_parser():
    """
    get_parser [a helper function for the argparse module]

    Returns: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="convert a model to a format that can be uploaded to huggingface"
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        required=True,  # model_dir is needed to find the model files
        type=str,
        help="filepath to directory that contains the model files (pytorch.bin + config.json)",
    )

    parser.add_argument(
        "-n",
        "--hf-name",
        required=False,
        type=str,
        help="the name of the model to be uploaded to huggingface",
    )

    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="pass this argument if you want all the printouts",
    )

    return parser


if __name__ == "__main__":

    # Set up the parsing of command-line arguments
    parser = get_parser()
    args = parser.parse_args()
    want_verbose = args.verbose
    if want_verbose:
        print(f"args: {args}")
    model_dir = str(args.model_dir)
    model_path = Path(os.path.abspath(model_dir))
    hf_name = (
        str(args.hf_name) if args.hf_name is not None else str(model_path.name)
    )  # model_path.name is the backup name
    hf_name = hf_name.replace(" ", "_")
    hf_name = hf_name.lower().strip()

    # load the model
    if want_verbose:
        print("Loading model...")
    ai = aitextgen(
        model_folder=model_path.resolve(),
        to_gpu=False,
    )
    # get the model config and save it
    if want_verbose:
        print("saving model config...")
    ai.save_for_upload(hf_name)

    print(f"created {hf_name} for uploading to huggingface - {get_timestamp()}")
