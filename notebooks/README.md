# README

This directory contains notebooks for training and testing GPT generation models. Notebooks in the top-level directory can be run locally and are not required to be run on the cloud. Unless otherwise noted, the notebooks in subdirectories are required to be run on the cloud (if a use-case arises to use them locally, some changes are required).

The following notebooks & directories are available:

- colab-aitextgen/
- colab-huggingface-API/
- hf_hub_push.ipynb
- wiki_wizard_eda_parse.ipynb

The purpose of each of the above is:

| Item                        | Purpose                                                                                                                          |
|:----------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| colab-aitextgen/            | contains notebooks for training and testing GPT generation models with aiTextGen                                                 |
| colab-huggingface-API/      | contains notebooks for training and testing GPT generation models with HuggingFace  trainer API and advanced training techniques |
| hf_hub_push.ipynb           | a notebook guide for exporting models from aitextgen to HuggingFace Hub                                                          |
| wiki_wizard_eda_parse.ipynb | this is a notebook for parsing the WikiWizard EDA data and creating a dataframe for use in training GPT generation models        |

## aiTextGen

aitextgen is an open-source library for generating text using GPT-2. It is a Python library that uses the GPT model to generate text and is better suited to an introduction to GPT-2.

## HuggingFace

The HuggingFace API can be used to train GPT models. It's the classical _transformers_ library and is more feature-rich than aitextgen, but it is not as easy to use.

* * *
