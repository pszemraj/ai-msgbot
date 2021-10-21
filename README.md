# AI Chatbots based on GPT Architecture

A little example / guide for building a chatbot that sounds like you (or some dataset / persona you choose) by training a GPT-based model (either GPT2 or GPT-neo). Note that the training was done on Colab to leverage the GPU.

## Quick outline:

- training and message EDA notebooks in `colab-notebooks`
- python scripts for parsing message data into a standard format for training GPT are in `parsing-messages`
- example data (from the _Daily Dialogues_ dataset) is in `conversation-data`
- model files need to be downloaded from my dropbox (run `download_models.py`) and it will be done "automatically". They will then show up in folders in your cwd

## quickstart

- clone the repo
- with a terminal opened in the repo folder:
  - `pip install -r requirements.txt`
  - `python download_models.py`
  - `python .\ai_single_response.py --responder jonathan --prompt "do you know how to get rich?"`
- then it will respond

the other files (`gptPeter_gpt2_335M.py` and `gptPeter-125M.py` specifically) are work-in-progress attempts to have longer conversations with the model.

---

## TODO and idea list

- try generating 5-10 responses at once instead of n=1, and return the one with the highest harmonic mean sentence score.
  - **Rationale**: based on _UNVALIDATED AND UNQUANTIFIED_ trends in the grid search data (see [gridsearch v1](https://www.dropbox.com/s/uanhf2kuyoybo4x/GPT-Peter%20Hyperparam%20Analysis%20w%20Metrics%20-%20Oct-20-2021_15-49.xlsx?dl=0) and [gridsearch v2](https://www.dropbox.com/s/r2xv66wdfyalwyi/GPT-Peter%20Hyperparam%20Analysis%20w%20Metrics%20-%20Oct-21-2021_02-01.xlsx?dl=0)), the responses that rank high on the harmonic mean score also seem the most coherent and responsive to the question at hand *this is anecdotal
  - jury is still out as to what the intuition / reason behind that is. The _product score_ results being useful makes sense, but these are even better
  - therefore, generating 5-10 reponses at once, scoring them all at once (_check docs for function_) and returning the corresponding highest-scoring prompt should have the bot behaving more realistically. 
  -
---
