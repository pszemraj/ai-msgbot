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
