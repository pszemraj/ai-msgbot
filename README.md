# AI Chatbots based on GPT Architecture

A little example / guide for building a chatbot that sounds like you (or some dataset / persona you choose) by training a GPT-based model (either GPT2 or GPT-neo). Primarily relies on the `aitextgen` and `python-telegram-bot` libraries.

> Note that the training was done on Colab to leverage the GPU.

```bazaar
C:\Users\peter\PycharmProjects\gpt2_chatbot>python ai_single_response.py --model gpt2_dailydialogue_355M_75Ksteps --prompt "what are you doing?" --temp 0.8 --topp
 0.9

... generating...

finished!

"i was just going to show that it's all about $ 5."
```

-   you can message the bot by clicking [this link](https://t.me/GPTPeter_bot)

* * *

<!-- TOC -->

-   [AI Chatbots based on GPT Architecture](#ai-chatbots-based-on-gpt-architecture)
    -   [Quick outline of repo:](#quick-outline-of-repo)
    -   [quickstart](#quickstart)
    -   [TODO and idea list](#todo-and-idea-list)
    -   [Extras, Asides, and Examples](#extras-asides-and-examples)

<!-- /TOC -->

* * *

## Quick outline of repo:

-   training and message EDA notebooks in `colab-notebooks`
-   python scripts for parsing message data into a standard format for training GPT are in `parsing-messages`
-   example data (from the _Daily Dialogues_ dataset) is in `conversation-data`
-   model files need to be downloaded from my dropbox (run `download_models.py`) and it will be done "automatically". They will then show up in folders in your cwd

**A friend's screenshot: an example bot response on Telegram**

<img src="https://user-images.githubusercontent.com/74869040/138378871-d3508ce8-8dd0-45b8-92e2-92bc5ae1d530.jpg" width="600" height="200">

## quickstart

-   clone the repo
-   with a terminal opened in the repo folder:
    -   `pip install -r requirements.txt`
    -   `python download_models.py`
    -   `python .\ai_single_response.py --responder jonathan --prompt "do you know how to get rich?"`
-   then it will respond!
-   other models are available / will be downloaded, to change the model that generates a response you can pass the argument `--model` for example:


    python ai_single_response.py --model "GPT2_dailydialogue_355M_150Ksteps" --prompt "are you free tomorrow?"

the other files (`gptPeter_gpt2_335M.py` and `gptPeter-125M.py` specifically) are work-in-progress attempts to have longer conversations with the model.

* * *

## TODO and idea list

1.  try generating 5-10 responses at once instead of n=1, and return the one with the highest [harmonic mean sentence score](https://github.com/simonepri/lm-scorer).

- > **Rationale**: based on _UNVALIDATED AND UNQUANTIFIED_ trends in the grid search data (see [gridsearch v1](https://www.dropbox.com/s/uanhf2kuyoybo4x/GPT-Peter%20Hyperparam%20Analysis%20w%20Metrics%20-%20Oct-20-2021_15-49.xlsx?dl=0) and [gridsearch v2](https://www.dropbox.com/s/r2xv66wdfyalwyi/GPT-Peter%20Hyperparam%20Analysis%20w%20Metrics%20-%20Oct-21-2021_02-01.xlsx?dl=0)), the responses that rank high on the harmonic mean score also seem the most coherent and responsive to the question at hand \*this is anecdotal
-   > jury is still out as to what the intuition / reason behind that is. The _product score_ results being useful makes sense, but these are even better
-   > therefore, generating 5-10 reponses at once, scoring them all at once (_check docs for function_) and returning the corresponding highest-scoring prompt should have the bot behaving more realistically.

2.  continue with hyperparamter optimization on 774M model GPT-Peter. Status of hyperparameter "search" is kept (_and will be updated_) [here](https://gpt-peter-eda.netlify.app/)

-   > examine if any basic ML approaches can model the harmonic mean with [Pycaret](http://www.pycaret.org/tutorials/html/REG102.html)

3.  investigate whatsapp bot potential and utility
4.  evaluate if pretrained on the _Daily Dialogues_ data set and then training for other purposes helps with the "transfer learning" of teaching the GPT model that it is now a chatbot vs. just directly training the "standard" checkpoint

-   > in short, `355M checkpoint -> daily dialogues -> message data` vs. `355M checkpoint -> message data`

5.  evaluate whether pretraining on other datasets, such as [CoQA (Conversational Question Answering Challenge)](https://paperswithcode.com/dataset/coqa) or [TriviaQA](https://paperswithcode.com/dataset/triviaqa) improves transfer learning to being a chatbot
    -   > this applies for a text message chat bot _and_ also the "resources for learning english in a safer environment" bot

* * *

## Extras, Asides, and Examples

_An example of end-of-pipeline capabilities (further tuning to come)_

<img src="https://user-images.githubusercontent.com/74869040/138378926-03c57fa5-d3e9-4a9b-a463-df4b7f66a6af.jpg" width="420" height="960">

-   **Aside: the submitter of this image is also in the analytics club @ ETH Zurich, which the bot knew to reference.**
