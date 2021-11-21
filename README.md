# AI Chatbots based on GPT Architecture
![ACE-newsletter-workshop-img-cropped](https://user-images.githubusercontent.com/74869040/141669429-6bfd3e3f-2436-482b-b7b3-367bee6c23d3.png)

A little example / guide for building a chatbot that sounds like you (or some dataset / persona you choose) by training a GPT-based model (either GPT2 or GPT-neo). Primarily relies on the `[aitextgen](https://github.com/minimaxir/aitextgen)` and `[python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)` libraries.

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

**Table of Contents**

<!-- TOC -->

- [AI Chatbots based on GPT Architecture](#ai-chatbots-based-on-gpt-architecture)
  - [Quick outline of repo:](#quick-outline-of-repo)
  - [quickstart](#quickstart)
  - [Repo Overview and Usage](#repo-overview-and-usage)
    - [Training a model](#training-a-model)
      - [Training: Details](#training-details)
    - [Parsing Message Data](#parsing-message-data)
    - [Interaction with a Trained model](#interaction-with-a-trained-model)
      - [Model Responses: Spelling / Grammar Correction](#model-responses-spelling--grammar-correction)
  - [TODO and idea list](#todo-and-idea-list)
  - [Extras, Asides, and Examples](#extras-asides-and-examples)
    - [Other resources](#other-resources)

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
    -   `pip install -r requirements.txt` if using conda: `conda env create --file environment.yml`
        -   _NOTE:_ if any errors with the conda install, it may ask for an environment name which is `gpt2_chatbot`
    -   `python download_models.py`
    -   `python .\ai_single_response.py --responder jonathan --prompt "do you know how to get rich?"`
    -   `MacOS: python ./ai_single_response.py --responder jonathan --prompt "do you know how to get rich?"`

-   then it will respond!
-   other models are available / will be downloaded, to change the model that generates a response you can pass the argument `--model` for example:

      python ai_single_response.py --model "GPT2_dailydialogue_355M_150Ksteps" --prompt "are you free tomorrow?"

the other files (`gptPeter_gpt2_335M.py` and `gptPeter-125M.py` specifically) are work-in-progress attempts to have longer conversations with the model.

* * *

## Repo Overview and Usage

### Training a model

-   the first step in understanding what is going on here is to understand ultimately what is happening is teaching GPT-2 to recognize a "script" of messages and respond as such.
    -   this is done with the `aitextgen` library, and it's recommended to read through [some of their docs](https://docs.aitextgen.io/tutorials/colab/) and take a look at the _Training your model_ section before returning here.
-   essentially, to generate a novel chatbot from just text (without going through too much trouble as required in other libraries.. can you abstract your friends whatsapp messages easily into a "persona"?)

**An example of what a "script" is:**

    speaker a:
    hi, becky, what's up?

    speaker b:
    not much, except that my mother-in-law is driving me up the wall.

    speaker a:
    what's the problem?

    speaker b:
    she loves to nit-pick and criticizes everything that i do. i can never do anything right when she's around.

    speaker a:
    for example?

    speaker b:
    well, last week i invited her over to dinner. my husband and i had no problem with the food, but if you listened to her, then it would seem like i fed her old meat and rotten vegetables. there's just nothing can please her.

-   then, you leverage the text-generative model to reply to messages. This is done by "behind the scenes" parsing/presenting the query with either a real or artificial speaker name, and having the response be from `target_name` and in the case of GPT-Peter, is me.
    Then, you leverage the
    Then, you leverage the
-   depending on compute resources and so forth, it is possible to keep track of the conversation in a helper script/loop, and then feed in the prior conversation and _then_ the prompt so the model can use the context as part of the generation sequence, with of course the [attention mechanism](https://arxiv.org/abs/1706.03762) ultimately focusing on the last text past to it (the actual prompt)
-   then, a matter of deploying it, whether it is a bot that can help children learn conversation, local lingo, etc in a foreign language or a whatsapp bot to automate social upkeep, the possibilities are endless.
-

#### Training: Details

-   an example dataset (_Daily Dialogues_) parsed into the script format can be found locally at `conversation-data\Daily-Dialogues\daily_dialogue_augment.txt`.
    -   When learning, this is probably best to use to finetune the GPT2-model, but there are several other datasets (that need to be parsed) available in the repo at `*\datasets`
    -   many more datasets are available online at [PapersWithCode](https://paperswithcode.com/datasets?task=dialogue-generation&mod=texts) and [GoogleResearch](https://research.google/tools/datasets/). Seems that _GoogleResearch_ also has a tool for searching for datasets online.

<font color="yellow"> TODO: more words </font>

### Parsing Message Data

<font color="yellow"> TODO </font>

### Interaction with a Trained model

-   command line
-   "bot mode"
    -   telegram
    -   whatsapp (still mostly unexplored)
-   _real deployment_ @ jonathan lehner\_

<font color="yellow"> TODO </font>

#### Model Responses: Spelling / Grammar Correction

-   obviously, one of the primary goals of this project is to be able to train a chatbot/QA bot that can respond to the user "unaided" where it does not need hardcoding to be able to handle questions / edge cases. That said, sometimes the model will generate a bunch of strings together,, and applying "general" spell correction helps keep the model responses as comprehensible as possible without interfering with the response / semantics itself.
-   two methods of doing this are currently added:
    -   symspell (via the pysymspell library) _NOTE: while this is fast and works, it sometimes corrects out common text abbreviations to random other short words that are hard to understand, i.e. **tues** and **idk** and so forth_
    -   gramformer (via transformers `pipeline()`object). a pretrained NN that corrects grammar and (to be tested) hopefully does not have the issue described above. Links: [model page](https://huggingface.co/prithivida/grammar_error_correcter_v1), [the models github](https://github.com/PrithivirajDamodaran/Gramformer/) (\_note: not using this because it isnt a pypy package, so i just use the hosted model on huggingface), [hf docs on pipelines() object](https://huggingface.co/transformers/main_classes/pipelines.html?highlight=textgeneration)

## TODO and idea list

1.  try generating 5-10 responses at once instead of n=1, and return the one with the highest [harmonic mean sentence score](https://github.com/simonepri/lm-scorer).

-   > **Rationale**: based on _UNVALIDATED AND UNQUANTIFIED_ trends in the grid search data (see [gridsearch v1](https://www.dropbox.com/s/uanhf2kuyoybo4x/GPT-Peter%20Hyperparam%20Analysis%20w%20Metrics%20-%20Oct-20-2021_15-49.xlsx?dl=0) and [gridsearch v2](https://www.dropbox.com/s/r2xv66wdfyalwyi/GPT-Peter%20Hyperparam%20Analysis%20w%20Metrics%20-%20Oct-21-2021_02-01.xlsx?dl=0)), the responses that rank high on the harmonic mean score also seem the most coherent and responsive to the question at hand \*this is anecdotal
-   > jury is still out as to what the intuition / reason behind that is. The _product score_ results being useful makes sense, but these are even better
-   > therefore, generating 5-10 reponses at once, scoring them all at once (_check docs for function_) and returning the corresponding highest-scoring prompt should have the bot behaving more realistically.

2.  continue with hyperparamter optimization on 774M model GPT-Peter. Status of hyperparameter "search" is kept (_and will be updated_) [here](https://gpt-peter-eda.netlify.app/)

-   > examine if any basic ML approaches can model the harmonic mean with [Pycaret](http://www.pycaret.org/tutorials/html/REG102.html)

3.  investigate whatsapp bot potential and utility
4.  ~~evaluate if pretrained on the _Daily Dialogues_ data set and then training for other purposes helps with the "transfer learning" of teaching the GPT model that it is now a chatbot vs. just directly training the "standard" checkpoint~~

-   > ~~in short, `355M checkpoint -> daily dialogues -> message data` vs. `355M checkpoint -> message data`~~
-   _yes, it does improve things a lot_ TODO: writeup theory 

5.  ~~evaluate whether pretraining on other datasets, such as [CoQA (Conversational Question Answering Challenge)](https://paperswithcode.com/dataset/coqa) or [TriviaQA](https://paperswithcode.com/dataset/triviaqa) improves transfer learning to being a chatbot~~

    -   > ~~this applies for a text message chat bot _and_ also the "resources for learning english in a safer environment" bot~~
    -   _using Trivia/CoCaQA did help model responses_

6.  ~~try gradio deployment~~
    -   _implemented_
8.  try huggingface spaces deployment
9.  Auto_ML based approach to see if multi dimensional hyperparameter

* * *

## Extras, Asides, and Examples

_An example of end-of-pipeline capabilities (further tuning to come)_

<img src="https://user-images.githubusercontent.com/74869040/138378926-03c57fa5-d3e9-4a9b-a463-df4b7f66a6af.jpg" width="420" height="960">

-   **Aside: the submitter of this image is also in the analytics club @ ETH Zurich, which the bot knew to reference.**


### examples of command-line interaction with "general" conversation bot

The following responses were received for general conversational questions with the `GPT2_trivNatQAdailydia_774M_175Ksteps` model. This is an example of what is capable (and much more!!) in terms of learning to interact with another person, especially in a different language:

```
C:\Users\peter\PycharmProjects\gpt2_chatbot>python ai_single_response.py --time --model "GPT2_trivNatQAdailydia_774M_175Ksteps" --prompt "where is the grocery store?"

... generating...

finished!

"it's on the highway over there."
took 38.9 seconds to generate.

C:\Users\peter\PycharmProjects\gpt2_chatbot>python ai_single_response.py --time --model "GPT2_trivNatQAdailydia_774M_175Ksteps" --prompt "what should I bring to the party?"

... generating...

finished!

'you need to just go to the station to pick up a bottle.'
took 45.9 seconds to generate.


C:\Users\peter\PycharmProjects\gpt2_chatbot>python ai_single_response.py --time --model "GPT2_trivNatQAdailydia_774M_175Ksteps" --prompt "can we be friends?"

... generating...

finished!

"sure, let's go."
took 46.6 seconds to generate.

C:\Users\peter\PycharmProjects\gpt2_chatbot>python ai_single_response.py --time --model "GPT2_trivNatQAdailydia_774M_175Ksteps" --prompt "are you free on tuesday?"

... generating...

finished!

"what's the date today?"
took 41.8 seconds to generate.


C:\Users\peter\PycharmProjects\gpt2_chatbot>python ai_single_response.py --time --model "GPT2_trivNatQAdailydia_774M_175Ksteps" --prompt "do you like your job?"

... generating...

finished!

'no, not very much.'
took 50.1 seconds to generate.
```

### Other resources

These are probably worth checking out if you find you like NLP/transformer-style language modeling:
