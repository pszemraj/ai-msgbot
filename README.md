# AI Chatbots based on GPT Architecture

![ACE-newsletter-workshop-img-cropped](https://user-images.githubusercontent.com/74869040/141669429-6bfd3e3f-2436-482b-b7b3-367bee6c23d3.png)

This repo covers the practical use case of building a chatbot that sounds like you (or some dataset / persona you choose) by training a GPT-based model (either GPT2 or GPT-neo). Primarily relies on the [`aitextgen`](https://github.com/minimaxir/aitextgen) and [`python-telegram-bot`](https://github.com/python-telegram-bot/python-telegram-bot) libraries.

> **Note** that most model training was done on Colab to leverage the GPU.

```bazaar
$ python ai_single_response.py --prompt "how can I order food?" --temp 0.7

... generating...

finished!

'what kind of food do you want?'
```

- you can message an example bot by clicking [this link](https://t.me/GPTfriend_bot). Please note that at present this bot is run locally on a machine, and may not be online 24/7.

---

**Table of Contents**

<!-- TOC -->

- [AI Chatbots based on GPT Architecture](#ai-chatbots-based-on-gpt-architecture)
  - [Quick outline of repo:](#quick-outline-of-repo)
  - [quickstart](#quickstart)
  - [Repo Overview and Usage](#repo-overview-and-usage)
    - [New to Colab?](#new-to-colab)
    - [Training a model](#training-a-model)
      - [Training: Details](#training-details)
    - [Parsing Message Data](#parsing-message-data)
    - [Interaction with a Trained model](#interaction-with-a-trained-model)
      - [Model Responses: Spelling / Grammar Correction](#model-responses-spelling--grammar-correction)
  - [Work and Idea Lists](#work-and-idea-lists)
    - [worklist](#worklist)
    - [idea list](#idea-list)
  - [Extras, Asides, and Examples](#extras-asides-and-examples)
    - [examples of command-line interaction with "general" conversation bot](#examples-of-command-line-interaction-with-general-conversation-bot)
    - [Other resources](#other-resources)
  - [Citations](#citations)

<!-- /TOC -->

---

## Quick outline of repo:

- training and message EDA notebooks in `colab-notebooks`
- python scripts for parsing message data into a standard format for training GPT are in `parsing-messages`
- example data (from the _Daily Dialogues_ dataset) is in `conversation-data`
- model files need to be downloaded from a hosted dropbox due to filesize (run `download_models.py`) and it will be done "automatically". They will then show up in folders in your cwd

**A friend's screenshot: an example bot response on Telegram**

<img src="https://user-images.githubusercontent.com/74869040/138378871-d3508ce8-8dd0-45b8-92e2-92bc5ae1d530.jpg" width="600" height="200">

_Note: the bot in the image of question was trained on the author's text message data, who sometimes has a problem showing up on time for events._

## quickstart

- clone the repo
- with a terminal opened in the repo folder:

  - `pip install -r requirements.txt` if using conda: `conda env create --file environment.yml`
    - _NOTE:_ if any errors with the conda install, it may ask for an environment name which is `gpt2_chatbot`
    - As the conda environment has _everything_ you may need to install C++ development packages via Visual Studio Code installer.
  - `python download_models.py`
  - `python .\ai_single_response.py --prompt "do you know how to get rich?"`
  - `MacOS: python ./ai_single_response.py --prompt "do you know how to get rich?"`
  - _Note:_ for either of the above, the `-h` parameter can be passed to see the options (or just look in the script file)

- then it will respond!
- other models are available / will be downloaded, to change the model that generates a response you can pass the argument `--model` for example:

  `python ai_single_response.py --model "GPT2_dailydialogue_355M_150Ksteps" --prompt "are you free tomorrow?"`

---

## Repo Overview and Usage

### New to Colab?

`aitextgen` is largely designed around leveraging Colab's free-GPU capabilities to train models. Training a text generation model, and most transformer models, _is resource intensive_. If new to the Google Colab environment, check out the below to understand more of what it is and how it works.

- [Google's FAQ](https://research.google.com/colaboratory/faq.html)
- [Medium Article on Colab + Large Datasets](https://satyajitghana.medium.com/working-with-huge-datasets-800k-files-in-google-colab-and-google-drive-bcb175c79477)
- [Google's Demo Notebook on I/O](https://colab.research.google.com/notebooks/io.ipynb)
- [A better Colab Experience](https://towardsdatascience.com/10-tips-for-a-better-google-colab-experience-33f8fe721b82)

### Training a model

- the first step in understanding what is going on here is to understand ultimately what is happening is teaching GPT-2 to recognize a "script" of messages and respond as such.
  - this is done with the `aitextgen` library, and it's recommended to read through [some of their docs](https://docs.aitextgen.io/tutorials/colab/) and take a look at the _Training your model_ section before returning here.
- essentially, to generate a novel chatbot from just text (without going through too much trouble as required in other libraries.. can you abstract your friends whatsapp messages easily into a "persona"?)

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

- then, you leverage the text-generative model to reply to messages. This is done by "behind the scenes" parsing/presenting the query with either a real or artificial speaker name, and having the response be from `target_name` and in the case of GPT-Peter, is me.
- depending on compute resources and so forth, it is possible to keep track of the conversation in a helper script/loop, and then feed in the prior conversation and _then_ the prompt so the model can use the context as part of the generation sequence, with of course the [attention mechanism](https://arxiv.org/abs/1706.03762) ultimately focusing on the last text past to it (the actual prompt)
- then, a matter of deploying it, whether it is a bot that can help children learn conversation, local lingo, etc in a foreign language or a whatsapp bot to automate social upkeep, the possibilities are endless.
-

#### Training: Details

- an example dataset (_Daily Dialogues_) parsed into the script format can be found locally in the `conversation-data` directory.
  - When learning, it is probably best to use a conversational dataset such as _Daily Dialogues_ as the last dataset to finetune the GPT2-model, but before that, the model can "learn" various pieces of information from something like a natural questions focused dataset.
  - many more datasets are available online at [PapersWithCode](https://paperswithcode.com/datasets?task=dialogue-generation&mod=texts) and [GoogleResearch](https://research.google/tools/datasets/). Seems that _Google Research_ also has a tool for searching for datasets online.
- Note that training is done in google colab itself. try opening `notebooks\colab-notebooks\GPT_general_conv_textgen_training_GPU.ipynb` in Google Colab (see the HTML button at the top of that notebook or click [this link to a shared git gist](https://colab.research.google.com/gist/pszemraj/06a95c7801b7b95e387eafdeac6594e7/gpt2-general-conv-textgen-training-gpu.ipynb))
- Essentially, a script needs to be parsed and loaded into the notebook as a standard .txt file with formatting as outlined above. Then, the text-generation model will load and train using _aitextgen's_ wrapper around the PyTorch lightning trainer. Essentially, text is fed into the model, and it self-evaluates for "test" as to whether a text message chain (somewhere later in the doc) was correctly predicted or not.

<font color="yellow"> TODO: more words </font>

### Parsing Message Data

- more to come, but check out `parsing-messages\parse_whatsapp_output.py` for a script that will parse messages exported with the standard whatsapp chat -> export feature. consolidate all the whatsapp message export folders into a root directory, and pass the root directory to this

<font color="yellow"> TODO: more words </font>

### Interaction with a Trained model

- command line
- general deployment via Gradio
- "bot mode"
  - telegram
  - whatsapp (still mostly unexplored)
- _real deployment_ @ jonathan lehner\_

<font color="yellow"> TODO: more words </font>

#### Model Responses: Spelling / Grammar Correction

- obviously, one of the primary goals of this project is to be able to train a chatbot/QA bot that can respond to the user "unaided" where it does not need hardcoding to be able to handle questions / edge cases. That said, sometimes the model will generate a bunch of strings together,, and applying "general" spell correction helps keep the model responses as comprehensible as possible without interfering with the response / semantics itself.
- two methods of doing this are currently added:
  - **symspell** (via the pysymspell library) _NOTE: while this is fast and works, it sometimes corrects out common text abbreviations to random other short words that are hard to understand, i.e. **tues** and **idk** and so forth_
  - **gramformer** (via transformers `pipeline()`object). a pretrained NN that corrects grammar and (to be tested) hopefully does not have the issue described above. Links: [model page](https://huggingface.co/prithivida/grammar_error_correcter_v1), [the models github](https://github.com/PrithivirajDamodaran/Gramformer/) (\_note: not using this because it isnt a pypy package, so i just use the hosted model on huggingface), [hf docs on pipelines() object](https://huggingface.co/transformers/main_classes/pipelines.html?highlight=textgeneration)

## Work and Idea Lists

_What we plan to add to this repo in the foreseeable future._

### worklist

1.  finish out `conv_w_ai.py` that is capable of being fed a whole conversation (or at least, the last several messages) to prime response and "remember" things.
2.  add-in option of generating multiple responses to user prompt and automatically applying sentence scoring to them and returning the one with the highest mean sentence score.
3.  assess generalization of hyperparameters for "text-message-esque" bots
4.  provide more parsed datasets to be used for training models

### idea list

1.  try generating 5-10 responses at once instead of n=1, and return the one with the highest [harmonic mean sentence score](https://github.com/simonepri/lm-scorer). **IN PROGRESS**

- > **Rationale**: based on _UNVALIDATED AND UNQUANTIFIED_ trends in the grid search data (see [gridsearch v1](https://www.dropbox.com/s/uanhf2kuyoybo4x/GPT-Peter%20Hyperparam%20Analysis%20w%20Metrics%20-%20Oct-20-2021_15-49.xlsx?dl=0) and [gridsearch v2](https://www.dropbox.com/s/r2xv66wdfyalwyi/GPT-Peter%20Hyperparam%20Analysis%20w%20Metrics%20-%20Oct-21-2021_02-01.xlsx?dl=0)), the responses that rank high on the harmonic mean score also seem the most coherent and responsive to the question at hand \*this is anecdotal
- > jury is still out as to what the intuition / reason behind that is. The _product score_ results being useful makes sense, but these are even better
- > therefore, generating 5-10 reponses at once, scoring them all at once (_check docs for function_) and returning the corresponding highest-scoring prompt should have the bot behaving more realistically.

2.  continue with hyperparamter optimization on fine-tuned models. Status of hyperparameter "search" is kept (_and will be updated_) [here](https://ai-msgbot-gptneo-1pt3b.netlify.app/) for a **general** chatbot that is a fine-tuned version of GPT-Neo 1.3B. Data related to hyperparamter optimization for GPT-Peter (on personal whatsapp messages ) will be further made available if useful. **IN PROGRESS**

- > examine if any basic ML approaches can model the harmonic/geometric mean response scores with [Pycaret](http://www.pycaret.org/tutorials/html/REG102.html)

3.  investigate whatsapp bot potential and utility
4.  ~~evaluate if pretrained on the _Daily Dialogues_ data set and then training for other purposes helps with the "transfer learning" of teaching the GPT model that it is now a chatbot vs. just directly training the "standard" checkpoint~~

- > ~~in short, `355M checkpoint -> daily dialogues -> message data` vs. `355M checkpoint -> message data`~~
- > **yes, it does improve things a lot\_ TODO: writeup theory**

5.  ~~evaluate whether pretraining on other datasets, such as [CoQA (Conversational Question Answering Challenge)](https://paperswithcode.com/dataset/coqa) or [TriviaQA](https://paperswithcode.com/dataset/triviaqa) improves transfer learning to being a chatbot~~

    - > ~~this applies for a text message chat bot _and_ also the "resources for learning english in a safer environment" bot~~
    - > **using Trivia/CoCaQA did help model responses**

6.  ~~try gradio deployment~~
    - _implemented_
7.  try huggingface spaces deployment
8.  Auto_ML based approach to see if multi dimensional hyperparameter

---

## Extras, Asides, and Examples

### examples of command-line interaction with "general" conversation bot

The following responses were received for general conversational questions with the `GPT2_trivNatQAdailydia_774M_175Ksteps` model. This is an example of what is capable (and much more!!) in terms of learning to interact with another person, especially in a different language:

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

### Other resources

These are probably worth checking out if you find you like NLP/transformer-style language modeling:

1.  [The Huggingface Transformer and NLP course](https://huggingface.co/course/chapter1/2?fw=pt)
2.  [Practical Deep Learning for Coders](https://course.fast.ai/) from fast.ai

## Citations

TODO: add citations for datasets, main packages used.
