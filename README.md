# AI Chatbots based on GPT Architecture

> It sure seems like there are a lot of text-generation chatbots out there, but it's hard to find a python package or model that is easy to tune around a simple text file of message data. This repo is a simple attempt to help solve that problem.

![ACE-newsletter-workshop-img-cropped](https://user-images.githubusercontent.com/74869040/141669429-6bfd3e3f-2436-482b-b7b3-367bee6c23d3.png)

`ai-msgbot` covers the practical use case of building a chatbot that sounds like you (or some dataset / persona you choose) by training a text-generation model to generate conversation in a consistent structure. This structure is then leveraged to deploy a chatbot that is a "free-form" model that *consistently* replies like a human.

 There are three primary components to this project:

1. parsing a dataset of conversation-like data
2. training a text-generation model. This repo is designed around using Google Colab environment for model training.
3. deploying the model to a chatbot interface for users to interact with, either locally or on a cloud service.

It relies on the [`aitextgen`](https://github.com/minimaxir/aitextgen) and [`python-telegram-bot`](https://github.com/python-telegram-bot/python-telegram-bot) libraries. Examples on how to train larger models with DeepSpeed are in `notebooks/colab-huggingface-API` directory.

```sh
python ai_single_response.py --model "GPT2_conversational_355M_WoW10k" --prompt "hey, what's up?"

... generating...

finished!
"i'm doing a lot of traveling. do you like to travel?"
```

Some of the trained models can be interacted with through the HuggingFace spaces and model inference APIs on the [ETHZ Analytics Organization](https://huggingface.co/ethzanalytics) page on huggingface.co.

* * *

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

* * *

## Quick outline of repo

- training and message EDA notebooks in `notebooks/`
- python scripts for parsing message data into a standard format for training GPT are in `parsing-messages/`
- example data (from the *Daily Dialogues* dataset) is in `conversation-data/`
- Usage of default models is available via the `dowload_models.py` script

### Example

This response is from a [bot on Telegram](https://t.me/GPTPeter_bot), finetuned on the author's messages

<img src="https://i.imgur.com/OJB5EMw.png" max-width="50%" />

The model card for this can be found [here](https://huggingface.co/pszemraj/opt-peter-2.7B).

## Quickstart

> **NOTE: to build all the requirements, you *may* need Microsoft C++ Build Tools, found [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)**

1. clone the repo
2. cd into the repo directory: `cd ai-msgbot/`
3. Install the requirements: `pip install -r requirements.txt`
    - if using conda: `conda env create --file environment.yml`
    - *NOTE:* if any errors with the conda install, it may ask for an environment name which is `msgbot`
4. download the models: `python download_models.py` *(if you have a GPT-2 model, save the model to the working directory and you can skip this step)*
5. run the bot: `python ai_single_response.py -p "hey, what's up?"` or enter a "chatroom" with `python conv_w_ai.py -p "hey, what's up?"`
    - *Note:* for either of the above, the `-h` parameter can be passed to see the options (or just look in the script file)

Put together in a shell block:

```sh
git clone https://github.com/pszemraj/ai-msgbot.git
cd ai-msgbot/
pip install -r requirements.txt
python download_models.py
python ai_single_response.py -p "hey, what's up?"
```

* * *

## Repo Overview and Usage

![process flow](https://user-images.githubusercontent.com/74869040/144432206-f584d1c1-a890-4a56-8439-ac7542c7af8c.png)

### New to Colab?

`aitextgen` is largely designed around leveraging Colab's free-GPU capabilities to train models. Training a text generation model, and most transformer models, *is resource intensive*. If new to the Google Colab environment, check out the below to understand more of what it is and how it works.

- [Google's FAQ](https://research.google.com/colaboratory/faq.html)
- [Medium Article on Colab + Large Datasets](https://satyajitghana.medium.com/working-with-huge-datasets-800k-files-in-google-colab-and-google-drive-bcb175c79477)
- [Google's Demo Notebook on I/O](https://colab.research.google.com/notebooks/io.ipynb)
- [A better Colab Experience](https://towardsdatascience.com/10-tips-for-a-better-google-colab-experience-33f8fe721b82)

### Parsing Data

- the first step in understanding what is going on here is to understand ultimately what is happening is teaching GPT-2 to recognize a "script" of messages and respond as such.
  - this is done with the `aitextgen` library, and it's recommended to read through [some of their docs](https://docs.aitextgen.io/tutorials/colab/) and take a look at the *Training your model* section before returning here.
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

    ..._Continued_...

more to come, but check out `parsing-messages/parse_whatsapp_output.py` for a script that will parse messages exported with the standard [whatsapp chat export feature](https://faq.whatsapp.com/196737011380816/?locale=en_US#:~:text=You%20can%20use%20the%20export,with%20media%20or%20without%20media.). consolidate all the whatsapp message export folders into a root directory, and pass the root directory to this

<font color="yellow"> TODO: more words </font>

### Training a text generation model

The next step is to leverage the text-generative model to reply to messages. This is done by "behind the scenes" parsing/presenting the query with either a real or artificial speaker name, and having the response be from `target_name` and in the case of GPT-Peter, is me.

Depending on compute resources and so forth, it is possible to keep track of the conversation in a helper script/loop, and then feed in the prior conversation and *then* the prompt so the model can use the context as part of the generation sequence, with of course the [attention mechanism](https://arxiv.org/abs/1706.03762) ultimately focusing on the last text past to it (the actual prompt)

Then, it's a matter of deploying this pipeline, to an endpoint where a user can send in a message, and the model will respond with a response. There are several options provided in this repo, see the `deploy-as-bot/` directory which has an associated README.md file.

#### Training: Details

- an example dataset (*Daily Dialogues*) parsed into the script format can be found locally in the `conversation-data` directory.
  - When learning, it is probably best to use a conversational dataset such as *Daily Dialogues* as the last dataset to finetune the GPT2-model, but before that, the model can "learn" various pieces of information from something like a natural questions focused dataset.
  - many more datasets are available online at [PapersWithCode](https://paperswithcode.com/datasets?task=dialogue-generation&mod=texts) and [GoogleResearch](https://research.google/tools/datasets/). Seems that *Google Research* also has a tool for searching for datasets online.
- Note that training is done in google colab itself. try opening `notebooks/colab-notebooks/GPT_general_conv_textgen_training_GPU.ipynb` in Google Colab (see the HTML button at the top of that notebook or click [this link to a shared git gist](https://colab.research.google.com/gist/pszemraj/06a95c7801b7b95e387eafdeac6594e7/gpt2-general-conv-textgen-training-gpu.ipynb))
- Essentially, a script needs to be parsed and loaded into the notebook as a standard .txt file with formatting as outlined above. Then, the text-generation model will load and train using *aitextgen's* wrapper around the PyTorch lightning trainer. Essentially, text is fed into the model, and it self-evaluates for "test" as to whether a text message chain (somewhere later in the doc) was correctly predicted or not.

<font color="yellow"> TODO: more words </font>

### Interaction with a Trained model

- Command line scripts:
  - `python ai_single_response.py -p "hey, what's up?"`
  - `python conv_w_ai.py -p "hey, what's up?"`
  - You can pass the argument `--model <NAME OF LOCAL MODEL DIR>`, to change the model.
  - Example: `python conv_w_ai.py -p "hey, what's up?" --model "GPT2_trivNatQAdailydia_774M_175Ksteps"`
- Some demos are available on the ETHZ Analytics Group's huggingface.co page (*no code required!*):
  - [basic chatbot](<https://huggingface.co/spaces/ethzanalytics/dialogue-demo>)
  - [GPT-2 XL Conversational Chatbot](<https://huggingface.co/spaces/ethzanalytics/dialogue-demo>)
- Gradio - locally hosted runtime with public URL.
  - See: `deploy-as-bot/gradio_chatbot.py`
  - The UI and interface will look similar to the demos above, but run locally & are more customizable.
- Telegram bot - Runs locally, and anyone can message the model from the Telegram messenger app(s).
  - See: `deploy-as-bot/telegram_bot.py`
  - An example chatbot by one of the authors is usually online and can be found [here](https://t.me/GPTPeter_bot)

### Improving Response Quality: Spelling & Grammar Correction

One of the primary goals of this project is to be able to train a chatbot/QA bot that can respond to the user "unaided" where it does not need hardcoding to be able to handle questions / edge cases. That said, sometimes the model will generate a bunch of strings together,, and applying "general" spell correction helps keep the model responses as comprehensible as possible without interfering with the response / semantics itself.

- Implemented methods:
  - **symspell** (via the pysymspell library) _NOTE: while this is fast and works, it sometimes corrects out common text abbreviations to random other short words that are hard to understand, i.e. **tues** and **idk** and so forth_
  - **gramformer** (via transformers `pipeline()`object). a pretrained NN that corrects grammar and (to be tested) hopefully does not have the issue described above. Links: [model page](https://huggingface.co/prithivida/grammar_error_correcter_v1), [the models github](https://github.com/PrithivirajDamodaran/Gramformer/)
- **Grammar Synthesis** (WIP) - Some promising results come from training a text2text generation model that through "pseudo-diffusion" is trained to denoise **heavily** corrupted text, while learning to *not* change the semantics of the text. A checkpoint and more details can be found [here](https://huggingface.co/pszemraj/grammar-synthesis-base) and a notebook [here](https://colab.research.google.com/gist/pszemraj/91abb08aa99a14d9fdc59e851e8aed66/demo-for-grammar-synthesis-base.ipynb).

---

## WIP: Tasks & Ideas

[x] finish out `conv_w_ai.py` that is capable of being fed a whole conversation (or at least, the last several messages) to prime response and "remember" things.
[ ] better text generation

- add-in option of generating multiple responses to user prompt and automatically applying sentence scoring to them and returning the one with the highest mean sentence score.
- constrained textgen
  - [x] explore constrained textgen
  - [ ] add constrained textgen to repo
[x] assess generalization of hyperparameters for "text-message-esque" bots
- [ ] add write-up with hyperparameter optimization results/learnings

[ ] switch repo API from `aitextgen` to `transformers pipeline` object

[ ] Explore model size in relation to "human-ness"

## Extras, Asides, and Examples

### examples of command-line interaction with "general" conversation bot

The following responses were received for general conversational questions with the `GPT2_trivNatQAdailydia_774M_175Ksteps` model. This is an example of what is capable (and much more!!) in terms of learning to interact with another person, especially in a different language:

    python ai_single_response.py --time --model "GPT2_trivNatQAdailydia_774M_175Ksteps" --prompt "where is the grocery store?"

    ... generating...

    finished!

    "it's on the highway over there."
    took 38.9 seconds to generate.

    python ai_single_response.py --time --model "GPT2_trivNatQAdailydia_774M_175Ksteps" --prompt "what should I bring to the party?"

    ... generating...

    finished!

    'you need to just go to the station to pick up a bottle.'
    took 45.9 seconds to generate.

    C:\Users\peter\PycharmProjects\gpt2_chatbot>python ai_single_response.py --time --model "GPT2_trivNatQAdailydia_774M_175Ksteps" --prompt "do you like your job?"

    ... generating...

    finished!

    'no, not very much.'
    took 50.1 seconds to generate.

### Other resources

These are probably worth checking out if you find you like NLP/transformer-style language modeling:

1. [The Huggingface Transformer and NLP course](https://huggingface.co/course/chapter1/2?fw=pt)
2. [Practical Deep Learning for Coders](https://course.fast.ai/) from fast.ai

---

## Citations

TODO: add citations for datasets, main packages used.
