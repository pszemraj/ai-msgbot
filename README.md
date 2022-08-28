# AI Chatbots based on GPT Architecture

> It seems like there are a lot of text-generation chatbots out there, but it's hard to find a framework easy to tune around a basic set of messages. `ai-msgbot` is an attempt to help solve that problem.

<img src="https://i.imgur.com/aPxUkH7.png" width="384" height="256"/>

`ai-msgbot` is designed to help you build a chatbot that sounds like you (_or any custom dialogue data_) by training a text-generation model to generate conversation in a consistent structure. This structure is then leveraged to deploy a chatbot based on a trained "barebones" model that replies **consistently** like a human might.

 There are three primary components to this project:

1. Parse conversation-like data into a simple, formatted dialogue script.
2. Train a text-generation model to learn and generate conversation in the script's structure.
3. Deploy the model as a chatbot interface, locally or on a cloud service.

This repo relies on the [`aitextgen`](https://github.com/minimaxir/aitextgen) and [`python-telegram-bot`](https://github.com/python-telegram-bot/python-telegram-bot) packages.

```sh
python ai_single_response.py -p "greetings sir! what is up?"

... generating...

finished!
('hello, i am interested in the history of vegetarianism. i do not like the '
 'idea of eating meat out of respect for sentient life')
```

* * *

**Table of Contents**

<!-- TOC -->

- [Quick outline of repo](#quick-outline-of-repo)
- [Quickstart](#quickstart)
- [Repo Overview and Usage](#repo-overview-and-usage)
  - [A general process-flow](#a-general-process-flow)
  - [Consistent Dialogue Structure](#consistent-dialogue-structure)
  - [Training a text generation model](#training-a-text-generation-model)
    - [Training: Details](#training-details)
    - [New to Colab?](#new-to-colab)
  - [Interaction with a Trained model](#interaction-with-a-trained-model)
  - [Improving Response Quality: Spelling & Grammar Correction](#improving-response-quality-spelling--grammar-correction)
- [WIP: Tasks & Ideas](#wip-tasks--ideas)
- [Extras, Asides, and Examples](#extras-asides-and-examples)
- [Citations](#citations)

<!-- /TOC -->

* * *

## Quick outline of repo

- training and message EDA notebooks in `notebooks/`
- python scripts for parsing message data into a standard format for training are in `parsing-messages/`
- pre-processed example _dialogue_scripts_ is in `conversation-data/`
- Download and test some models right away with the `dowload_models.py` script

### Example

This response is from a [bot on Telegram](https://t.me/GPTPeter_bot), finetuned on the author's messages

<img src="https://i.imgur.com/OJB5EMw.png" width="550" height="150" />

The model card can be found [here](https://huggingface.co/pszemraj/opt-peter-2.7B).

## Quickstart

> **NOTE: to build all the requirements, you _may_ need Microsoft C++ Build Tools, found [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)**

1. clone the repo
2. cd into the repo directory: `cd ai-msgbot/`
3. Install the requirements: `pip install -r requirements.txt`
    - if using conda: `conda env create --file environment.yml`
    - _NOTE:_ If you get any error referring to the "name" you may want to use the default `--name`, ``msgbot
4. download the models: `python download_models.py` _(if you have a GPT-2 model, save the model to the working directory, and you can skip this step)_
5. run the bot: `python ai_single_response.py -p "hey, what's up?"` or enter a "chatroom" with `python conv_w_ai.py -p "hey, what's up?"`
    - _Note:_ pass `-h` for help & options

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

### A general process-flow

<img src="https://i.imgur.com/9iUEzvV.png" mwidth="872" height="400" />

This project aims to generate a chatbot from simple text files using the GPT family of models. This easy-to-create structure helps avoid spending too much time on dataset curation and associated questions/trouble (_Can you easily abstract you and your friend's WhatsApp messages into a "persona"?_).

Three steps: **1)** acquire & format data **2)** train model **3)** deploy model with "helper" code to implement as a chatbot.

### Consistent Dialogue Structure

The first step is to find and parse the data into a standard format to train the model. The _"why"_ and the _"how"_ are somewhat intertwined, so the process is described in detail below.

#### What?

The goal is to take a conversation-like dataset and parse it into a "script" that can be used for training the model.

What a "dialogue script" consists of:

- Simple, repeated formatting for each message or "utterance" in the conversation
  - every **message** is a line in the file. A message can contain multiple sentences (dataset dependent)
  - every message is preceded by a line with the **speaker pseudo-label**
  - every message is followed by an empty blank line (_this is somewhat arbitrary_)
  - messages _by "conversation"_ are in chronological order. Conversations themselves. The order of conversations does not matter.
- text data in this format is fed when training the model. It learns to predict the next N tokens (~words) based on the previous N tokens and starts to understand the structure of the conversation (response types, emphasis, etc.)
- Text data in the same format is used when `predict`-ing text as a chatbot as well. We know the expected output format and can use this knowledge to extract a single message response to arbitrary input.

_Example Dialogue Script_:

    Person Alpha:
    So what about the tennis racket?

    Person Beta:
    Look! It's amazing. I can't wait to try it out!

    Person Alpha:
    How much did that end up costing you?

    Person Beta:
    Oh... around twenty bucks. A bargain if you ask me. Look at the picture of her playing with it!

    Person Alpha:
    Hey, two for one. That's a super deal.

    (...)

#### Why?

In a nutshell, to leverage the strengths of the GPT family of models while using the script structure and domain knowledge to compensate for tasks not well-handled by pure text generation.

"Chatbots" exist today and are used in many contexts. You can find examples of them on your local massive service-oriented corporation's website. These chatbots are typically scripted around a few likely inputs and desired answers for the use case. They fail, however, at generating "real" responses to arbitrary inputs, which makes them seem limited.

> Ask the next chatbot on the web: "A further question of your preferences: straight booling vs. dimensional booling?"

- GPT excels at generating **text that sounds like a real conversation**, can be adapted to new domains, and can generate coherent new responses even with the same input.
- A GPT model trained in a generalizable way could be useful to humans without needing specific scripted answers.
- Some caveats:
  - how would a model know how to respond to a given input?
  - Further, how would the model tailor its response to a given speaker?
  - how long is a single "response"? How would a model know when to stop?
  - how would a model know to stay consistently on topic/with the speaker?
- Separating the conversational messages with "tokens/labels" helps solve the above concerns.
  - The model learns to generate these **structure labels** as part of the training process and generates them as part of inference.
  - Isolating a single response becomes easy:
    - train with "Speaker Alpha" and "Speaker Beta" labels
    - pass "Speaker Alpha" and the user prompt to the model
    - append a blank line and "Speaker Beta" label to the prompt
    - model responds as "Speaker Beta" to the "Speaker Alpha" prompt!
    - isolate the "full" response by stopping at 1) the first blank line or 2) at the first `Speaker Alpha` token.
  - the model implicitly learns the back-and-forth structure of the conversation by "seeing" how text following _Speaker Alpha_ is responded to by text with the other label  _Speaker Beta_, and vice-versa. **It learns interactions between the two speakers**.
    - this is especially important in today's messaging world, where a response may be distributed over several messages. Generated messages with labels enable aggregating multiple messages into a single response to a prompt. #TODO get the source for this
    - Multiple individuals or "personas" are possible by adding unique labels to the training data, i.e., "Speaker Gamma" and "Speaker Delta".
  - part of "reading the room" in the conversational domain is to simulate empathy, even in cases where a perfect scripted response is either impossible or does not exist.
    - For example, there are many ways to respond to "My dog died.." that would be contextually correct. See [example](https://pastebin.com/XipcF71F)

While these improvements are promising, some use cases are far from perfect. Recalling factual information is especially tricky, so avoiding factually dependent use cases (_like robot museum tour guides_) is recommended.

#### How?

The minimum is to create a `training_data.txt` file in the above format. Ideally, create all three of the train/valid/test files to validate the model generalization to unseen data.

Examples and resources:

- Code to parse WhatsApp messages and exported iMessage conversations can be found in `parsing-messages/`.
- Already-parsed datasets can be found in `conversation-data/`.
- A companion repository at [pszemraj/DailyDialogue-Parser](https://github.com/pszemraj/DailyDialogue-Parser) covers the entire process of converting/parsing the _DailyDialogues_ dataset.

Comments on finding datasets are below in [Training Details](#training-details).

#### A Chatbot of You

Almost everyone with a phone has a huge message dataset of their own. This dataset could be used to train a "_Chatbot of You_" to train a chatbot to respond to your messages as you would.

See `parsing-messages/parse_whatsapp_output.py` for code to parse messages exported with the [whatsapp chat export feature](https://faq.whatsapp.com/196737011380816/?locale=en_US#:~:text=You%20can%20use%20the%20export,with%20media%20or%20without%20media.). consolidate all the message export folders into a root directory, and pass the root directory to `parse_whatsapp_output.py`.

### Training a text generation model

> The `aitextgen` package is used to train the model (_in the current_state of the repo_). It's recommended to read through [some of the docs](https://docs.aitextgen.io/tutorials/colab/) (see _Training your model_ section) before continuing.

The next step is to teach the textgen model to "reply" to messages in the same way as your dataset. This is done by training the model to generate text based on previously seen text, called Causal Language Model modeling. For more detailed info, see [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) or [the huggingface course](https://huggingface.co/course/chapter1/6?fw=pt).

#### Training: Details

The primary training notebook is found at: `*/notebooks/colab-notebooks/GPT_general_conv_textgen_training_GPU.ipynb` (Alternatively, click [this link to a shared git gist](https://colab.research.google.com/gist/pszemraj/06a95c7801b7b95e387eafdeac6594e7/gpt2-general-conv-textgen-training-gpu.ipynb)).

- The notebook covers training GPT-2 models up to GPT-2 L.
- high-level overview:
  - A _dialogue script_ (parsed from the dataset)), formatted as per the above structure, is loaded and tokenized.
  - Then, the next-generation model will load and train using _aitextgen's_ wrapper around the PyTorch lightning trainer.
  - Essentially, the text is fed into the model, and it self-evaluates by "testing" whether a text message chain (somewhere later in the data) was correctly generated or not, adjusting the model's parameters as necessary.
- more datasets are available at [PapersWithCode](https://paperswithcode.com/datasets?task=dialogue-generation&mod=texts) and [GoogleResearch](https://research.google/tools/datasets/). _Google Research_ also has a tool for searching for datasets.

Examples of how to train larger models with DeepSpeed are in the `notebooks/colab-huggingface-API` directory. These notebooks have been used to tune up to `GPT-Neo-2.7B` and `GPT-J-6B-8bit.`

#### New to Colab?

`aitextgen` (and this repo) leverage Colab's free-GPU capabilities to train models. If new to the Google Colab environment, look at the below to understand more of what/how it works.

- Google's [FAQ](https://research.google.com/colaboratory/faq.html) and [Demo Notebook on I/O](https://colab.research.google.com/notebooks/io.ipynb)
- [A better Colab Experience](https://towardsdatascience.com/10-tips-for-a-better-google-colab-experience-33f8fe721b82)

### Interaction with a Trained model

Finally, it's time to deploy the pipeline so users can send and receive messages from the model. There are several options; see the `deploy-as-bot/` directory, which has an associated `README.md` file.

- Command line scripts:
  - `python ai_single_response.py -p "hey, what's up?"`
  - `python conv_w_ai.py -p "hey, what's up?"`
  - pass the argument `--model <NAME OF LOCAL MODEL DIR>` to change the model.
  - Example: `python conv_w_ai.py -p "hey, what's up?" --model "GPT2_trivNatQAdailydia_774M_175Ksteps"`
- Some demos are available on the ETHZ Analytics Group's huggingface.co page (_no code required!_):
  - [basic chatbot](https://huggingface.co/spaces/ethzanalytics/dialogue-demo)
  - [GPT-2 XL Conversational Chatbot](https://huggingface.co/spaces/ethzanalytics/dialogue-demo)
- Gradio - locally hosted runtime with public URL.
  - See: `deploy-as-bot/gradio_chatbot.py`
  - The UI and interface will look similar to the demos above but run locally & are more customizable.
- Telegram bot - Runs locally, and anyone can message the model from the Telegram messenger app(s).
  - See: `deploy-as-bot/telegram_bot.py`
  - An example chatbot by one of the authors is usually online and can be found [here](https://t.me/GPTPeter_bot)

### Improving Response Quality: Spelling & Grammar Correction

Especially with models sampling from their distribution of possible tokens, occasionally, the output will not be perfect in spelling/Grammar. One way to automatically fix this **without** changing the response's meaning is with spelling and grammar correction:

- Implemented methods are in `deploy-as-bot/`:
  - **symspell** (via the pysymspell library) _corrects out common text abbreviations to random other short words that are hard to understand, i.e., tues and idk._
  - **Gramformer** is a pretrained NN that corrects grammar and seems to avoid the issue mentioned above, | [Github](https://github.com/PrithivirajDamodaran/Gramformer/)
- **Grammar Synthesis** (WIP) - We've been working on a text2text model that, through "pseudo-diffusion," is trained to denoise corrupted text while learning not to change the semantics. Checkpoint and usage [here](https://huggingface.co/pszemraj/grammar-synthesis-base) & notebook [here](https://colab.research.google.com/gist/pszemraj/91abb08aa99a14d9fdc59e851e8aed66/demo-for-grammar-synthesis-base.ipynb).

* * *

## WIP: Tasks & Ideas

- [x] finish out `conv_w_ai.py` capable of being fed a whole conversation (or at least, the last several messages) to prime response and "remember" things.
- [ ] better text generation

- add-in option of generating multiple responses to user prompts, automatically applying sentence scoring to them, and returning the one with the highest mean sentence score.
- constrained textgen
  - [x] explore constrained textgen
  - [ ] add constrained textgen to repo
                [x] assess generalization of hyperparameters for "text-message-esque" bots
- [ ] add write-up with hyperparameter optimization results/learnings

- [ ] switch repo API from `aitextgen` to `transformers pipeline` object
- [ ] Explore model size about "human-ness."

## Extras, Asides, and Examples

### examples of command-line interaction with "general" conversation bot

> Note: these are from December 2021 and are slightly outdated.

Below are some general conversational questions & responses with the `GPT2_trivNatQAdailydia_774M_175Ksteps` model. It is an example of what is capable (and much more):

    python ai_single_response.py --time --model "GPT2_trivNatQAdailydia_774M_175Ksteps" --prompt "where is the grocery store?"

    ... generating...

    finished!

    "it's on the highway over there."
    took 38.9 seconds to generate.

    Python ai_single_response.py --time --model "GPT2_trivNatQAdailydia_774M_175Ksteps" --prompt "what should I bring to the party?"

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

Worth checking out if you find you are interested in NLP/transformer-style language modeling:

1. [The Huggingface Transformer and NLP course](https://huggingface.co/course/chapter1/2?fw=pt)
2. [Practical Deep Learning for Coders](https://course.fast.ai/) from fast.ai

* * *

## Citations

TODO: add citations for datasets and main packages used.
