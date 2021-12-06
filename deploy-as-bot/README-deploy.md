# deployment

Two methods of deployment are explored in this repo. A telegram bot, which has the downside of the user getting telegram (because not everyone has / wants it), and gradio, which is easier to access, but if disrupted the link changes and users would need to be informed.

_more deployment ideas / options to come_

## gradio

- gradio allows you to deploy the model from your local computer and users essentially just need to open the link you send them in their browser
- find out more [here](https://www.gradio.app/getting_started)

## telegram

- Deploy the bot to telegram using the `python-telegram-bot` package
- You can message an example bot by clicking [this link](https://t.me/GPTfriend_bot). _Please note that at present this bot is run locally on a machine, and may not be online 24/7._

---

## Resources / Links

### telegram

- python-telegram-bot [quickstart](https://github.com/python-telegram-bot/python-telegram-bot/wiki/Extensions-%E2%80%93-Your-first-Bot)
- python-telegram-bot [docs](https://python-telegram-bot.readthedocs.io/en/stable/telegram.html)
- [How to create a Telegram Bot in Python in under 10 minutes - codementor](https://www.codementor.io/@karandeepbatra/part-1-how-to-create-a-telegram-bot-in-python-in-under-10-minutes-19yfdv4wrq)
- An [article](https://www.section.io/engineering-education/building-a-telegram-bot-with-python-to-generate-quotes/) from 2021 that covers creating Telegram API token, bot, etc in Python.

### gradio

- quickstart [here](https://www.gradio.app/getting_started)
- [documentation page](https://gradio.app/docs)

### permanent gradio 
https://abidlabs.medium.com/quickly-deploying-gradio-on-aws-242af2374784

### hosting on huggingface spaces
https://huggingface.co/docs/hub/spaces

### preparing VPS
- ssh root@
- swapon --show 
- free -h 
- df -h 
- fallocate -l 38G /swapfile 
- ls -lh /swapfile 
- chmod 600 /swapfile 
- ls -lh /swapfile 
- mkswap /swapfile 
- swapon /swapfile 
- swapon --show 
- free -h 
- cp /etc/fstab /etc/fstab.bak 
- echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
- git clone git clone https://github.com/pszemraj/ai-msgbot
- cd ai-msgbot
- apt update
- apt install pipenv
- pipenv shell
- pip install -r requirements.txt
- cd deploy-as-bot
- scp -r ../GPTneo_conv_33kWoW_18kDD root@ :/root/ai-msgbot
- nohup python gradio_chatbot_DO.py --model GPTneo_conv_33kWoW_18kDD &
