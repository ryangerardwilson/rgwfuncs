import sys
import os
import json
from typing import Optional, Tuple
import requests

# Allow imports after sys.path modifications
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rgwfuncs.str_lib import send_telegram_message


# Telegram credentials
BOT_TOKEN = ""
CHAT_ID = ""

send_telegram_message("wow", bot_token=BOT_TOKEN, chat_id=CHAT_ID)
send_telegram_message("ahem", bot_token=BOT_TOKEN, chat_id=CHAT_ID)
