import sys
import os
import json
from typing import Optional, Tuple
import requests

# Allow imports after sys.path modifications
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rgwfuncs.str_lib import send_telegram_message


# Telegram credentials
BOT_TOKEN = "7450388541:AAFvy4rpcF3e6gDb1NqDwo0WG7Q_ixdtLnQ"
CHAT_ID = "7214830293"

send_telegram_message("wow", bot_token=BOT_TOKEN, chat_id=CHAT_ID)
send_telegram_message("ahem", bot_token=BOT_TOKEN, chat_id=CHAT_ID)
