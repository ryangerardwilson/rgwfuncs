import sys
import time
import os
import json
import requests
from typing import Tuple, Optional, Union, Dict
from collections import defaultdict
import warnings
from pyfiglet import Figlet
from datetime import datetime

# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Module-level variables
_PRINT_HEADING_CURRENT_CALL = 0
_PRINT_SUBHEADING_COUNTS = defaultdict(int)  # Tracks sub-headings per heading
_CURRENT_HEADING_NUMBER = 0


def send_telegram_message(message: str, preset_name: Optional[str] = None, bot_token: Optional[str] = None, 
                        chat_id: Optional[str] = None) -> None:
    """
    Send a Telegram message using either a preset or provided bot token and chat ID.

    Args:
        message (str): The message to send.
        preset_name (Optional[str]): The name of the preset to use for sending the message.
        bot_token (Optional[str]): The Telegram bot token.
        chat_id (Optional[str]): The Telegram chat ID.

    Raises:
        FileNotFoundError: If no '.rgwfuncsrc' file is found when preset_name is used.
        ValueError: If preset_name is used with bot_token/chat_id, neither preset_name nor both bot_token/chat_id are provided,
                    or the config file is empty/invalid.
        RuntimeError: If the preset is not found or necessary preset details are missing.
    """
    def get_config() -> dict:
        """Get configuration by searching for '.rgwfuncsrc' in current directory and upwards."""
        def find_config_file() -> str:
            """Search for '.rgwfuncsrc' in current directory and upwards."""
            current_dir = os.getcwd()
            while True:
                config_path = os.path.join(current_dir, '.rgwfuncsrc')
                if os.path.isfile(config_path):
                    return config_path
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # Reached root directory
                    raise FileNotFoundError(f"No '.rgwfuncsrc' file found in {os.getcwd()} or parent directories")
                current_dir = parent_dir

        config_path = find_config_file()
        with open(config_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if not content.strip():
                raise ValueError(f"Config file {config_path} is empty")
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in config file {config_path}: {e}")

    def get_telegram_preset(config: dict, preset_name: str) -> dict:
        """Get the Telegram preset configuration."""
        presets = config.get("telegram_bot_presets", [])
        for preset in presets:
            if preset.get("name") == preset_name:
                return preset
        return None

    def get_telegram_bot_details(config: dict, preset_name: str) -> Tuple[str, str]:
        """Retrieve the Telegram bot token and chat ID from the preset."""
        preset = get_telegram_preset(config, preset_name)
        if not preset:
            raise RuntimeError(f"Telegram bot preset '{preset_name}' not found in the configuration file")

        bot_token = preset.get("bot_token")
        chat_id = preset.get("chat_id")

        if not bot_token or not chat_id:
            raise RuntimeError(f"Telegram bot token or chat ID for '{preset_name}' not found in the configuration file")

        return bot_token, chat_id

    # Validate input parameters
    if preset_name and (bot_token or chat_id):
        raise ValueError("Cannot specify both preset_name and bot_token/chat_id")
    if not preset_name and (bool(bot_token) != bool(chat_id)):
        raise ValueError("Both bot_token and chat_id must be provided if preset_name is not used")
    if not preset_name and not bot_token and not chat_id:
        raise ValueError("Either preset_name or both bot_token and chat_id must be provided")

    # Get bot token and chat ID
    if preset_name:
        config = get_config()
        bot_token, chat_id = get_telegram_bot_details(config, preset_name)

    # Prepare and send the request
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, json=payload)
    response.raise_for_status()


def title(text: str, font: str = "slant", typing_speed: float = 0.005) -> None:
    """
    Print text as ASCII art with a typewriter effect using the specified font (default: slant),
    indented by 4 spaces. All output, including errors and separators, is printed with the
    typewriter effect within this function.

    Args:
        text (str): The text to convert to ASCII art.
        font (str, optional): The pyfiglet font to use. Defaults to "slant".
        typing_speed (float, optional): Delay between printing each character in seconds.
                                      Defaults to 0.005.

    Raises:
        ValueError: If the specified font is invalid or unavailable.
        RuntimeError: If there is an error generating the ASCII art.
    """
    # ANSI color codes
    heading_color = '\033[92m'  # Bright green for headings
    reset_color = '\033[0m'     # Reset to default

    try:
        # Initialize Figlet with the specified font
        figlet = Figlet(font=font)
        # Generate ASCII art
        ascii_art = figlet.renderText(text)
        # Indent each line by 4 spaces
        indented_ascii_art = '\n'.join('    ' + line for line in ascii_art.splitlines())

        # Print ASCII art with typewriter effect
        print(heading_color, end='')
        for char in indented_ascii_art + '\n':
            print(char, end='', flush=True)
            if char != '\n':  # Don't delay on newlines
                time.sleep(typing_speed)
        print(reset_color, end='')

        # Print separator line with typewriter effect
        print(heading_color, end='')
        for char in '=' * 79 + '\n':
            print(char, end='', flush=True)
            if char != '\n':
                time.sleep(typing_speed)
        print(reset_color, end='')

    except Exception as e:
        error_msg = ''
        if "font" in str(e).lower():
            error_msg = f"Invalid or unavailable font: {font}. Ensure the font is supported by pyfiglet.\n"
            print(reset_color, end='')
            for char in error_msg:
                print(char, end='', flush=True)
                if char != '\n':
                    time.sleep(typing_speed)
            raise ValueError(error_msg)
        error_msg = f"Error generating ASCII art for \"{text}\" with font {font}: {e}\n"
        print(reset_color, end='')
        for char in error_msg:
            print(char, end='', flush=True)
            if char != '\n':
                time.sleep(typing_speed)
        raise RuntimeError(error_msg)

def heading(text: str, typing_speed: float = 0.002) -> None:
    """
    Print a heading with the specified text in uppercase,
    formatted as '[current_call] TEXT' with a timestamp and typewriter effect.
    Ensures the formatted heading is <= 50 characters and total line is 79 characters.
    Adds empty lines before and after the heading.

    Args:
        text (str): The heading text to print.
        typing_speed (float, optional): Delay between printing each character in seconds.
                                      Defaults to 0.005.
    """
    global _PRINT_HEADING_CURRENT_CALL, _CURRENT_HEADING_NUMBER, _PRINT_SUBHEADING_COUNTS

    # Increment heading call
    _PRINT_HEADING_CURRENT_CALL += 1
    _CURRENT_HEADING_NUMBER = _PRINT_HEADING_CURRENT_CALL
    _PRINT_SUBHEADING_COUNTS[_CURRENT_HEADING_NUMBER] = 0  # Reset sub-heading count

    # ANSI color codes
    color = '\033[92m'  # Bright green for headings
    reset_color = '\033[0m'

    # Format heading
    prefix = f"[{_PRINT_HEADING_CURRENT_CALL}] "
    max_text_length = 50 - len(prefix)
    formatted_text = text.upper()[:max_text_length]
    heading = f"{prefix}{formatted_text}"

    # Get timestamp
    timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"

    # Calculate padding
    padding_length = 79 - len(heading) - 1 - len(timestamp) - 1  # Spaces before padding and timestamp
    padding = '=' * padding_length if padding_length > 0 else ''
    full_line = f"{heading} {padding} {timestamp}"

    # Print with line breaks and typewriter effect
    print()  # Empty line before
    print(color, end='')
    for char in full_line + '\n':
        print(char, end='', flush=True)
        if char != '\n':
            time.sleep(typing_speed)
    print(reset_color, end='')
    print()  # Empty line after

def sub_heading(text: str, typing_speed: float = 0.002) -> None:
    """
    Print a sub-heading under the most recent heading, formatted as
    '[heading_num.sub_heading_num] TEXT' with a timestamp and typewriter effect.
    Ensures the formatted sub-heading is <= 50 characters and total line is 79 characters.
    Adds empty lines before and after the sub-heading.

    Args:
        text (str): The sub-heading text to print.
        typing_speed (float, optional): Delay between printing each character in seconds.
                                      Defaults to 0.005.

    Raises:
        ValueError: If no heading has been called.
    """
    global _PRINT_SUBHEADING_COUNTS, _CURRENT_HEADING_NUMBER

    if _CURRENT_HEADING_NUMBER == 0:
        raise ValueError("No heading called before sub_heading.")

    # Increment sub-heading count
    _PRINT_SUBHEADING_COUNTS[_CURRENT_HEADING_NUMBER] += 1
    current_sub = _PRINT_SUBHEADING_COUNTS[_CURRENT_HEADING_NUMBER]

    # ANSI color codes
    color = '\033[92m'  # Bright green for sub-headings
    reset_color = '\033[0m'

    # Format sub-heading
    prefix = f"[{_CURRENT_HEADING_NUMBER}.{current_sub}] "
    max_text_length = 50 - len(prefix)
    formatted_text = text[:max_text_length]  # Removed .lower()
    sub_heading = f"{prefix}{formatted_text}"

    # Get timestamp
    timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"

    # Calculate padding
    padding_length = 79 - len(sub_heading) - 1 - len(timestamp) - 1  # Spaces before padding and timestamp
    padding = '-' * padding_length if padding_length > 0 else ''
    full_line = f"{sub_heading} {padding} {timestamp}"

    # Print with line breaks and typewriter effect
    print()  # Empty line before
    print(color, end='')
    for char in full_line + '\n':
        print(char, end='', flush=True)
        if char != '\n':
            time.sleep(typing_speed)
    print(reset_color, end='')
    print()  # Empty line after
