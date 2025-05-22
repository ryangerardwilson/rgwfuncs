import os
import json
import requests
from typing import Tuple, Optional, Union, Dict
import warnings

# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


def send_telegram_message(preset_name: str, message: str, config: Optional[Union[str, dict]] = None) -> None:
    """
    Send a Telegram message using the specified preset.

    Args:
        preset_name (str): The name of the preset to use for sending the message.
        message (str): The message to send.
        config (Optional[Union[str, dict]], optional): Configuration source. Can be:
          - None: Searches for '.rgwfuncsrc' in current directory and upwards
          - str: Path to a JSON configuration file
          - dict: Direct configuration dictionary

    Raises:
        FileNotFoundError: If no '.rgwfuncsrc' file is found after traversing all parent directories.
        RuntimeError: If the preset is not found or necessary details are missing.
        ValueError: If the config parameter is neither a path string nor a dictionary.
    """
    def get_config(config: Optional[Union[str, dict]] = None) -> dict:
        """Get configuration either from a path, direct dictionary, or by searching upwards."""
        def get_config_from_file(config_path: str) -> dict:
            """Load configuration from a JSON file."""
            with open(config_path, 'r') as file:
                return json.load(file)

        def find_config_file() -> str:
            """Search for '.rgwfuncsrc' in current directory and upwards."""
            current_dir = os.getcwd()
            while True:
                config_path = os.path.join(current_dir, '.rgwfuncsrc')
                if os.path.isfile(config_path):
                    return config_path
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # Reached root directory
                    raise FileNotFoundError("No '.rgwfuncsrc' file found in current or parent directories")
                current_dir = parent_dir

        # Determine the config to use
        if config is None:
            # Search for .rgwfuncsrc upwards from current directory
            config_path = find_config_file()
            return get_config_from_file(config_path)
        elif isinstance(config, str):
            # If config is a string, treat it as a path and load it
            return get_config_from_file(config)
        elif isinstance(config, dict):
            # If config is already a dict, use it directly
            return config
        else:
            raise ValueError("Config must be either a path string or a dictionary")

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
            raise RuntimeError(
                f"Telegram bot preset '{preset_name}' not found in the configuration file")

        bot_token = preset.get("bot_token")
        chat_id = preset.get("chat_id")

        if not bot_token or not chat_id:
            raise RuntimeError(
                f"Telegram bot token or chat ID for '{preset_name}' not found in the configuration file")

        return bot_token, chat_id

    config = get_config(config)
    # Get bot details from the configuration
    bot_token, chat_id = get_telegram_bot_details(config, preset_name)

    # Prepare the request
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}

    # Send the message
    response = requests.post(url, json=payload)
    response.raise_for_status()
