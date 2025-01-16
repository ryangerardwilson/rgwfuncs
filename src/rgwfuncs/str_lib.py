import os
import json
import requests
import inspect
from typing import Tuple, Optional, Dict, Callable
import warnings

# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


def str_docs(method_type_filter: Optional[str] = None) -> None:
    """
    Print a list of function names in alphabetical order. If method_type_filter
    is specified, print the docstrings of the functions that match the filter.
    Using '*' as a filter will print the docstrings for all functions.

    Parameters:
        method_type_filter: Optional filter string representing a function name,
        or '*' to display docstrings for all functions.
    """
    # Get the current module's namespace
    current_module = __name__

    local_functions: Dict[str, Callable] = {
        name: obj for name, obj in globals().items()
        if inspect.isfunction(obj) and obj.__module__ == current_module
    }

    # List of function names sorted alphabetically
    function_names = sorted(local_functions.keys())

    # Print function names
    print("Functions in alphabetical order:")
    for name in function_names:
        print(name)

    # If a filter is provided or '*', print the docstrings of functions
    if method_type_filter:
        # print("\nFiltered function documentation:")
        for name, func in local_functions.items():
            docstring: Optional[str] = func.__doc__
            if docstring:
                if method_type_filter == '*' or method_type_filter == name:
                    # Print the entire docstring for the matching function
                    print(f"\n{name}:\n{docstring}")


def send_telegram_message(preset_name: str, message: str) -> None:
    """Send a Telegram message using the specified preset.

    Args:
        preset_name (str): The name of the preset to use for sending the message.
        message (str): The message to send.

    Raises:
        RuntimeError: If the preset is not found or necessary details are missing.
    """

    # Set the config path to ~/.rgwfuncsrc
    config_path = os.path.expanduser("~/.rgwfuncsrc")

    def load_config() -> dict:
        """Load the configuration from the .rgwfuncsrc file."""
        with open(config_path, 'r') as file:
            return json.load(file)

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
            raise RuntimeError(
                f"Telegram bot token or chat ID for '{preset_name}' not found in the configuration file"
            )

        return bot_token, chat_id

    # Load the configuration
    config = load_config()

    # Get bot details from the configuration
    bot_token, chat_id = get_telegram_bot_details(config, preset_name)

    # Prepare the request
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}

    # Send the message
    response = requests.post(url, json=payload)
    response.raise_for_status()
