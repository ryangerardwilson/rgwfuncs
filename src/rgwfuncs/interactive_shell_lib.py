import code
import readline
import rlcompleter  # noqa: F401
import sys  # noqa: F401
from typing import Dict, Any
from .df_lib import *  # noqa: F401, F403, E402
from .algebra_lib import *  # noqa: F401, F403, E402
from .str_lib import *  # noqa: F401, F403, E402
from .docs_lib import *  # noqa: F401, F403, E402


def interactive_shell(local_vars: Dict[str, Any]) -> None:
    """
    Launches an interactive prompt for inspecting and modifying local variables, making all methods
    in the rgwfuncs library available by default.

    Parameters:
        local_vars (dict): Dictionary of local variables to be available in the interactive shell.
    """
    if not isinstance(local_vars, dict):
        raise TypeError("local_vars must be a dictionary")

    readline.parse_and_bind("tab: complete")

    # Make imported functions available in the REPL
    local_vars.update(globals())

    # Create interactive console with local context
    console = code.InteractiveConsole(locals=local_vars)

    # Start interactive session
    console.interact(banner="Welcome to the rgwfuncs interactive shell.")
