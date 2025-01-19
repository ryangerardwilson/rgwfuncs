import os
import inspect
from typing import Optional
import warnings

# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


def docs(method_type_filter: Optional[str] = None) -> None:
    """
    Print a list of function names in alphabetical order from all modules.
    If method_type_filter is specified, print the docstrings of the functions
    that match the filter based on a substring. Using '*' as a filter will print
    the docstrings for all functions.

    Parameters:
        method_type_filter: Optional filter string representing a filter for
        function names, or '*' to display docstrings for all functions.
    """

    # Directory containing your modules
    module_dir = os.path.dirname(__file__)

    # Iterate over each file in the module directory
    for filename in sorted(os.listdir(module_dir)):
        if filename.endswith('.py') and filename != '__init__.py':
            module_name, _ = os.path.splitext(filename)
            print(f"\n# {module_name}.py")

            # Import the module
            module_path = f"rgwfuncs.{module_name}"
            module = __import__(module_path, fromlist=[module_name])

            # Get all functions from the module
            functions = {
                name: obj for name, obj
                in inspect.getmembers(module, inspect.isfunction)
                if obj.__module__ == module_path
            }

            # List function names
            function_names = sorted(functions.keys())
            for name in function_names:
                # If a filter is provided or '*', check if the function name
                # contains the filter
                if method_type_filter and (
                        method_type_filter == '*' or method_type_filter in name):
                    docstring: Optional[str] = functions[name].__doc__
                    if docstring:
                        print(f"\n{name}:\n{docstring}")
