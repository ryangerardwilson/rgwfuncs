import sys
import os
import math
import csv
import tempfile
import pandas as pd
import numpy as np

# flake8: noqa: E402
# Allow the following import statement to be AFTER sys.path modifications
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rgwfuncs.df_lib import load_data_from_path
from src.rgwfuncs.interactive_shell_lib import interactive_shell


# Create a temporary file
with tempfile.NamedTemporaryFile(mode='w', newline='', delete=False, suffix='.csv') as temp_file:
    # Get the name of the temp file
    temp_file_name = temp_file.name

    # Define sample data
    data = [
        ['id', 'name', 'age', 'city'],
        [1, 'Alice', 30, 'New York'],
        [2, 'Bob', 25, 'Los Angeles'],
        [3, 'Charlie', 35, 'Chicago'],
        [4, 'David', 28, 'San Francisco'],
        [5, 'Eva', 22, 'Boston']
    ]

    # Write data to the temporary CSV file
    writer = csv.writer(temp_file)
    writer.writerows(data)

# Display the name of the created temporary file
print(f"Temporary CSV file created at: {temp_file_name}")

df = load_data_from_path(temp_file_name)
print(df)

# Launch interactive shell
interactive_shell(locals())
