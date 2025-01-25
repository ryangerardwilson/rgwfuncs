import os
import sys
import math
import csv
import tempfile
import pandas as pd
import numpy as np
import subprocess

# Adjust import path for local modules
# flake8: noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rgwfuncs.algebra_lib import plot_polynomial_functions

# Generate the SVG
svg_output = plot_polynomial_functions(
    functions=[
        {"x**2": {"x": "*"}},  # Single expression, "*" means use default domain
        {"x**2/(2 + a) + a": {"x": np.linspace(-3, 4, 101), "a": 1.23}},
        {"np.diff(x**3, 2)": {"x": np.linspace(-2, 2, 10)}}
    ],
    zoom=2
)

# Write the SVG to a temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmpfile:
    temp_svg_path = tmpfile.name
    tmpfile.write(svg_output.encode('utf-8'))

print(f"SVG written to temporary file: {temp_svg_path}")

# Path relative to your project directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
media_directory = os.path.join(project_root, 'media')
media_file_path = os.path.join(media_directory, 'plot_polynomial_functions_example_1.svg')

# Ensure the 'media' directory exists
os.makedirs(media_directory, exist_ok=True)

# Save the SVG to the 'media' directory
try:
    with open(media_file_path, 'w', encoding='utf-8') as media_file:
        media_file.write(svg_output)
    print(f"SVG also saved to: {media_file_path}")
except IOError as e:
    print(f"Failed to save SVG to {media_file_path}. IOError: {e}")

# Open the SVG file with the system's default viewer
result = subprocess.run(["xdg-open", temp_svg_path], stderr=subprocess.DEVNULL)

# Check if the process failed
if result.returncode != 0:
    print("Failed to open the SVG file with the default viewer.")
