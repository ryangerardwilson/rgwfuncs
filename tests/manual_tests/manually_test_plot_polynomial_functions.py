import os
import sys
import numpy as np
# from typing import List, Dict, Any, Optional

# flake8: noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rgwfuncs.algebra_lib import plot_polynomial_functions


def main() -> None:
    """
    Runs a series of test cases to evaluate the plot_polynomial_functions function.
    """
    # Define the project root and media directory for saving files
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    media_directory = os.path.join(project_root, 'media')

    # Ensure the 'media' directory exists
    os.makedirs(media_directory, exist_ok=True)

    test_cases = [
        {
            "functions": [
                {"x**2": {"x": "*"}},
                {"x**2/(2 + a) + a": {"x": np.linspace(-3, 4, 101), "a": 1.23}},
                {"np.diff(x**3, 2)": {"x": np.linspace(-2, 2, 10)}}
            ],
            "zoom": 2,
            "file_suffix": "example_1"
        },
        {
            "functions": [
                {"x**3 + 5*x - 2": {"x": np.linspace(-5, 5, 201)}},
                {"x**4 - 4*x**2 + 3": {"x": np.linspace(-3, 3, 151)}},
            ],
            "zoom": 12,
            "file_suffix": "example_2"
        },
        {
            "functions": [
                {"np.sin(x)": {"x": np.linspace(-2 * np.pi, 2 * np.pi, 300)}},
                {"np.cos(x)": {"x": np.linspace(-2 * np.pi, 2 * np.pi, 300)}},
            ],
            "zoom": 3,
            "file_suffix": "example_3"
        },
        {
            "functions": [
                {"x**5 - 4*x**3 + x": {"x": np.linspace(-2, 2, 100)}},
            ],
            "zoom": 10,
            "file_suffix": "example_4"
        },
        {
            "functions": [
                {"np.log(x)": {"x": np.linspace(0.1, 10, 100)}},
                {"np.exp(x)": {"x": np.linspace(0, 3, 100)}},
            ],
            "zoom": 10,
            "file_suffix": "example_5"
        }
    ]

    for test_case in test_cases:
        save_path = os.path.join(media_directory, f'plot_polynomial_functions_{test_case["file_suffix"]}.svg')

        try:
            # Call the plotting function with appropriate flags
            plot_polynomial_functions(
                functions=test_case["functions"],
                zoom=test_case["zoom"],
                show_legend=True,
                open_file=True,
                save_path=save_path
            )
            print(f"SVG successfully saved and opened for: {test_case['file_suffix']}")
        except Exception as e:
            print(f"Failed to plot for {test_case['file_suffix']}. Exception: {e}")


if __name__ == "__main__":
    main()
