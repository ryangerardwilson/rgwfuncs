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
        },
        {
            "functions": [
                {"x**2": {"x": "*"}},
                {"x**2/(2 + a) + a": {"x": np.array([-3.  , -2.93, -2.86, -2.79, -2.72, -2.65, -2.58, -2.51, -2.44,
       -2.37, -2.3 , -2.23, -2.16, -2.09, -2.02, -1.95, -1.88, -1.81,
       -1.74, -1.67, -1.6 , -1.53, -1.46, -1.39, -1.32, -1.25, -1.18,
       -1.11, -1.04, -0.97, -0.9 , -0.83, -0.76, -0.69, -0.62, -0.55,
       -0.48, -0.41, -0.34, -0.27, -0.2 , -0.13, -0.06,  0.01,  0.08,
        0.15,  0.22,  0.29,  0.36,  0.43,  0.5 ,  0.57,  0.64,  0.71,
        0.78,  0.85,  0.92,  0.99,  1.06,  1.13,  1.2 ,  1.27,  1.34,
        1.41,  1.48,  1.55,  1.62,  1.69,  1.76,  1.83,  1.9 ,  1.97,
        2.04,  2.11,  2.18,  2.25,  2.32,  2.39,  2.46,  2.53,  2.6 ,
        2.67,  2.74,  2.81,  2.88,  2.95,  3.02,  3.09,  3.16,  3.23,
        3.3 ,  3.37,  3.44,  3.51,  3.58,  3.65,  3.72,  3.79,  3.86,
        3.93,  4.  ]), "a": 1.23}},
                {"np.diff(x**3, 2)": {"x": np.linspace(-2, 2, 10)}}
            ],
            "zoom": 10,
            "file_suffix": "example_6"
        },


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
