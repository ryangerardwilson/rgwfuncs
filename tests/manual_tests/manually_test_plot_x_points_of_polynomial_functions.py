import os
import sys
import numpy as np

# flake8: noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the revised function (now can handle multiple expressions in a list).
from src.rgwfuncs.algebra_lib import plot_x_points_of_polynomial_functions


def main() -> None:

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    media_directory = os.path.join(project_root, 'media')
    os.makedirs(media_directory, exist_ok=True)

    test_cases = [
        {
            # Single expression in a list
            "functions": [
                {"x**2 - np.sin(x**2)": {"x": np.linspace(-3, 3, 7)}}
            ],
            "zoom": 10,
            "file_suffix": "example_1"
        },
        {
            # Single expression in a list
            "functions": [
                {"np.sin(x)": {"x": np.linspace(0, 2 * np.pi, 10)}}
            ],
            "zoom": 7,
            "file_suffix": "example_2"
        },
        {
            # Single expression with an extra parameter 'a'
            "functions": [
                {"x**3 + a": {"x": np.array([-2, -1, 0, 1, 2]), "a": 2}}
            ],
            "zoom": 5,
            "file_suffix": "example_3"
        },
        {
            # Single expression focusing on np.log
            "functions": [
                {"np.log(x)": {"x": np.array([0.1, 0.2, 0.5, 1.0, 2.0, 3.0])}}
            ],
            "zoom": 5,
            "file_suffix": "example_4"
        },
        {
            # Demonstrates plotting multiple expressions in one figure
            "functions": [
                {"x**2": {"x": np.array([-2, -1, 0, 1, 2])}},
                {"np.sin(x)": {"x": np.linspace(0, 2 * np.pi, 5)}},
                {"x - 3": {"x": np.array([-2, 0, 2, 4, 6])}}
            ],
            "zoom": 6,
            "file_suffix": "example_5"
        },
    ]

    for test_case in test_cases:
        save_path = os.path.join(
            media_directory,
            f'plot_x_points_of_polynomial_functions_{test_case["file_suffix"]}.svg'
        )

        try:
            # Call the plotting function for each test case
            plot_x_points_of_polynomial_functions(
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
