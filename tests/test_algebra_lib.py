import sys
import os
import math

# flake8: noqa: E402
# Allow the following import statement to be AFTER sys.path modifications
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rgwfuncs.algebra_lib import (
    compute_algebraic_expression,
    simplify_algebraic_expression,
    solve_algebraic_expression,
    compute_matrix_operation,
    compute_ordered_series_operation,
    get_prime_factors_latex)


def test_compute_algebraic_expression():
    test_cases = [
        ("2 + 2", 4.0),
        ("5 - 3", 2.0),
        ("3 * 3", 9.0),
        ("8 / 2", 4.0),
        ("10 % 3", 1.0),
        ("math.gcd(36, 60) * math.sin(math.radians(45)) * 10000", 84852.8137423857),
        # ("np.diff([2,6,9,60])", r"\left[\begin{matrix}4\\3\\51\end{matrix}\right]"),
    ]

    for input_data, expected_output in test_cases:
        result = compute_algebraic_expression(input_data)
        assert math.isclose(result, expected_output, rel_tol=1e-9), f"Failed for {input_data}, got {result}"


def test_simplify_algebraic_expression():
    test_cases = [
        ("(np.diff(3*x**8)) / (np.diff(8*x**30) * 11*y**3)", r"\frac{1}{110 x^{22} y^{3}}"),
    ]

    for input_data, expected_output in test_cases:
        assert simplify_algebraic_expression(input_data) == expected_output


def test_solve_algebraic_expression():
    test_cases = [
        # Test case with substitutions
        (
            ("a*x**2 + b*x + c", "x", {"a": 3, "b": 7, "c": 5}),
            r"\left[-7/6 - sqrt(11)*I/6, -7/6 + sqrt(11)*I/6\right]"
        ),
    ]

    for (expression, variable, subs), expected_output in test_cases:
        assert solve_algebraic_expression(expression, variable, subs) == expected_output


def test_compute_matrix_operation():
    test_cases = [
        ("[[2, 6, 9],[1, 3, 5]] + [[1, 2, 3],[4, 5, 6]]", r"\begin{bmatrix}3 & 8 & 12\\5 & 8 & 11\end{bmatrix}"),
        ("[[10, 10, 10],[2, 4, 6]] - [[5, 3, 2],[1, 2, 1]]", r"\begin{bmatrix}5 & 7 & 8\\1 & 2 & 5\end{bmatrix}"),
        ("[[2, 4],[6, 8]] * [[1, 0.5],[2, 0.25]]", r"\begin{bmatrix}2 & 2.0\\12 & 2.0\end{bmatrix}"),
        ("[[8, 16],[32, 64]] / [[2, 2],[8, 16]]", r"\begin{bmatrix}4.0 & 8.0\\4.0 & 4.0\end{bmatrix}"),

        ("[[2, 6, 9], [1, 3, 5]] + [[1, 2, 3], [4, 5, 6]] - [[1, 1, 1], [1, 1, 1]]", r"\begin{bmatrix}2 & 7 & 11\\4 & 7 & 10\end{bmatrix}"),
        ("[2, 6, 9] + [1, 2, 3] - [1, 1, 1]", r"\begin{bmatrix}2 & 7 & 11\end{bmatrix}"),
        ("[[1, 2], [3, 4]] + [[2, 3], [4, 5]] + [[1, 1], [1, 1]]", r"\begin{bmatrix}4 & 6\\8 & 10\end{bmatrix}"),
        ("[3, 6, 9] - [1, 2, 3] + [5, 5, 5]", r"\begin{bmatrix}7 & 9 & 11\end{bmatrix}"),


        ("[3, 6, 9] - [1, 2, 3, 4]", r"Operations between matrices must involve matrices of the same dimension"),

        # Edge cases
        ("[]", r"\begin{bmatrix}\end{bmatrix}"),  # Empty list
        ("[5]", r"\begin{bmatrix}5\end{bmatrix}"),  # Single-element list
    ]

    for input_data, expected_output in test_cases:
        result = compute_matrix_operation(input_data)
        assert result == expected_output, f"Failed for {input_data}, got {result}"

# Example test function


def test_compute_ordered_series_operations():
    test_cases = [
        ("[2, 6, 9] + [1, 2, 3]", "[3, 8, 12]"),
        ("[10, 15, 21] - [5, 5, 5]", "[5, 10, 16]"),
        ("[2, 4, 6] * [1, 2, 3]", "[2, 8, 18]"),
        ("[8, 16, 32] / [2, 2, 8]", "[4.0, 8.0, 4.0]"),
        ("ddd([2, 6, 9, 60]) + ddd([78, 79, 80])", "Operations between ordered series must involve series of equal length"),
        ("ddd([1, 3, 6, 10]) - ddd([0, 1, 1, 2])", "[1, 3, 3]"),
        # Edge cases
        ("ddd([1])", "[]"),  # Single-element list, becomes empty
        ("ddd([])", "[]"),   # Empty list case
        ("[5]", "[5]"),  # Single-element list, unchanged
        ("[]", "[]"),    # Empty list
        ("[4, 3, 51] + [1, 1]", "Operations between ordered series must involve series of equal length"),  # Test unequal lengths
    ]

    for input_data, expected_output in test_cases:
        result = compute_ordered_series_operation(input_data)
        assert result == expected_output, f"Failed for {input_data}, got {result}"


def test_get_prime_factors_latex():
    test_cases = [
        (100, "2^{2} \\cdot 5^{2}"),
        (60, "2^{2} \\cdot 3 \\cdot 5"),
        (45, "3^{2} \\cdot 5"),
        (1, ""),  # Handle case with 1, which has no prime factors
        (17, "17")  # Prime number itself
    ]

    for n, expected_output in test_cases:
        result = get_prime_factors_latex(n)
        assert result == expected_output, f"Failed for {n}, got {result}"
