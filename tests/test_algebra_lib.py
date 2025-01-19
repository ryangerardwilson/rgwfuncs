from src.rgwfuncs.algebra_lib import compute_algebraic_expression, simplify_algebraic_expression, solve_algebraic_expression, get_prime_factors_latex
import sys
import os
import math

# flake8: noqa: E402
# Allow the following import statement to be AFTER sys.path modifications
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_compute_algebraic_expression():
    test_cases = [
        ("2 + 2", 4.0),
        ("5 - 3", 2.0),
        ("3 * 3", 9.0),
        ("8 / 2", 4.0),
        ("10 % 3", 1.0),
        ("math.gcd(36, 60) * math.sin(math.radians(45)) * 10000", 84852.8137423857),
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
