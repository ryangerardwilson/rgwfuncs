import sys
import os
import math

# flake8: noqa: E402
# Allow the following import statement to be AFTER sys.path modifications
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rgwfuncs.algebra_lib import (
    compute_prime_factors,
    compute_constant_expression,
    compute_constant_expression_involving_matrices,
    compute_constant_expression_involving_ordered_series,
    python_polynomial_expression_to_latex,
    expand_polynomial_expression,
    factor_polynomial_expression,
    cancel_polynomial_expression,
    simplify_polynomial_expression,
    solve_homogeneous_polynomial_expression)


def test_compute_prime_factors():
    test_cases = [
        (100, "2^{2} \\cdot 5^{2}"),
        (60, "2^{2} \\cdot 3 \\cdot 5"),
        (45, "3^{2} \\cdot 5"),
        (1, ""),  # Handle case with 1, which has no prime factors
        (17, "17")  # Prime number itself
    ]

    for n, expected_output in test_cases:
        result = compute_prime_factors(n)
        assert result == expected_output, f"Failed for {n}, got {result}"


def test_compute_constant_expression():
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
        result = compute_constant_expression(input_data)
        assert math.isclose(result, expected_output, rel_tol=1e-9), f"Failed for {input_data}, got {result}"


def test_compute_constant_expression_involving_matrices():
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
        result = compute_constant_expression_involving_matrices(input_data)
        assert result == expected_output, f"Failed for {input_data}, got {result}"

# Example test function


def test_compute_constant_expression_involving_ordered_series():
    test_cases = [
        ("[2, 6, 9] + [1, 2, 3]", "[3, 8, 12]"),
        ("[10, 15, 21] - [5, 5, 5]", "[5, 10, 16]"),
        ("[2, 4, 6] * [1, 2, 3]", "[2, 8, 18]"),
        ("[8, 16, 32] / [2, 2, 8]", "[4.0, 8.0, 4.0]"),
        ("dd([2, 6, 9, 60]) + dd([78, 79, 80])", "Operations between ordered series must involve series of equal length"),
        ("dd([1, 3, 6, 10]) - dd([0, 1, 1, 2])", "[1, 3, 3]"),

        # Edge cases
        ("dd([1])", "[]"),  # Single-element list, becomes empty
        ("dd([])", "[]"),   # Empty list case
        ("[5]", "[5]"),  # Single-element list, unchanged
        ("[]", "[]"),    # Empty list
        ("[4, 3, 51] + [1, 1]", "Operations between ordered series must involve series of equal length"),  # Test unequal lengths
    ]

    for input_data, expected_output in test_cases:
        result = compute_constant_expression_involving_ordered_series(input_data)
        assert result == expected_output, f"Failed for {input_data}, got {result}"


def test_python_polynomial_expression_to_latex():
    test_cases = [
        # Without substitutions
        ("x**2 + y**2", None, r"x^{2} + y^{2}"),
        ("3*a + 4*b", None, r"3 a + 4 b"),
        ("3*x**3 / (7*y*m**(n+2) + 103)", None, r"\frac{3 x^{3}}{7 m^{n + 2} y + 103}"),

        # With substitutions
        ("x**2 + y**2", {"x": 3, "y": 4}, r"25"),
        ("x**2 + y**2", {"x": 3}, r"y^{2} + 9"),
        ("a*b + b", {"b": 2}, r"2 a + 2"),
        ("sin(x+z**2) + cos(y)", {"x": 55}, r"cos y + sin \left(z^{2} + 55\right)"),
        ("sin(x) + cos(y)", {"x": 0}, r"cos y")
    ]

    for expression, subs, expected_output in test_cases:
        output = python_polynomial_expression_to_latex(expression, subs)
        assert output == expected_output, (
            f"Test failed for expression: {expression} with substitutions: {subs}. "
            f"Expected {expected_output}, got {output}"
        )


def test_expand_polynomial_expression():
    test_cases = [
        # Without substitutions
        ("(x + y)**2", None, r"x^{2} + 2 x y + y^{2}"),
        ("(a + b)**3", None, r"a^{3} + 3 a^{2} b + 3 a b^{2} + b^{3}"),
        ("(p + q + r)**2", None, r"p^{2} + 2 p q + 2 p r + q^{2} + 2 q r + r^{2}"),

        # With substitutions
        ("(x + y)**2", {"x": 3, "y": 4}, r"49"),
        ("(a + b)**2", {"a": 1}, r"b^{2} + 2 b + 1"),
        ("(u + v)**3", {"v": 2}, r"u^{3} + 6 u^{2} + 12 u + 8"),
        ("(s * t)**2", {"s": 2, "t": 3}, r"36"),
        ("cos(x + y)", {"x": 0}, r"cos y"),
    ]

    for expression, subs, expected_output in test_cases:
        output = expand_polynomial_expression(expression, subs)
        assert output == expected_output, (
            f"Test failed for expression: {expression} with substitutions: {subs}. "
            f"Expected {expected_output}, got {output}"
        )


def test_factor_polynomial_expression():
    test_cases = [
        # Without substitutions
        ("x**2 - 4", None, r"\left(x - 2\right) \left(x + 2\right)"),
        ("a**3 + 3*a**2*b + 3*a*b**2 + b**3", None, r"\left(a + b\right)^{3}"),
        ("p**2 - 2*p*q + q**2", None, r"\left(p - q\right)^{2}"),

        # With substitutions
        ("x**2 - 4", {"x": 2}, r"0"),
        ("a**2 - b**2", {"b": 1}, r"\left(a - 1\right) \left(a + 1\right)"),
        ("u**2 - 1", {"u": 1}, r"0"),
        ("t**2 - 4*t + 4", {"t": 2}, r"0"),  # Perfect square
        ("x**2 + 2*x*y + y**2", {"y": 1}, r"\left(x + 1\right)^{2}"),
    ]

    for expression, subs, expected_output in test_cases:
        output = factor_polynomial_expression(expression, subs)
        assert output == expected_output, (
            f"Test failed for expression: {expression} with substitutions: {subs}. "
            f"Expected {expected_output}, got {output}"
        )


def test_cancel_polynomial_expression():
    test_cases = [
        # Without substitutions
        ("(x**2 - 4) / (x - 2)", None, r"x + 2"),
        ("(a**2 - b**2) / (a - b)", None, r"a + b"),
        ("(p**2 - q**2) / (p + q)", None, r"p - q"),

        # With substitutions
        ("(x**2 - 4) / (x - 2)", {"x": 2}, r"Undefined result. This could be a division by zero error."),
        ("(a**2 - 1) / (a - 1)", {"a": 2}, r"3"),
        ("(u**2 - 1) / (u - 1)", {"u": 1}, r"Undefined result. This could be a division by zero error."),
        ("(t**2 - 4*t + 4) / (t - 2)", {"t": 3}, r"1"),
        ("(x**2 + 2*x + 1) / (x + 1)", {"x": 0}, r"1"),
    ]

    for expression, subs, expected_output in test_cases:
        output = cancel_polynomial_expression(expression, subs)
        assert output == expected_output, (
            f"Test failed for expression: {expression} with substitutions: {subs}. "
            f"Expected {expected_output}, got {output}"
        )


def test_simplify_polynomial_expression():
    test_cases = [
        # Without substitutions
        (("(np.diff(3*x**8)) / (np.diff(8*x**30) * 11*y**3)", None), r"\frac{1}{110 x^{22} y^{3}}"),

        # With substitutions
        (("x**2 + y**2", {"x": 3, "y": 4}), "25"),
        (("a*b + b", {"b": 2}), r"2 a + 2"),  # Assumes no simplification of `a*b`
        (("(x**2 + y**2 + z**2)", {"x": 1, "y": 0, "z": 0}), "1")
    ]

    for (expression, subs), expected_output in test_cases:
        output = simplify_polynomial_expression(expression, subs)
        assert output == expected_output, (f"Test failed for expression: {expression} with substitutions: {subs}. Expected {expected_output}, got {output}")


def test_solve_homogeneous_polynomial_expression():
    test_cases = [
        # Test case with substitutions
        (("a*x**2 + b*x + c", "x", {"a": 3, "b": 7, "c": 5}), r"\left[-7/6 - sqrt(11)*I/6, -7/6 + sqrt(11)*I/6\right]"),
    ]

    for (expression, variable, subs), expected_output in test_cases:
        assert solve_homogeneous_polynomial_expression(expression, variable, subs) == expected_output
