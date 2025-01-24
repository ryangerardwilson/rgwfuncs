import re
import math
import ast
# import numpy as np
from sympy import symbols, latex, simplify, solve, diff, Expr, factor, cancel
from sympy.core.sympify import SympifyError
from sympy.core import S
from sympy.parsing.sympy_parser import parse_expr
from sympy import __all__ as sympy_functions
from sympy.parsing.sympy_parser import (standard_transformations, implicit_multiplication_application)

from typing import Tuple, List, Dict, Optional


def compute_prime_factors(n: int) -> str:
    """
    Computes the prime factors of a number and returns the factorization as a LaTeX string.

    Determines the prime factorization of the given integer. The result is formatted as a LaTeX
    string, enabling easy integration into documents or presentations that require mathematical notation.

    Parameters:
    n (int): The number for which to compute prime factors.

    Returns:
    str: The LaTeX representation of the prime factorization.
    """

    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n //= i
    if n > 2:
        factors.append(n)

    factor_counts = {factor: factors.count(factor) for factor in set(factors)}
    latex_factors = [f"{factor}^{{{count}}}" if count > 1 else str(
        factor) for factor, count in factor_counts.items()]
    return " \\cdot ".join(latex_factors)


def compute_constant_expression(expression: str) -> float:
    """
    Computes the numerical result of a given expression, which can evaluate to a constant,
    represented as a float.

    Evaluates an constant expression provided as a string and returns the computed result.
    Supports various arithmetic operations, including addition, subtraction, multiplication,
    division, and modulo, as well as mathematical functions from the math module.

    Parameters:
    expression (str): The constant expression to compute. This should be a string consisting
                      of arithmetic operations and Python's math module functions.

    Returns:
    float: The evaluated numerical result of the expression.

    Raises:
    ValueError: If the expression cannot be evaluated due to syntax errors or other issues.
    """
    try:
        # Direct numerical evaluation
        # Safely evaluate the expression using the math module
        numeric_result = eval(expression, {"__builtins__": None, "math": math})

        # Convert to float if possible
        return float(numeric_result)
    except Exception as e:
        raise ValueError(f"Error computing expression: {e}")


def compute_constant_expression_involving_matrices(expression: str) -> str:
    """
    Computes the result of a constant expression involving matrices and returns it as a LaTeX string.

    Parameters:
    expression (str): The constant expression involving matrices. Example format includes operations such as "+",
    "-", "*", "/".

    Returns:
    str: The LaTeX-formatted string representation of the result or a message indicating an error in dimensions.
    """

    def elementwise_operation(matrix1: List[List[float]], matrix2: List[List[float]], operation: str) -> List[List[float]]:
        if len(matrix1) != len(matrix2) or any(len(row1) != len(row2) for row1, row2 in zip(matrix1, matrix2)):
            return "Operations between matrices must involve matrices of the same dimension"

        if operation == '+':
            return [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(matrix1, matrix2)]
        elif operation == '-':
            return [[a - b for a, b in zip(row1, row2)] for row1, row2 in zip(matrix1, matrix2)]
        elif operation == '*':
            return [[a * b for a, b in zip(row1, row2)] for row1, row2 in zip(matrix1, matrix2)]
        elif operation == '/':
            return [[a / b for a, b in zip(row1, row2) if b != 0] for row1, row2 in zip(matrix1, matrix2)]
        else:
            return f"Unsupported operation {operation}"

    try:
        # Use a stack-based method to properly parse matrices
        elements = []
        buffer = ''
        bracket_level = 0
        operators = set('+-*/')

        for char in expression:
            if char == '[':
                if bracket_level == 0 and buffer.strip():
                    elements.append(buffer.strip())
                    buffer = ''
                bracket_level += 1
            elif char == ']':
                bracket_level -= 1
                if bracket_level == 0:
                    buffer += char
                    elements.append(buffer.strip())
                    buffer = ''
                    continue
            if bracket_level == 0 and char in operators:
                if buffer.strip():
                    elements.append(buffer.strip())
                    buffer = ''
                elements.append(char)
            else:
                buffer += char

        if buffer.strip():
            elements.append(buffer.strip())

        result = ast.literal_eval(elements[0])

        if not any(isinstance(row, list) for row in result):
            result = [result]  # Convert 1D matrix to 2D

        i = 1
        while i < len(elements):
            operation = elements[i]
            matrix = ast.literal_eval(elements[i + 1])

            if not any(isinstance(row, list) for row in matrix):
                matrix = [matrix]

            operation_result = elementwise_operation(result, matrix, operation)

            # Check if the operation resulted in an error message
            if isinstance(operation_result, str):
                return operation_result

            result = operation_result
            i += 2

        # Create a LaTeX-style matrix representation
        matrix_entries = '\\\\'.join(' & '.join(str(x) for x in row) for row in result)
        return r"\begin{bmatrix}" + f"{matrix_entries}" + r"\end{bmatrix}"

    except Exception as e:
        return f"Error computing matrix operation: {e}"


def compute_constant_expression_involving_ordered_series(expression: str) -> str:
    """
    Computes the result of a constant expression involving ordered series, and returns it as a Latex string.
    Supports operations lile "+", "-", "*", "/", as well as "dd()" (the discrete difference operator).

    The function first applies the discrete difference operator to any series where applicable, then evaluates
    arithmetic operations between series.

    Parameters:
    expression (str): The series operation expression to compute. Includes operations "+", "-", "*", "/", and "dd()".

    Returns:
    str: The string representation of the resultant series after performing operations, or an error message
    if the series lengths do not match.

    Raises:
    ValueError: If the expression cannot be evaluated.
    """

    def elementwise_operation(series1: List[float], series2: List[float], operation: str) -> List[float]:
        if len(series1) != len(series2):
            return "Operations between ordered series must involve series of equal length"

        if operation == '+':
            return [a + b for a, b in zip(series1, series2)]
        elif operation == '-':
            return [a - b for a, b in zip(series1, series2)]
        elif operation == '*':
            return [a * b for a, b in zip(series1, series2)]
        elif operation == '/':
            return [a / b for a, b in zip(series1, series2) if b != 0]
        else:
            return f"Unsupported operation {operation}"

    def discrete_difference(series: list) -> list:
        """Computes the discrete difference of a series."""
        return [series[i + 1] - series[i] for i in range(len(series) - 1)]

    try:
        # First, apply the discrete difference operator where applicable
        pattern = r'dd\((\[[^\]]*\])\)'
        matches = re.findall(pattern, expression)

        for match in matches:
            if match.strip() == '[]':
                result_series = []  # Handle the empty list case
            else:
                series = ast.literal_eval(match)
                result_series = discrete_difference(series)
            expression = expression.replace(f'dd({match})', str(result_series))

        # Now parse and evaluate the full expression with basic operations
        elements = []
        buffer = ''
        bracket_level = 0
        operators = set('+-*/')

        for char in expression:
            if char == '[':
                if bracket_level == 0 and buffer.strip():
                    elements.append(buffer.strip())
                    buffer = ''
                bracket_level += 1
            elif char == ']':
                bracket_level -= 1
                if bracket_level == 0:
                    buffer += char
                    elements.append(buffer.strip())
                    buffer = ''
                    continue
            if bracket_level == 0 and char in operators:
                if buffer.strip():
                    elements.append(buffer.strip())
                    buffer = ''
                elements.append(char)
            else:
                buffer += char

        if buffer.strip():
            elements.append(buffer.strip())

        result = ast.literal_eval(elements[0])

        i = 1
        while i < len(elements):
            operation = elements[i]
            series = ast.literal_eval(elements[i + 1])
            operation_result = elementwise_operation(result, series, operation)

            # Check if the operation resulted in an error message
            if isinstance(operation_result, str):
                return operation_result

            result = operation_result
            i += 2

        return str(result)

    except Exception as e:
        return f"Error computing ordered series operation: {e}"


def python_polynomial_expression_to_latex(
    expression: str,
    subs: Optional[Dict[str, float]] = None
) -> str:
    """
    Converts a polynomial expression written in Python syntax to LaTeX format.

    This function takes an algebraic expression written in Python syntax and converts it
    to a LaTeX formatted string. The expression is assumed to be in terms acceptable by
    sympy, with named variables, and optionally includes substitutions for variables.

    Parameters:
    expression (str): The algebraic expression to convert to LaTeX. The expression should
                      be written using Python syntax.
    subs (Optional[Dict[str, float]]): An optional dictionary of substitutions for variables
                                       in the expression.

    Returns:
    str: The expression represented as a LaTeX string.

    Raises:
    ValueError: If the expression cannot be parsed due to syntax errors.
    """

    transformations = standard_transformations + (implicit_multiplication_application,)

    def parse_and_convert_expression(expr_str: str, sym_vars: Dict[str, Expr]) -> Expr:
        try:
            # Parse with transformations to handle implicit multiplication
            expr = parse_expr(expr_str, local_dict=sym_vars, transformations=transformations)
            if subs:
                subs_symbols = {symbols(k): v for k, v in subs.items()}
                expr = expr.subs(subs_symbols)
            return expr
        except (SyntaxError, ValueError, TypeError) as e:
            raise ValueError(f"Error parsing expression: {expr_str}. Error: {e}")

    # Extract variable names used in the expression
    variable_names = set(re.findall(r'\b[a-zA-Z]\w*\b', expression))
    sym_vars = {var: symbols(var) for var in variable_names}

    # Import all general function names from SymPy into local scope

    # Dynamically add SymPy functions to the symbol dictionary
    for func_name in sympy_functions:
        try:
            candidate = globals().get(func_name) or locals().get(func_name)
            if callable(candidate):  # Ensure it's actually a callable
                sym_vars[func_name] = candidate
        except KeyError:
            continue  # Skip any non-callable or unavailable items

    # Attempt to parse the expression
    expr = parse_and_convert_expression(expression, sym_vars)

    # Convert the expression to LaTeX format
    latex_result = latex(expr)
    return latex_result


def expand_polynomial_expression(
    expression: str,
    subs: Optional[Dict[str, float]] = None
) -> str:
    """
    Expands a polynomial expression written in Python syntax and converts it to LaTeX format.

    This function takes an algebraic expression written in Python syntax,
    applies polynomial expansion, and converts the expanded expression
    to a LaTeX formatted string. The expression should be compatible with sympy.

    Parameters:
    expression (str): The algebraic expression to expand and convert to LaTeX.
                      The expression should be written using Python syntax.
    subs (Optional[Dict[str, float]]): An optional dictionary of substitutions
                                       to apply to variables in the expression
                                       before expansion.

    Returns:
    str: The expanded expression represented as a LaTeX string.

    Raises:
    ValueError: If the expression cannot be parsed due to syntax errors.
    """
    transformations = standard_transformations + (implicit_multiplication_application,)

    def parse_and_expand_expression(expr_str: str, sym_vars: Dict[str, symbols]) -> symbols:
        try:
            expr = parse_expr(expr_str, local_dict=sym_vars, transformations=transformations)
            if subs:
                # Ensure that subs is a dictionary
                if not isinstance(subs, dict):
                    raise ValueError(f"Substitutions must be a dictionary. Received: {subs}")
                subs_symbols = {symbols(k): v for k, v in subs.items()}
                expr = expr.subs(subs_symbols)
            return expr.expand()
        except (SyntaxError, ValueError, TypeError, AttributeError) as e:
            raise ValueError(f"Error parsing expression: {expr_str}. Error: {e}")

    variable_names = set(re.findall(r'\b[a-zA-Z]\w*\b', expression))
    sym_vars = {var: symbols(var) for var in variable_names}

    expr = parse_and_expand_expression(expression, sym_vars)
    latex_result = latex(expr)
    return latex_result


def factor_polynomial_expression(
    expression: str,
    subs: Optional[Dict[str, float]] = None
) -> str:
    """
    Factors a polynomial expression written in Python syntax and converts it to LaTeX format.

    This function accepts an algebraic expression in Python syntax, performs polynomial factoring,
    and translates the factored expression into a LaTeX formatted string.

    Parameters:
    expression (str): The algebraic expression to factor and convert to LaTeX.
    subs (Optional[Dict[str, float]]): An optional dictionary of substitutions to apply before factoring.

    Returns:
    str: The LaTeX formatted string of the factored expression.

    Raises:
    ValueError: If the expression cannot be parsed due to syntax errors.
    """
    transformations = standard_transformations + (implicit_multiplication_application,)

    def parse_and_factor_expression(expr_str: str, sym_vars: Dict[str, symbols]) -> symbols:
        try:
            expr = parse_expr(expr_str, local_dict=sym_vars, transformations=transformations)
            if subs:
                if not isinstance(subs, dict):
                    raise ValueError(f"Substitutions must be a dictionary. Received: {subs}")
                subs_symbols = {symbols(k): v for k, v in subs.items()}
                expr = expr.subs(subs_symbols)
            return factor(expr)
        except (SyntaxError, ValueError, TypeError, AttributeError) as e:
            raise ValueError(f"Error parsing expression: {expr_str}. Error: {e}")

    variable_names = set(re.findall(r'\b[a-zA-Z]\w*\b', expression))
    sym_vars = {var: symbols(var) for var in variable_names}

    expr = parse_and_factor_expression(expression, sym_vars)
    latex_result = latex(expr)
    return latex_result


def cancel_polynomial_expression(
    expression: str,
    subs: Optional[Dict[str, float]] = None
) -> str:
    """
    Cancels common factors within a polynomial expression and converts it to LaTeX format.

    This function parses an algebraic expression given in Python syntax, cancels any common factors,
    and converts the resulting simplified expression into a LaTeX formatted string. The function can
    also handle optional substitutions of variables before performing the cancellation.

    Parameters:
    expression (str): The algebraic expression to simplify and convert to LaTeX.
                      It should be a valid expression formatted using Python syntax.
    subs (Optional[Dict[str, float]]): An optional dictionary where the keys are variable names in the
                                       expression, and the values are the corresponding numbers to substitute
                                       into the expression before simplification.

    Returns:
    str: The LaTeX formatted string of the simplified expression. If the expression involves
         indeterminate forms due to operations like division by zero, a descriptive error message is returned instead.

    Raises:
    ValueError: If the expression cannot be parsed due to syntax errors or if operations result in
                undefined behavior, such as division by zero.

    """
    transformations = standard_transformations + (implicit_multiplication_application,)

    def parse_and_cancel_expression(expr_str: str, sym_vars: Dict[str, symbols]) -> symbols:
        try:
            expr = parse_expr(expr_str, local_dict=sym_vars, transformations=transformations)
            if subs:
                if not isinstance(subs, dict):
                    raise ValueError(f"Substitutions must be a dictionary. Received: {subs}")
                subs_symbols = {symbols(k): v for k, v in subs.items()}
                expr = expr.subs(subs_symbols)

            canceled_expr = cancel(expr)

            # Check for NaN or indeterminate forms
            if canceled_expr.has(S.NaN) or canceled_expr.has(S.Infinity) or canceled_expr.has(S.ComplexInfinity):
                return "Undefined result. This could be a division by zero error."

            return canceled_expr

        except (SyntaxError, ValueError, TypeError, AttributeError, ZeroDivisionError, SympifyError) as e:
            return f"Error: {str(e)}"

    variable_names = set(re.findall(r'\b[a-zA-Z]\w*\b', expression))
    sym_vars = {var: symbols(var) for var in variable_names}

    expr = parse_and_cancel_expression(expression, sym_vars)

    # If the expression is already a string (i.e., "Undefined" or error message), return it directly
    if isinstance(expr, str):
        return expr

    # Otherwise, convert to LaTeX as usual
    latex_result = latex(expr)
    return latex_result


def simplify_polynomial_expression(
    expression: str,
    subs: Optional[Dict[str, float]] = None
) -> str:
    """
    Simplifies an algebraic expression in polynomial form and returns it in LaTeX format.

    Takes an algebraic expression, in polynomial form, written in Python syntax and simplifies it.
    The result is returned as a LaTeX formatted string, suitable for academic or professional
    documentation.

    Parameters:
    expression (str): The algebraic expression, in polynomial form, to simplify. For instance,
                      the expression `np.diff(8*x**30)` is a polynomial, whereas np.diff([2,5,9,11)
                      is not a polynomial.
    subs (Optional[Dict[str, float]]): An optional dictionary of substitutions for variables
                                       in the expression.

    Returns:
    str: The simplified expression represented as a LaTeX string.

    Raises:
    ValueError: If the expression cannot be simplified due to errors in expression or parameters.
    """

    def recursive_parse_function_call(
            func_call: str, prefix: str, sym_vars: Dict[str, Expr]) -> Tuple[str, List[Expr]]:
        # print(f"Parsing function call: {func_call}")

        # Match the function name and arguments
        match = re.match(fr'{prefix}\.(\w+)\((.*)\)', func_call, re.DOTALL)
        if not match:
            raise ValueError(f"Invalid function call: {func_call}")

        func_name = match.group(1)
        args_str = match.group(2)

        # Check if it's a list for np
        if prefix == 'np' and args_str.startswith(
                "[") and args_str.endswith("]"):
            parsed_args = [ast.literal_eval(args_str.strip())]
        else:
            parsed_args = []
            raw_args = re.split(r',(?![^{]*\})', args_str)
            for arg in raw_args:
                arg = arg.strip()
                if re.match(r'\w+\.\w+\(', arg):
                    # Recursively evaluate the argument if it's another
                    # function call
                    arg_val = recursive_eval_func(
                        re.match(r'\w+\.\w+\(.*\)', arg), sym_vars)
                    parsed_args.append(
                        parse_expr(
                            arg_val,
                            local_dict=sym_vars))
                else:
                    parsed_args.append(parse_expr(arg, local_dict=sym_vars))

        # print(f"Function name: {func_name}, Parsed arguments: {parsed_args}")
        return func_name, parsed_args

    def recursive_eval_func(match: re.Match, sym_vars: Dict[str, Expr]) -> str:
        # print("152", match)
        func_call = match.group(0)
        # print(f"153 Evaluating function call: {func_call}")

        if func_call.startswith("np."):
            func_name, args = recursive_parse_function_call(
                func_call, 'np', sym_vars)
            if func_name == 'diff':
                expr = args[0]
                if isinstance(expr, list):
                    # Calculate discrete difference
                    diff_result = [expr[i] - expr[i - 1]
                                   for i in range(1, len(expr))]
                    return str(diff_result)
                # Perform symbolic differentiation
                diff_result = diff(expr)
                return str(diff_result)

        if func_call.startswith("math."):
            func_name, args = recursive_parse_function_call(
                func_call, 'math', sym_vars)
            if hasattr(math, func_name):
                result = getattr(math, func_name)(*args)
                return str(result)

        if func_call.startswith("sym."):
            initial_method_match = re.match(
                r'(sym\.\w+\([^()]*\))(\.(\w+)\((.*?)\))*', func_call, re.DOTALL)
            if initial_method_match:
                base_expr_str = initial_method_match.group(1)
                base_func_name, base_args = recursive_parse_function_call(
                    base_expr_str, 'sym', sym_vars)
                if base_func_name == 'solve':
                    solutions = solve(base_args[0], base_args[1])
                    # print(f"Solutions found: {solutions}")

                method_chain = re.findall(
                    r'\.(\w+)\((.*?)\)', func_call, re.DOTALL)
                final_solutions = [execute_chained_methods(sol, [(m, [method_args.strip(
                )]) for m, method_args in method_chain], sym_vars) for sol in solutions]

                return "[" + ",".join(latex(simplify(sol))
                                      for sol in final_solutions) + "]"

        raise ValueError(f"Unknown function call: {func_call}")

    def execute_chained_methods(sym_expr: Expr,
                                method_chain: List[Tuple[str,
                                                         List[str]]],
                                sym_vars: Dict[str,
                                               Expr]) -> Expr:
        for method_name, method_args in method_chain:
            # print(f"Executing method: {method_name} with arguments: {method_args}")
            method = getattr(sym_expr, method_name, None)
            if method:
                if method_name == 'subs' and isinstance(method_args[0], dict):
                    kwargs = method_args[0]
                    kwargs = {
                        parse_expr(
                            k,
                            local_dict=sym_vars): parse_expr(
                            v,
                            local_dict=sym_vars) for k,
                        v in kwargs.items()}
                    sym_expr = method(kwargs)
                else:
                    args = [parse_expr(arg.strip(), local_dict=sym_vars)
                            for arg in method_args]
                    sym_expr = method(*args)
            # print(f"Result after {method_name}: {sym_expr}")
        return sym_expr

    variable_names = set(re.findall(r'\b[a-zA-Z]\w*\b', expression))
    sym_vars = {var: symbols(var) for var in variable_names}

    patterns = {
        # "numpy_diff_brackets": r"np\.diff\(\[.*?\]\)",
        "numpy_diff_no_brackets": r"np\.diff\([^()]*\)",
        "math_functions": r"math\.\w+\((?:[^()]*(?:\([^()]*\)[^()]*)*)\)",
        # "sympy_functions": r"sym\.\w+\([^()]*\)(?:\.\w+\([^()]*\))?",
    }

    function_pattern = '|'.join(patterns.values())

    # Use a lambda function to pass additional arguments
    processed_expression = re.sub(
        function_pattern, lambda match: recursive_eval_func(
            match, sym_vars), expression)
    # print("Level 2 processed_expression:", processed_expression)

    try:
        # Parse the expression
        expr = parse_expr(processed_expression, local_dict=sym_vars)

        # Apply substitutions if provided
        if subs:
            subs_symbols = {symbols(k): v for k, v in subs.items()}
            expr = expr.subs(subs_symbols)

        # Simplify the expression
        final_result = simplify(expr)

        # Convert the result to LaTeX format
        if final_result.free_symbols:
            latex_result = latex(final_result)
            return latex_result
        else:
            return str(final_result)

    except Exception as e:
        raise ValueError(f"Error simplifying expression: {e}")


def solve_homogeneous_polynomial_expression(
    expression: str,
    variable: str,
    subs: Optional[Dict[str, float]] = None
) -> str:
    """
    Solves a homogeneous polynomial expression for a specified variable and returns solutions
    in LaTeX format.

    Assumes that the expression is homoegeneous (i.e. equal to zero), and solves for a
    designated variable. May optionally include substitutions for other variables in the
    equation. The solutions are provided as a LaTeX formatted string.

    Parameters:
    expression (str): The homogeneous polynomial expression to solve.
    variable (str): The variable to solve the equation for.
    subs (Optional[Dict[str, float]]): An optional dictionary of substitutions for variables
                                       in the equation.

    Returns:
    str: The solutions of the equation, formatted as a LaTeX string.

    Raises:
    ValueError: If the equation cannot be solved due to errors in expression or parameters.
    """

    try:
        # Create symbols for the variables in the expression
        variable_symbols = set(re.findall(r'\b[a-zA-Z]\w*\b', expression))
        sym_vars = {var: symbols(var) for var in variable_symbols}

        # Parse the expression and solve it
        expr = parse_expr(expression, local_dict=sym_vars)
        var_symbol = symbols(variable)
        solutions = solve(expr, var_symbol)

        # Apply substitutions if provided
        if subs:
            subs_symbols = {symbols(k): v for k, v in subs.items()}
            solutions = [simplify(sol.subs(subs_symbols)) for sol in solutions]

        # Convert solutions to LaTeX strings if possible
        latex_solutions = [
            latex(
                simplify(sol)) if sol.free_symbols else str(sol) for sol in solutions]
        result = r"\left[" + ", ".join(latex_solutions) + r"\right]"
        print("158", result)
        return result

    except Exception as e:
        raise ValueError(f"Error solving the expression: {e}")
