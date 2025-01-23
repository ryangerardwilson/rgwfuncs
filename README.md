# RGWFUNCS

***By Ryan Gerard Wilson (https://ryangerardwilson.com)***

This library is meant to make ML/ Data Science pipelines more readable. It assumes a linux environment, and the existence of a `.rgwfuncsrc` file for certain features (like db querying, sending data to slack, etc.)

--------------------------------------------------------------------------------

## Installation

Install the package using:

    pip install rgwfuncs

--------------------------------------------------------------------------------

## Create a `.rgwfuncsrc` File

A `.rgwfuncsrc` file (located at `vi ~/.rgwfuncsrc) is required for MSSQL, CLICKHOUSE, MYSQL, GOOGLE BIG QUERY, SLACK, TELEGRAM, and GMAIL integrations.

    {
      "db_presets" : [
        {
          "name": "mssql_db9",
          "db_type": "mssql",
          "host": "",
          "username": "",
          "password": "",
          "database": ""
        },
        {
          "name": "clickhouse_db7",
          "db_type": "clickhouse",
          "host": "",
          "username": "",
          "password": "",
          "database": ""
        },
        {
          "name": "mysql_db2",
          "db_type": "mysql",
          "host": "",
          "username": "",
          "password": "",
          "database": ""
        },
        {
          "name": "bq_db1",
          "db_type": "google_big_query",
          "json_file_path": "",
          "project_id": ""
        },
        {
          "name": "athena_db1",
          "db_type": "aws_athena",
          "aws_access_key": "",
          "aws_secret_key": "",
          "aws_region: "",
          "database": "logs",
          "output_bucket": "s3://bucket-name"
        }
      ],
      "vm_presets": [
        {
          "name": "main_server",
          "host": "",
          "ssh_user": "",
          "ssh_key_path": ""
        }
      ],
      "cloud_storage_presets": [
        {
          "name": "gcs_bucket_name",
          "credential_path": "/path/to/your/credentials.json"
        }
      ],
      "telegram_bot_presets": [
        {
          "name": "rgwml-bot",
          "chat_id": "",
          "bot_token": ""
        }
      ],
      "slack_bot_presets": [
        {
          "name": "labs-channel",
          "channel_id": "",
          "bot_token": ""
        }
      ],
      "gmail_bot_presets": [
        {
          "name": "info@xyz.com",
          "service_account_credentials_path": "/path/to/your/credentials.json"
        }
      ]
    }
    
--------------------------------------------------------------------------------

## Basic Usage

Import the library:

    import rgwfuncs

View available function docstrings in alphabetical order:
    
    rgwfuncs.docs()

View specific docstrings by providing a filter (comma-separated). For example, to display docstrings about "numeric_clean":
    
    rgwfuncs.docs(method_type_filter='numeric_clean')

To display all docstrings, use:
    
    rgwfuncs.docs(method_type_filter='*')

--------------------------------------------------------------------------------

## Documentation Access

### 1. docs
Print a list of available function names in alphabetical order. If a filter is provided, print the docstrings of functions containing the term.

• Parameters:
  - `method_type_filter` (str): Optional, comma-separated to select docstring types, or '*' for all.

• Example:

    from rgwfuncs import docs
    docs(method_type_filter='numeric_clean,limit_dataframe')

--------------------------------------------------------------------------------

## Interactive Shell

This section includes functions that facilitate launching an interactive Python shell to inspect and modify local variables within the user's environment.

### 1. `interactive_shell`

Launches an interactive prompt for inspecting and modifying local variables, making all methods in the rgwfuncs library available by default. This REPL (Read-Eval-Print Loop) environment supports command history and autocompletion, making it easier to interact with your Python code. This function is particularly useful for debugging purposes when you want real-time interaction with your program's execution environment.

• Parameters:
  - `local_vars` (dict, optional): A dictionary of local variables to be accessible within the interactive shell. If not provided, defaults to an empty dictionary.

• Usage:
  - You can call this function to enter an interactive shell where you can view and modify the variables in the given local scope.

• Example:

    from rgwfuncs import interactive_shell
    import pandas as pd
    import numpy as np

    # Example DataFrame
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'age': [30, 25, 35, 28, 22],
        'city': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Boston']
    })

    # Launch the interactive shell with local variables
    interactive_shell(locals())

Subsequently, in the interactive shell you can use any library in your python file, as well as all rgwfuncs methods (even if they are not imported). Notice, that while pandas and numpy are available in the shell as a result of importing them in the above script, the rgwfuncs method `first_n_rows` was not imported - yet is available for use.

    Welcome to the rgwfuncs interactive shell.
    >>> pirst_n_rows(df, 2)
    Traceback (most recent call last):
      File "<console>", line 1, in <module>
    NameError: name 'pirst_n_rows' is not defined. Did you mean: 'first_n_rows'?
    >>> first_n_rows(df, 2)
    {'age': '30', 'city': 'New York', 'id': '1', 'name': 'Alice'}
    {'age': '25', 'city': 'Los Angeles', 'id': '2', 'name': 'Bob'}
    >>> print(df)
      id     name age           city
    0  1    Alice  30       New York
    1  2      Bob  25    Los Angeles
    2  3  Charlie  35        Chicago
    3  4    David  28  San Francisco
    4  5      Eva  22         Boston
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> arr
    array([1, 2, 3, 4, 5])

--------------------------------------------------------------------------------

## Algebra Based Functions

This section provides comprehensive functions for handling algebraic expressions, performing tasks such as computation, simplification, solving equations, and prime factorization, all outputted in LaTeX format.

### 1. `compute_prime_factors`

Computes prime factors of a number and presents them in LaTeX format.

• Parameters:
  - `n` (int): The integer to factorize.

• Returns:
  - `str`: Prime factorization in LaTeX.

• Example:

    from rgwfuncs import compute_prime_factors
    factors_1 = compute_prime_factors(100)
    print(factors_1)  # Output: "2^{2} \cdot 5^{2}"

    factors_2 = compute_prime_factors(60)
    print(factors_2)  # Output: "2^{2} \cdot 3 \cdot 5"

    factors_3 = compute_prime_factors(17)
    print(factors_3)  # Output: "17"

--------------------------------------------------------------------------------

### 2. `compute_constant_expression`

Computes the numerical result of a given expression, which can evaluate to a constant, represented as a float. Evaluates an constant expression provided as a string and returns the computed result. Supports various arithmetic operations, including addition, subtraction, multiplication, division, and modulo, as well as mathematical functions from the math module.

• Parameters:
  - `expression` (str): The constant expression to compute. This should be a string consisting of arithmetic operations and Python's math module functions.

• Returns:
  - `float`: The computed numerical result.

• Example:

    from rgwfuncs import compute_constant_expression
    result1 = compute_constant_expression("2 + 2")
    print(result1)  # Output: 4.0

    result2 = compute_constant_expression("10 % 3")
    print(result2)  # Output: 1.0

    result3 = compute_constant_expression("math.gcd(36, 60) * math.sin(math.radians(45)) * 10000")
    print(result3)  # Output: 84852.8137423857

--------------------------------------------------------------------------------

### 3. `compute_constant_expression_involving_matrices`

Computes the result of a constant expression involving matrices and returns it as a LaTeX string.

• Parameters:
  - `expression` (str): The constant expression involving matrices. Example format includes operations such as "+", "-", "*", "/".

• Returns:
  - `str`: The LaTeX-formatted string representation of the computed matrix, or an error message if the operations cannot be performed due to dimensional mismatches.

• Example:

    from rgwfuncs import compute_constant_expression_involving_matrices

    # Example with addition of 2D matrices
    result = compute_constant_expression_involving_matrices("[[2, 6, 9], [1, 3, 5]] + [[1, 2, 3], [4, 5, 6]]")
    print(result)  # Output: \begin{bmatrix}3 & 8 & 12\\5 & 8 & 11\end{bmatrix}

    # Example of mixed operations with 1D matrices treated as 2D
    result = compute_constant_expression_involving_matrices("[3, 6, 9] + [1, 2, 3] - [2, 2, 2]")
    print(result)  # Output: \begin{bmatrix}2 & 6 & 10\end{bmatrix}

    # Example with dimension mismatch
    result = compute_constant_expression_involving_matrices("[[4, 3, 51]] + [[1, 1]]")
    print(result)  # Output: Operations between matrices must involve matrices of the same dimension

--------------------------------------------------------------------------------

### 4. `compute_constant_expression_involving_ordered_series`

Computes the result of a constant expression involving ordered series, and returns it as a Latex string. 


• Parameters:
  - `expression` (str): A series operation expression. Supports operations such as "+", "-", "*", "/", and `dd()` for discrete differences.

• Returns:
  - `str`: The string representation of the resultant series after performing operations, or an error message if series lengths do not match.

• Example:

    from rgwfuncs import compute_constant_expression_involving_ordered_series

    # Example with addition and discrete differences
    result = compute_constant_expression_involving_ordered_series("dd([2, 6, 9, 60]) + dd([78, 79, 80])")
    print(result)  # Output: [4, 3, 51] + [1, 1]

    # Example with elementwise subtraction
    result = compute_constant_expression_involving_ordered_series("[10, 15, 21] - [5, 5, 5]")
    print(result)  # Output: [5, 10, 16]

    # Example with length mismatch
    result = compute_constant_expression_involving_ordered_series("[4, 3, 51] + [1, 1]")
    print(result)  # Output: Operations between ordered series must involve series of equal length

--------------------------------------------------------------------------------

### 5. `python_polynomial_expression_to_latex`

Converts a polynomial expression written in Python syntax to a LaTeX formatted string. This function parses algebraic expressions provided as strings using Python’s syntax and translates them into equivalent LaTeX representations, making them suitable for academic or professional documentation. The function supports inclusion of named variables, with an option to substitute specific values into the expression.

• Parameters:
  - `expression` (str): The algebraic expression to convert to LaTeX. This should be a string formatted with Python syntax acceptable by SymPy.
  - `subs` (Optional[Dict[str, float]]): An optional dictionary of substitutions where the keys are variable names in the expression, and the values are the numbers with which to substitute those variables.

• Returns:
  - `str`: The LaTeX formatted string equivalent to the provided expression.

• Raises:
  - `ValueError`: If the expression cannot be parsed due to syntax errors.

• Example:

    from rgwfuncs import python_polynomial_expression_to_latex
    
    # Convert a simple polynomial expression to LaTeX format
    latex_result1 = python_polynomial_expression_to_latex("x**2 + y**2")
    print(latex_result1)  # Output: "x^{2} + y^{2}"
    
    # Convert polynomial expression with substituted values
    latex_result2 = python_polynomial_expression_to_latex("x**2 + y**2", {"x": 3, "y": 4})
    print(latex_result2)  # Output: "25"
    
    # Another example with partial substitution
    latex_result3 = python_polynomial_expression_to_latex("x**2 + y**2", {"x": 3})
    print(latex_result3)  # Output: "y^{2} + 9"
    
    # Trigonometric functions included with symbolic variables
    latex_result4 = python_polynomial_expression_to_latex("sin(x+z**2) + cos(y)", {"x": 55})
    print(latex_result4)  # Output: "cos y + sin \\left(z^{2} + 55\\right)"
    
    # Simplified trigonometric functions example with substitution
    latex_result5 = python_polynomial_expression_to_latex("sin(x) + cos(y)", {"x": 0})
    print(latex_result5)  # Output: "cos y"

--------------------------------------------------------------------------------

### 6. `simplify_polynomial_expression`

Simplifies an algebraic expression in polynomial form and returns it in LaTeX format. Takes an algebraic expression, in polynomial form, written in Python syntax and simplifies it. The result is returned as a LaTeX formatted string, suitable for academic or professional documentation.

• Parameters:
  - `expression` (str): The algebraic expression, in polynomial form, to simplify. For instance, the expression 'np.diff(8*x**30) where as 'np.diff([2,5,9,11)' is not a polynomial.
  - `subs` (Optional[Dict[str, float]]): An optional dictionary of substitutions where keys are variable names and values are the numbers to substitute them with.

• Returns:
  - `str`: The simplified expression formatted as a LaTeX string.

• Example Usage:

    from rgwfuncs import simplify_polynomial_expression

    # Example 1: Simplifying a polynomial expression without substitutions
    simplified_expr1 = simplify_polynomial_expression("2*x + 3*x")
    print(simplified_expr1)  # Output: "5 x"

    # Example 2: Simplifying a complex expression involving derivatives
    simplified_expr2 = simplify_polynomial_expression("(np.diff(3*x**8)) / (np.diff(8*x**30) * 11*y**3)")
    print(simplified_expr2)  # Output: r"\frac{1}{110 x^{22} y^{3}}"

    # Example 3: Simplifying with substitutions
    simplified_expr3 = simplify_polynomial_expression("x**2 + y**2", subs={"x": 3, "y": 4})
    print(simplified_expr3)  # Output: "25"

    # Example 4: Simplifying with partial substitution
    simplified_expr4 = simplify_polynomial_expression("a*b + b", subs={"b": 2})
    print(simplified_expr4)  # Output: "2 a + 2"

--------------------------------------------------------------------------------

### 7. `solve_homogeneous_polynomial_expression`

Solves a homogeneous polynomial expression for a specified variable and returns solutions in LaTeX format. Assumes that the expression is homoegeneous (i.e. equal to zero), and solves for a designated variable. May optionally include substitutions for other variables in the equation. The solutions are provided as a LaTeX formatted string. The method solves equations for specified variables, with optional substitutions, returning LaTeX-formatted solutions.

• Parameters:
  - `expression` (str): A string of the homogeneous polynomial expression to solve.
  - `variable` (str): The variable to solve for.
  - `subs` (Optional[Dict[str, float]]): Substitutions for variables.

• Returns:
  - `str`: Solutions formatted in LaTeX.

• Example:

    from rgwfuncs import solve_homogeneous_polynomial_expression
    solutions1 = solve_homogeneous_polynomial_expression("a*x**2 + b*x + c", "x", {"a": 3, "b": 7, "c": 5})
    print(solutions1)  # Output: "\left[-7/6 - sqrt(11)*I/6, -7/6 + sqrt(11)*I/6\right]"

    solutions2 = solve_homogeneous_polynomial_expression("x**2 - 4", "x")
    print(solutions2)  # Output: "\left[-2, 2\right]"
    
--------------------------------------------------------------------------------

## String Based Functions

### 1. send_telegram_message

Send a message to a Telegram chat using a specified preset from your configuration file.

• Parameters:
  - `preset_name` (str): The name of the preset to use for sending the message. This should match a preset in the configuration file.
  - `message` (str): The message text that you want to send to the Telegram chat.

• Raises:
  - `RuntimeError`: If the preset is not found in the configuration file or if necessary details (bot token or chat ID) are missing.

• Example:

    from rgwfuncs import send_telegram_message
    
    preset_name = "daily_updates"
    message = "Here is your daily update!"

    send_telegram_message(preset_name, message)

--------------------------------------------------------------------------------

## Dataframe Based Functions

Below is a quick reference of available functions, their purpose, and basic usage examples.

### 1. `numeric_clean`
Cleans the numeric columns in a DataFrame according to specified treatments.

• Parameters:
  - df (pd.DataFrame): The DataFrame to clean.
  - `column_names` (str): A comma-separated string containing the names of the columns to clean.
  - `column_type` (str): The type to convert the column to (`INTEGER` or `FLOAT`).
  - `irregular_value_treatment` (str): How to treat irregular values (`NAN`, `TO_ZERO`, `MEAN`).

• Returns:
  - pd.DataFrame: A new DataFrame with cleaned numeric columns.

• Example:

    from rgwfuncs import numeric_clean
    import pandas as pd

    # Sample DataFrame
    df = pd.DataFrame({
        'col1': [1, 2, 3, 'x', 4],
        'col2': [10.5, 20.1, 'not_a_number', 30.2, 40.8]
    })

    # Clean numeric columns
    df_cleaned = numeric_clean(df, 'col1,col2', 'FLOAT', 'MEAN')
    print(df_cleaned)

--------------------------------------------------------------------------------

### 2. `limit_dataframe`
Limit the DataFrame to a specified number of rows.

• Parameters:
  - df (pd.DataFrame): The DataFrame to limit.
  - `num_rows` (int): The number of rows to retain.

• Returns:
  - pd.DataFrame: A new DataFrame limited to the specified number of rows.

• Example:
    
    from rgwfuncs import limit_dataframe
    import pandas as pd

    df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})
    df_limited = limit_dataframe(df, 5)
    print(df_limited)

--------------------------------------------------------------------------------

### 3. `from_raw_data`
Create a DataFrame from raw data.

• Parameters:
  - headers (list): A list of column headers.
  - data (list of lists): A two-dimensional list of data.

• Returns:
  - pd.DataFrame: A DataFrame created from the raw data.

• Example:
    
    from rgwfuncs import from_raw_data

    headers = ["Name", "Age"]
    data = [
        ["Alice", 30],
        ["Bob", 25],
        ["Charlie", 35]
    ]

    df = from_raw_data(headers, data)
    print(df)

--------------------------------------------------------------------------------

### 4. `append_rows`
Append rows to the DataFrame.

• Parameters:
  - df (pd.DataFrame): The original DataFrame.
  - rows (list of lists): Each inner list represents a row to be appended.

• Returns:
  - pd.DataFrame: A new DataFrame with appended rows.

• Example:
    
    from rgwfuncs import append_rows
    import pandas as pd

    df = pd.DataFrame({'Name': ['Alice'], 'Age': [30]})
    new_rows = [
        ['Bob', 25],
        ['Charlie', 35]
    ]
    df_appended = append_rows(df, new_rows)
    print(df_appended)

--------------------------------------------------------------------------------

### 5. `append_columns`
Append new columns to the DataFrame with None values.

• Parameters:
  - df (pd.DataFrame): The original DataFrame.
  - `col_names` (list or comma-separated string): The names of the columns to add.

• Returns:
  - pd.DataFrame: A new DataFrame with the new columns appended.

• Example:
    
    from rgwfuncs import append_columns
    import pandas as pd

    df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [30, 25]})
    df_new = append_columns(df, ['Salary', 'Department'])
    print(df_new)
    
--------------------------------------------------------------------------------

### 6. `update_rows`
Update specific rows in the DataFrame based on a condition.

• Parameters:
  - df (pd.DataFrame): The original DataFrame.
  - condition (str): A query condition to identify rows for updating.
  - updates (dict): A dictionary with column names as keys and new values as values.

• Returns:
  - pd.DataFrame: A new DataFrame with updated rows.

• Example:
    
    from rgwfuncs import update_rows
    import pandas as pd

    df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [30, 25]})
    df_updated = update_rows(df, "Name == 'Alice'", {'Age': 31})
    print(df_updated)
    
--------------------------------------------------------------------------------

### 7. `delete_rows`
Delete rows from the DataFrame based on a condition.

• Parameters:
  - df (pd.DataFrame): The original DataFrame.
  - condition (str): A query condition to identify rows for deletion.

• Returns:
  - pd.DataFrame: The DataFrame with specified rows deleted.

• Example:
    
    from rgwfuncs import delete_rows
    import pandas as pd

    df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [30, 25]})
    df_deleted = delete_rows(df, "Age < 28")
    print(df_deleted)
    
--------------------------------------------------------------------------------

### 8. `drop_duplicates`
Drop duplicate rows in the DataFrame, retaining the first occurrence.

• Parameters:
  - df (pd.DataFrame): The DataFrame from which duplicates will be dropped.

• Returns:
  - pd.DataFrame: A new DataFrame with duplicates removed.

• Example:
    
    from rgwfuncs import drop_duplicates
    import pandas as pd

    df = pd.DataFrame({'A': [1,1,2,2], 'B': [3,3,4,4]})
    df_no_dupes = drop_duplicates(df)
    print(df_no_dupes)
    
--------------------------------------------------------------------------------

### 9. `drop_duplicates_retain_first`
Drop duplicate rows based on specified columns, retaining the first occurrence.

• Parameters:
  - df (pd.DataFrame): The DataFrame from which duplicates will be dropped.
  - columns (str): Comma-separated string with column names used to identify duplicates.

• Returns:
  - pd.DataFrame: A new DataFrame with duplicates removed.

• Example:
    
    from rgwfuncs import drop_duplicates_retain_first
    import pandas as pd

    df = pd.DataFrame({'A': [1,1,2,2], 'B': [3,3,4,4]})
    df_no_dupes = drop_duplicates_retain_first(df, 'A')
    print(df_no_dupes)
    
--------------------------------------------------------------------------------

### 10. `drop_duplicates_retain_last`
Drop duplicate rows based on specified columns, retaining the last occurrence.

• Parameters:
  - df (pd.DataFrame): The DataFrame from which duplicates will be dropped.
  - columns (str): Comma-separated string with column names used to identify duplicates.

• Returns:
  - pd.DataFrame: A new DataFrame with duplicates removed.

• Example:
    
    from rgwfuncs import drop_duplicates_retain_last
    import pandas as pd

    df = pd.DataFrame({'A': [1,1,2,2], 'B': [3,3,4,4]})
    df_no_dupes = drop_duplicates_retain_last(df, 'A')
    print(df_no_dupes)
    

--------------------------------------------------------------------------------

### 11. `load_data_from_query`

Load data from a specified database using a SQL query and return the results in a Pandas DataFrame. The database connection configurations are determined by a preset name specified in a configuration file.

#### Features

- Multi-Database Support: This function supports different database types, including MSSQL, MySQL, ClickHouse, Google BigQuery, and AWS Athena, based on the configuration preset selected.
- Configuration-Based: It utilizes a configuration file to store database connection details securely, avoiding hardcoding sensitive information directly into the script.
- Dynamic Query Execution: Capable of executing custom user-defined SQL queries against the specified database.
- Automatic Result Loading: Fetches query results and loads them directly into a Pandas DataFrame for further manipulation and analysis.

#### Parameters

- `db_preset_name` (str): The name of the database preset found in the configuration file. This preset determines which database connection details to use.
- `query` (str): The SQL query string to be executed on the database.

#### Returns

- `pd.DataFrame`: Returns a DataFrame that contains the results from the executed SQL query.

#### Configuration Details

- The configuration file is expected to be in JSON format and located at `~/.rgwfuncsrc`.
- Each preset within the configuration file must include:
  - `name`: Name of the database preset.
  - `db_type`: Type of the database (`mssql`, `mysql`, `clickhouse`, `google_big_query`, `aws_athena`).
  - `credentials`: Necessary credentials such as host, username, password, and potentially others depending on the database type.

#### Example

    from rgwfuncs import load_data_from_query

    # Load data using a preset configuration
    df = load_data_from_query(
        db_preset_name="MyDBPreset",
        query="SELECT * FROM my_table"
    )
    print(df)

#### Notes

- Security: Ensure that the configuration file (`~/.rgwfuncsrc`) is secure and accessible only to authorized users, as it contains sensitive information.
- Pre-requisites: Ensure the necessary Python packages are installed for each database type you wish to query. For example, `pymssql` for MSSQL, `mysql-connector-python` for MySQL, and so on.
- Error Handling: The function raises a `ValueError` if the specified preset name does not exist or if the database type is unsupported. Additional exceptions may arise from network issues or database errors.
- Environment: For AWS Athena, ensure that AWS credentials are configured properly for the boto3 library to authenticate successfully. Consider using AWS IAM roles or AWS Secrets Manager for better security management.
    
--------------------------------------------------------------------------------

### 12. `load_data_from_path`
Load data from a file into a DataFrame based on the file extension.

• Parameters:
  - `file_path` (str): The absolute path to the data file.

• Returns:
  - pd.DataFrame: A DataFrame containing the loaded data.

• Example:
    
    from rgwfuncs import load_data_from_path

    df = load_data_from_path("/absolute/path/to/data.csv")
    print(df)
    

--------------------------------------------------------------------------------

### 13. `load_data_from_sqlite_path`
Execute a query on a SQLite database file and return the results as a DataFrame.

• Parameters:
  - `sqlite_path` (str): The absolute path to the SQLite database file.
  - query (str): The SQL query to execute.

• Returns:
  - pd.DataFrame: A DataFrame containing the query results.

• Example:
    
    from rgwfuncs import load_data_from_sqlite_path

    df = load_data_from_sqlite_path("/path/to/database.db", "SELECT * FROM my_table")
    print(df)
    

--------------------------------------------------------------------------------

### 14. `first_n_rows`
Display the first n rows of the DataFrame (prints out in dictionary format).

• Parameters:
  - df (pd.DataFrame)
  - n (int): Number of rows to display.

• Example:
    
    from rgwfuncs import first_n_rows
    import pandas as pd

    df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
    first_n_rows(df, 2)
    

--------------------------------------------------------------------------------

### 15. `last_n_rows`
Display the last n rows of the DataFrame (prints out in dictionary format).

• Parameters:
  - df (pd.DataFrame)
  - n (int): Number of rows to display.

• Example:
    
    from rgwfuncs import last_n_rows
    import pandas as pd

    df = pd.DataFrame({'A': [1,2,3,4,5], 'B': [6,7,8,9,10]})
    last_n_rows(df, 2)
    

--------------------------------------------------------------------------------

### 16. `top_n_unique_values`
Print the top n unique values for specified columns in the DataFrame.

• Parameters:
  - df (pd.DataFrame): The DataFrame to evaluate.
  - n (int): Number of top values to display.
  - columns (list): List of columns for which to display top unique values.

• Example:
    
    from rgwfuncs import top_n_unique_values
    import pandas as pd

    df = pd.DataFrame({'Cities': ['NY', 'LA', 'NY', 'SF', 'LA', 'LA']})
    top_n_unique_values(df, 2, ['Cities'])
    

--------------------------------------------------------------------------------

### 17. `bottom_n_unique_values`
Print the bottom n unique values for specified columns in the DataFrame.

• Parameters:
  - df (pd.DataFrame)
  - n (int)
  - columns (list)

• Example:
    
    from rgwfuncs import bottom_n_unique_values
    import pandas as pd

    df = pd.DataFrame({'Cities': ['NY', 'LA', 'NY', 'SF', 'LA', 'LA']})
    bottom_n_unique_values(df, 1, ['Cities'])
    

--------------------------------------------------------------------------------

### 18. `print_correlation`
Print correlation for multiple pairs of columns in the DataFrame.

• Parameters:
  - df (pd.DataFrame)
  - `column_pairs` (list of tuples): E.g., `[('col1','col2'), ('colA','colB')]`.

• Example:
    
    from rgwfuncs import print_correlation
    import pandas as pd

    df = pd.DataFrame({
        'col1': [1,2,3,4,5],
        'col2': [2,4,6,8,10],
        'colA': [10,9,8,7,6],
        'colB': [5,4,3,2,1]
    })

    pairs = [('col1','col2'), ('colA','colB')]
    print_correlation(df, pairs)
    

--------------------------------------------------------------------------------

### 19. `print_memory_usage`
Print the memory usage of the DataFrame in megabytes.

• Parameters:
  - df (pd.DataFrame)

• Example:
    
    from rgwfuncs import print_memory_usage
    import pandas as pd

    df = pd.DataFrame({'A': range(1000)})
    print_memory_usage(df)
    

--------------------------------------------------------------------------------

### 20. `filter_dataframe`
Return a new DataFrame filtered by a given query expression.

• Parameters:
  - df (pd.DataFrame)
  - `filter_expr` (str)

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import filter_dataframe
    import pandas as pd

    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [30, 20, 25]
    })

    df_filtered = filter_dataframe(df, "Age > 23")
    print(df_filtered)
    

--------------------------------------------------------------------------------

### 21. `filter_indian_mobiles`
Filter and return rows containing valid Indian mobile numbers in the specified column.

• Parameters:
  - df (pd.DataFrame)
  - `mobile_col` (str): The column name with mobile numbers.

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import filter_indian_mobiles
    import pandas as pd

    df = pd.DataFrame({'Phone': ['9876543210', '12345', '7000012345']})
    df_indian = filter_indian_mobiles(df, 'Phone')
    print(df_indian)
    

--------------------------------------------------------------------------------

### 22. `print_dataframe`
Print the entire DataFrame and its column types. Optionally print a source path.

• Parameters:
  - df (pd.DataFrame)
  - source (str, optional)

• Example:
    
    from rgwfuncs import print_dataframe
    import pandas as pd

    df = pd.DataFrame({'Name': ['Alice'], 'Age': [30]})
    print_dataframe(df, source='SampleData.csv')
    

--------------------------------------------------------------------------------

### 23. `send_dataframe_via_telegram`
Send a DataFrame via Telegram using a specified bot configuration.

• Parameters:
  - df (pd.DataFrame)
  - `bot_name` (str)
  - message (str)
  - `as_file` (bool)
  - `remove_after_send` (bool)

• Example:
    
    from rgwfuncs import send_dataframe_via_telegram

    # Suppose your bot config is in "rgwml.config" under [TelegramBots] section
    df = ...  # Some DataFrame
    send_dataframe_via_telegram(
        df,
        bot_name='MyTelegramBot',
        message='Hello from RGWFuncs!',
        as_file=True,
        remove_after_send=True
    )
    

--------------------------------------------------------------------------------

### 24. `send_data_to_email`
Send an email with an optional DataFrame attachment using the Gmail API via a specified preset.

• Parameters:
  - df (pd.DataFrame)
  - `preset_name` (str)
  - `to_email` (str)
  - subject (str, optional)
  - body (str, optional)
  - `as_file` (bool)
  - `remove_after_send` (bool)

• Example:
    
    from rgwfuncs import send_data_to_email

    df = ...  # Some DataFrame
    send_data_to_email(
        df,
        preset_name='MyEmailPreset',
        to_email='recipient@example.com',
        subject='Hello from RGWFuncs',
        body='Here is the data you requested.',
        as_file=True,
        remove_after_send=True
    )
    

--------------------------------------------------------------------------------

### 25. `send_data_to_slack`
Send a DataFrame or message to Slack using a specified bot configuration.

• Parameters:
  - df (pd.DataFrame)
  - `bot_name` (str)
  - message (str)
  - `as_file` (bool)
  - `remove_after_send` (bool)

• Example:
    
    from rgwfuncs import send_data_to_slack

    df = ...  # Some DataFrame
    send_data_to_slack(
        df,
        bot_name='MySlackBot',
        message='Hello Slack!',
        as_file=True,
        remove_after_send=True
    )
    

--------------------------------------------------------------------------------

### 26. `order_columns`
Reorder the columns of a DataFrame based on a string input.

• Parameters:
  - df (pd.DataFrame)
  - `column_order_str` (str): Comma-separated column order.

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import order_columns
    import pandas as pd

    df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [30, 25], 'Salary': [1000, 1200]})
    df_reordered = order_columns(df, 'Salary,Name,Age')
    print(df_reordered)
    

--------------------------------------------------------------------------------

### 27. `append_ranged_classification_column`
Append a ranged classification column to the DataFrame.

• Parameters:
  - df (pd.DataFrame)
  - ranges (str): Ranges separated by commas (e.g., "0-10,11-20,21-30").
  - `target_col` (str): The column to classify.
  - `new_col_name` (str): Name of the new classification column.

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import append_ranged_classification_column
    import pandas as pd

    df = pd.DataFrame({'Scores': [5, 12, 25]})
    df_classified = append_ranged_classification_column(df, '0-10,11-20,21-30', 'Scores', 'ScoreRange')
    print(df_classified)
    

--------------------------------------------------------------------------------

### 28. `append_percentile_classification_column`
Append a percentile classification column to the DataFrame.

• Parameters:
  - df (pd.DataFrame)
  - percentiles (str): Percentile values separated by commas (e.g., "25,50,75").
  - `target_col` (str)
  - `new_col_name` (str)

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import append_percentile_classification_column
    import pandas as pd

    df = pd.DataFrame({'Values': [10, 20, 30, 40, 50]})
    df_classified = append_percentile_classification_column(df, '25,50,75', 'Values', 'ValuePercentile')
    print(df_classified)
    

--------------------------------------------------------------------------------

### 29. `append_ranged_date_classification_column`
Append a ranged date classification column to the DataFrame.

• Parameters:
  - df (pd.DataFrame)
  - `date_ranges` (str): Date ranges separated by commas, e.g., `2020-01-01_2020-06-30,2020-07-01_2020-12-31`
  - `target_col` (str)
  - `new_col_name` (str)

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import append_ranged_date_classification_column
    import pandas as pd

    df = pd.DataFrame({'EventDate': pd.to_datetime(['2020-03-15','2020-08-10'])})
    df_classified = append_ranged_date_classification_column(
        df,
        '2020-01-01_2020-06-30,2020-07-01_2020-12-31',
        'EventDate',
        'DateRange'
    )
    print(df_classified)
    

--------------------------------------------------------------------------------

### 30. `rename_columns`
Rename columns in the DataFrame.

• Parameters:
  - df (pd.DataFrame)
  - `rename_pairs` (dict): Mapping old column names to new ones.

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import rename_columns
    import pandas as pd

    df = pd.DataFrame({'OldName': [1,2,3]})
    df_renamed = rename_columns(df, {'OldName': 'NewName'})
    print(df_renamed)
    

--------------------------------------------------------------------------------

### 31. `cascade_sort`
Cascade sort the DataFrame by specified columns and order.

• Parameters:
  - df (pd.DataFrame)
  - columns (list): e.g. ["Column1::ASC", "Column2::DESC"].

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import cascade_sort
    import pandas as pd

    df = pd.DataFrame({
        'Name': ['Charlie', 'Alice', 'Bob'],
        'Age': [25, 30, 22]
    })

    sorted_df = cascade_sort(df, ["Name::ASC", "Age::DESC"])
    print(sorted_df)
    

--------------------------------------------------------------------------------

### 32. `append_xgb_labels`
Append XGB training labels (TRAIN, VALIDATE, TEST) based on a ratio string.

• Parameters:
  - df (pd.DataFrame)
  - `ratio_str` (str): e.g. "8:2", "7:2:1".

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import append_xgb_labels
    import pandas as pd

    df = pd.DataFrame({'A': range(10)})
    df_labeled = append_xgb_labels(df, "7:2:1")
    print(df_labeled)
    

--------------------------------------------------------------------------------

### 33. `append_xgb_regression_predictions`
Append XGB regression predictions to the DataFrame. Requires an `XGB_TYPE` column for TRAIN/TEST splits.

• Parameters:
  - df (pd.DataFrame)
  - `target_col` (str)
  - `feature_cols` (str): Comma-separated feature columns.
  - `pred_col` (str)
  - `boosting_rounds` (int, optional)
  - `model_path` (str, optional)

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import append_xgb_regression_predictions
    import pandas as pd

    df = pd.DataFrame({
        'XGB_TYPE': ['TRAIN','TRAIN','TEST','TEST'],
        'Feature1': [1.2, 2.3, 3.4, 4.5],
        'Feature2': [5.6, 6.7, 7.8, 8.9],
        'Target': [10, 20, 30, 40]
    })

    df_pred = append_xgb_regression_predictions(df, 'Target', 'Feature1,Feature2', 'PredictedTarget')
    print(df_pred)
    

--------------------------------------------------------------------------------

### 34. `append_xgb_logistic_regression_predictions`
Append XGB logistic regression predictions to the DataFrame. Requires an `XGB_TYPE` column for TRAIN/TEST splits.

• Parameters:
  - df (pd.DataFrame)
  - `target_col` (str)
  - `feature_cols` (str)
  - `pred_col` (str)
  - `boosting_rounds` (int, optional)
  - `model_path` (str, optional)

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import append_xgb_logistic_regression_predictions
    import pandas as pd

    df = pd.DataFrame({
        'XGB_TYPE': ['TRAIN','TRAIN','TEST','TEST'],
        'Feature1': [1, 0, 1, 0],
        'Feature2': [0.5, 0.2, 0.8, 0.1],
        'Target': [1, 0, 1, 0]
    })

    df_pred = append_xgb_logistic_regression_predictions(df, 'Target', 'Feature1,Feature2', 'PredictedTarget')
    print(df_pred)
    

--------------------------------------------------------------------------------

### 35. `print_n_frequency_cascading`
Print the cascading frequency of top n values for specified columns.

• Parameters:
  - df (pd.DataFrame)
  - n (int)
  - columns (str): Comma-separated column names.
  - `order_by` (str): `ASC`, `DESC`, `FREQ_ASC`, `FREQ_DESC`.

• Example:
    
    from rgwfuncs import print_n_frequency_cascading
    import pandas as pd

    df = pd.DataFrame({'City': ['NY','LA','NY','SF','LA','LA']})
    print_n_frequency_cascading(df, 2, 'City', 'FREQ_DESC')
    

--------------------------------------------------------------------------------

### 36. `print_n_frequency_linear`

Prints the linear frequency of the top `n` values for specified columns.

#### Parameters:
- **df** (`pd.DataFrame`): The DataFrame to analyze.
- **n** (`int`): The number of top values to print for each column.
- **columns** (`list`): A list of column names to be analyzed.
- **order_by** (`str`): The order of frequency. The available options are:
  - `"ASC"`: Sort keys in ascending lexicographical order.
  - `"DESC"`: Sort keys in descending lexicographical order.
  - `"FREQ_ASC"`: Sort the frequencies in ascending order (least frequent first).
  - `"FREQ_DESC"`: Sort the frequencies in descending order (most frequent first).
  - `"BY_KEYS_ASC"`: Sort keys in ascending order, numerically if possible, handling special strings like 'NaN' as typical entries.
  - `"BY_KEYS_DESC"`: Sort keys in descending order, numerically if possible, handling special strings like 'NaN' as typical entries.

#### Example:

    from rgwfuncs import print_n_frequency_linear
    import pandas as pd

    df = pd.DataFrame({'City': ['NY', 'LA', 'NY', 'SF', 'LA', 'LA']})
    print_n_frequency_linear(df, 2, ['City'], 'FREQ_DESC')

This example analyzes the `City` column, printing the top 2 most frequent values in descending order of frequency.


--------------------------------------------------------------------------------

### 37. `retain_columns`
Retain specified columns in the DataFrame and drop the others.

• Parameters:
  - df (pd.DataFrame)
  - `columns_to_retain` (list or str)

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import retain_columns
    import pandas as pd

    df = pd.DataFrame({'A': [1,2], 'B': [3,4], 'C': [5,6]})
    df_reduced = retain_columns(df, ['A','C'])
    print(df_reduced)
    

--------------------------------------------------------------------------------

### 38. `mask_against_dataframe`
Retain only rows with common column values between two DataFrames.

• Parameters:
  - df (pd.DataFrame)
  - `other_df` (pd.DataFrame)
  - `column_name` (str)

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import mask_against_dataframe
    import pandas as pd

    df1 = pd.DataFrame({'ID': [1,2,3], 'Value': [10,20,30]})
    df2 = pd.DataFrame({'ID': [2,3,4], 'Extra': ['X','Y','Z']})

    df_masked = mask_against_dataframe(df1, df2, 'ID')
    print(df_masked)
    

--------------------------------------------------------------------------------

### 39. `mask_against_dataframe_converse`
Retain only rows with uncommon column values between two DataFrames.

• Parameters:
  - df (pd.DataFrame)
  - `other_df` (pd.DataFrame)
  - `column_name` (str)

• Returns:
  - pd.DataFrame

• Example:
    
    from rgwfuncs import mask_against_dataframe_converse
    import pandas as pd

    df1 = pd.DataFrame({'ID': [1,2,3], 'Value': [10,20,30]})
    df2 = pd.DataFrame({'ID': [2,3,4], 'Extra': ['X','Y','Z']})

    df_uncommon = mask_against_dataframe_converse(df1, df2, 'ID')
    print(df_uncommon)
    

--------------------------------------------------------------------------------

### 40. `union_join`
Perform a union join, concatenating two DataFrames and dropping duplicates.

• Parameters:
  - `df1` (pd.DataFrame): First DataFrame.
  - `df2` (pd.DataFrame): Second DataFrame.

• Returns:
  - pd.DataFrame: A new DataFrame with the union of `df1` and `df2`, without duplicates.

• Example:

    from rgwfuncs import union_join
    import pandas as pd

    df1 = pd.DataFrame({'ID': [1, 2, 3], 'Value': [10, 20, 30]})
    df2 = pd.DataFrame({'ID': [2, 3, 4], 'Value': [20, 30, 40]})

    df_union = union_join(df1, df2)
    print(df_union)

--------------------------------------------------------------------------------

### 41. `bag_union_join`
Perform a bag union join, concatenating two DataFrames without dropping duplicates.

• Parameters:
  - `df1` (pd.DataFrame): First DataFrame.
  - `df2` (pd.DataFrame): Second DataFrame.

• Returns:
  - pd.DataFrame: A new DataFrame with the concatenated data of `df1` and `df2`.

• Example:

    from rgwfuncs import bag_union_join
    import pandas as pd

    df1 = pd.DataFrame({'ID': [1, 2, 3], 'Value': [10, 20, 30]})
    df2 = pd.DataFrame({'ID': [2, 3, 4], 'Value': [20, 30, 40]})

    df_bag_union = bag_union_join(df1, df2)
    print(df_bag_union)

--------------------------------------------------------------------------------

### 42. `left_join`
Perform a left join on two DataFrames.

• Parameters:
  - `df1` (pd.DataFrame): The left DataFrame.
  - `df2` (pd.DataFrame): The right DataFrame.
  - `left_on` (str): Column name in `df1` to join on.
  - `right_on` (str): Column name in `df2` to join on.

• Returns:
  - pd.DataFrame: A new DataFrame as the result of a left join.

• Example:

    from rgwfuncs import left_join
    import pandas as pd

    df1 = pd.DataFrame({'ID': [1, 2, 3], 'Value': [10, 20, 30]})
    df2 = pd.DataFrame({'ID': [2, 3, 4], 'Extra': ['A', 'B', 'C']})

    df_left_join = left_join(df1, df2, 'ID', 'ID')
    print(df_left_join)

--------------------------------------------------------------------------------

### 43. `right_join`
Perform a right join on two DataFrames.

• Parameters:
  - `df1` (pd.DataFrame): The left DataFrame.
  - `df2` (pd.DataFrame): The right DataFrame.
  - `left_on` (str): Column name in `df1` to join on.
  - `right_on` (str): Column name in `df2` to join on.

• Returns:
  - pd.DataFrame: A new DataFrame as the result of a right join.

• Example:

    from rgwfuncs import right_join
    import pandas as pd

    df1 = pd.DataFrame({'ID': [1, 2, 3], 'Value': [10, 20, 30]})
    df2 = pd.DataFrame({'ID': [2, 3, 4], 'Extra': ['A', 'B', 'C']})

    df_right_join = right_join(df1, df2, 'ID', 'ID')
    print(df_right_join)

--------------------------------------------------------------------------------

### 44. `insert_dataframe_in_sqlite_database`

Inserts a Pandas DataFrame into a SQLite database table. If the specified table does not exist, it will be created with column types automatically inferred from the DataFrame's data types.

- **Parameters:**
  - `db_path` (str): The path to the SQLite database file. If the database does not exist, it will be created.
  - `tablename` (str): The name of the table in the database. If the table does not exist, it is created with the DataFrame's columns and data types.
  - `df` (pd.DataFrame): The DataFrame containing the data to be inserted into the database table.

- **Returns:**
  - `None`

- **Notes:**
  - Data types in the DataFrame are converted to SQLite-compatible types:
    - `int64` is mapped to `INTEGER`
    - `float64` is mapped to `REAL`
    - `object` is mapped to `TEXT`
    - `datetime64[ns]` is mapped to `TEXT` (dates are stored as text)
    - `bool` is mapped to `INTEGER` (SQLite does not have a separate Boolean type)

- **Example:**

    from rgwfuncs import insert_dataframe_in_sqlite_database
    import pandas as pd

    df = pd.DataFrame({
        'ID': [1, 2, 3],
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Score': [88.5, 92.3, 85.0]
    })

    db_path = 'my_database.db'
    tablename = 'students'

    insert_dataframe_in_sqlite_database(db_path, tablename, df)

--------------------------------------------------------------------------------

### 45. `sync_dataframe_to_sqlite_database`
Processes and saves a DataFrame to an SQLite database, adding a timestamp column and replacing the existing table if needed. Creates the table if it does not exist.

• Parameters:
  - `db_path` (str): Path to the SQLite database file.
  - `tablename` (str): The name of the table in the database.
  - `df` (pd.DataFrame): The DataFrame to be processed and saved.

• Returns:
  - None

• Example:

    from rgwfuncs import sync_dataframe_to_sqlite_database
    import pandas as pd

    df = pd.DataFrame({'ID': [1, 2, 3], 'Value': [10, 20, 30]})
    db_path = 'my_database.db'
    tablename = 'my_table'

    sync_dataframe_to_sqlite_database(db_path, tablename, df)

--------------------------------------------------------------------------------



## Additional Info

For more information, refer to each function’s docstring by calling:

    rgwfuncs.docs(method_type_filter='function_name')

or display all docstrings with:
    
    rgwfuncs.docs(method_type_filter='*')


--------------------------------------------------------------------------------

© 2025 Ryan Gerard Wilson. All rights reserved.

