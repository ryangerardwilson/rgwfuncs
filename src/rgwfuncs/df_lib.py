import pandas as pd
import pymssql
import os
import json
from datetime import datetime
import time
import gc
import mysql.connector
import tempfile
import clickhouse_connect
from google.cloud import bigquery
from google.oauth2 import service_account
import xgboost as xgb
from pprint import pprint
import requests
from slack_sdk import WebClient
import sqlite3
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from googleapiclient.discovery import build
import base64
import inspect
from typing import Optional, Callable, Dict, List, Tuple, Any
import warnings

# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


def df_docs(method_type_filter: Optional[str] = None) -> None:
    """
    Print a list of function names in alphabetical order. If method_type_filter
    is specified, print the docstrings of the functions that match the filter.
    Using '*' as a filter will print the docstrings for all functions.

    Parameters:
        method_type_filter: Optional filter string representing a function name,
        or '*' to display docstrings for all functions.
    """
    # Get the current module's namespace
    current_module = __name__

    local_functions: Dict[str, Callable] = {
        name: obj for name, obj in globals().items()
        if inspect.isfunction(obj) and obj.__module__ == current_module
    }

    # List of function names sorted alphabetically
    function_names = sorted(local_functions.keys())

    # Print function names
    print("Functions in alphabetical order:")
    for name in function_names:
        print(name)

    # If a filter is provided or '*', print the docstrings of functions
    if method_type_filter:
        # print("\nFiltered function documentation:")
        for name, func in local_functions.items():
            docstring: Optional[str] = func.__doc__
            if docstring:
                if method_type_filter == '*' or method_type_filter == name:
                    # Print the entire docstring for the matching function
                    print(f"\n{name}:\n{docstring}")


def numeric_clean(
        df: pd.DataFrame,
        column_names: str,
        column_type: str,
        irregular_value_treatment: str) -> pd.DataFrame:
    """
    Cleans the numeric columns based on specified treatments.

    Parameters:
        df: The DataFrame to clean.
        column_names: A comma-separated string containing the names of the columns to clean.
        column_type: The type to convert the column to ('INTEGER' or 'FLOAT').
        irregular_value_treatment: How to treat irregular values ('NAN', 'TO_ZERO', 'MEAN').

    Returns:
        A new DataFrame with cleaned numeric columns.
    """
    df_copy = df.copy()  # Avoid mutating the original DataFrame
    columns_list: List[str] = [name.strip()
                               for name in column_names.split(',')]

    for column_name in columns_list:
        if column_name not in df_copy.columns:
            raise ValueError(
                f"Column '{column_name}' does not exist in the DataFrame.")

        if column_type not in ['INTEGER', 'FLOAT']:
            raise ValueError("column_type must be 'INTEGER' or 'FLOAT'.")

        if irregular_value_treatment not in ['NAN', 'TO_ZERO', 'MEAN']:
            raise ValueError(
                "irregular_value_treatment must be 'NAN', 'TO_ZERO', or"
                + "'MEAN'.")

        # Convert column type
        if column_type == 'INTEGER':
            df_copy[column_name] = pd.to_numeric(
                df_copy[column_name],
                errors='coerce').astype(
                pd.Int64Dtype())
        elif column_type == 'FLOAT':
            df_copy[column_name] = pd.to_numeric(
                df_copy[column_name], errors='coerce').astype(float)

        # Handle irregular values
        if irregular_value_treatment == 'NAN':
            pass  # Already converted to NaN
        elif irregular_value_treatment == 'TO_ZERO':
            df_copy[column_name] = df_copy[column_name].fillna(0)
        elif irregular_value_treatment == 'MEAN':
            mean_value = df_copy[column_name].mean()
            df_copy[column_name] = df_copy[column_name].fillna(mean_value)

    return df_copy


def limit_dataframe(df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
    """
    Limit the DataFrame to a specified number of rows.

    Parameters:
        df: The DataFrame to limit.
        num_rows: The number of rows to retain.

    Returns:
        A new DataFrame limited to the specified number of rows.

    Raises:
        ValueError: If num_rows is not an integer.
    """
    if not isinstance(num_rows, int):
        raise ValueError("The number of rows should be an integer.")

    return df.head(num_rows)


def from_raw_data(headers: List[str], data: List[List[int]]) -> pd.DataFrame:
    """
    Create a DataFrame from raw data.

    Parameters:
        headers: A list of column headers.
        data: A two-dimensional list of data.

    Returns:
        A DataFrame created from the raw data.

    Raises:
        ValueError: If data is not in the correct format.
    """
    if isinstance(data, list) and all(isinstance(row, list) for row in data):
        df = pd.DataFrame(data, columns=headers)
    else:
        raise ValueError("Data should be an array of arrays.")

    return df


def append_rows(df: pd.DataFrame, rows: List[List]) -> pd.DataFrame:
    """
    Append rows to the DataFrame.

    Parameters:
        df: The original DataFrame.
        rows: A list of lists, where each inner list represents a row to be appended.

    Returns:
        A new DataFrame with the appended rows.

    Raises:
        ValueError: If rows are not in the correct format.
    """
    if not isinstance(
        rows,
        list) or not all(
        isinstance(
            row,
            list) for row in rows):
        raise ValueError("Rows should be provided as a list of lists.")

    if df.empty:
        new_df = pd.DataFrame(rows)
    else:
        new_rows_df = pd.DataFrame(rows, columns=df.columns)
        new_df = pd.concat([df, new_rows_df], ignore_index=True)

    return new_df


def append_columns(df: pd.DataFrame, *col_names: str) -> pd.DataFrame:
    """
    Append columns to the DataFrame with None values.

    Parameters:
        df: The original DataFrame.
        col_names: The names of the columns to add.

    Returns:
        A new DataFrame with the appended columns.

    Raises:
        ValueError: If column names are not provided correctly.
    """
    if not all(isinstance(col_name, str) for col_name in col_names):
        raise ValueError("Column names should be provided as strings.")

    new_df = df.copy()
    for col_name in col_names:
        new_df[col_name] = pd.Series([None] * len(df), dtype='object')

    return new_df


def update_rows(
    df: pd.DataFrame,
    condition: str,
    updates: Dict[str, any]
) -> pd.DataFrame:
    """
    Update specific rows in the DataFrame based on a condition.

    Parameters:
        df: The original DataFrame.
        condition: A query condition to identify rows for updating.
        updates: A dictionary with column names as keys and new values as values.

    Returns:
        A new DataFrame with the updated rows.

    Raises:
        ValueError: If no rows match the condition or updates are invalid.
    """
    mask = df.query(condition)

    if mask.empty:
        raise ValueError("No rows match the given condition.")

    if not isinstance(updates, dict):
        raise ValueError("Updates should be provided as a dictionary.")

    invalid_cols = [col for col in updates if col not in df.columns]
    if invalid_cols:
        raise ValueError(
            f"Columns {
                ', '.join(invalid_cols)} do not exist in the DataFrame.")

    new_df = df.copy()
    for col_name, new_value in updates.items():
        new_df.loc[mask.index, col_name] = new_value

    return new_df


def delete_rows(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    Delete rows from the DataFrame based on a condition.

    Parameters:
        df: The original DataFrame.
        condition: A query condition to identify rows for deletion.

    Returns:
        A new DataFrame with the specified rows deleted.

    Raises:
        ValueError: If no rows match the condition.
    """
    mask = df.query(condition)

    if mask.empty:
        raise ValueError("No rows match the given condition.")

    new_df = df.drop(mask.index).reset_index(drop=True)

    return new_df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows in the DataFrame, retaining the first occurrence.

    Parameters:
        df: The DataFrame from which duplicates will be dropped.

    Returns:
        A new DataFrame with duplicates removed.

    Raises:
        ValueError: If the DataFrame is None.
    """
    if df is None:
        raise ValueError("DataFrame is not initialized.")
    return df.drop_duplicates(keep='first')


def drop_duplicates_retain_first(
        df: pd.DataFrame,
        columns: Optional[str] = None) -> pd.DataFrame:
    """
    Drop duplicate rows in the DataFrame based on specified columns, retaining the first occurrence.

    Parameters:
        df: The DataFrame from which duplicates will be dropped.
        columns: A comma-separated string with the column names used to identify duplicates.

    Returns:
        A new DataFrame with duplicates removed.

    Raises:
        ValueError: If the DataFrame is None.
    """
    if df is None:
        raise ValueError("DataFrame is not initialized.")

    columns_list = [col.strip()
                    for col in columns.split(',')] if columns else None
    return df.drop_duplicates(subset=columns_list, keep='first')


def drop_duplicates_retain_last(
        df: pd.DataFrame,
        columns: Optional[str] = None) -> pd.DataFrame:
    """
    Drop duplicate rows in the DataFrame based on specified columns, retaining the last occurrence.

    Parameters:
        df: The DataFrame from which duplicates will be dropped.
        columns: A comma-separated string with the column names used to identify duplicates.

    Returns:
        A new DataFrame with duplicates removed.

    Raises:
        ValueError: If the DataFrame is None.
    """
    if df is None:
        raise ValueError("DataFrame is not initialized.")

    columns_list = [col.strip()
                    for col in columns.split(',')] if columns else None
    return df.drop_duplicates(subset=columns_list, keep='last')


def load_data_from_query(db_preset_name: str, query: str) -> pd.DataFrame:
    """
    Load data from a database query into a DataFrame based on a configuration preset.

    Parameters:
        db_preset_name: The name of the database preset in the configuration file.
        query: The SQL query to execute.

    Returns:
        A DataFrame containing the query result.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If the database preset or db_type is invalid.
    """

    def query_mssql(db_preset: Dict[str, Any], query: str) -> pd.DataFrame:
        server = db_preset['host']
        user = db_preset['username']
        password = db_preset['password']
        database = db_preset.get('database', '')

        with pymssql.connect(server=server, user=user, password=password, database=database) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

        return pd.DataFrame(rows, columns=columns)

    def query_mysql(db_preset: Dict[str, Any], query: str) -> pd.DataFrame:
        host = db_preset['host']
        user = db_preset['username']
        password = db_preset['password']
        database = db_preset.get('database', '')

        with mysql.connector.connect(host=host, user=user, password=password, database=database) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                columns = ([desc[0] for desc in cursor.description]
                           if cursor.description else [])

        return pd.DataFrame(rows, columns=columns)

    def query_clickhouse(
            db_preset: Dict[str, Any], query: str) -> pd.DataFrame:
        host = db_preset['host']
        user = db_preset['username']
        password = db_preset['password']
        database = db_preset['database']

        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                client = clickhouse_connect.get_client(
                    host=host, port='8123', username=user, password=password, database=database)
                data = client.query(query)
                rows = data.result_rows
                columns = data.column_names
                return pd.DataFrame(rows, columns=columns)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(
                        "All attempts to connect to ClickHouse failed.")

    def query_google_big_query(
            db_preset: Dict[str, Any], query: str) -> pd.DataFrame:
        json_file_path = db_preset['json_file_path']
        project_id = db_preset['project_id']

        credentials = service_account.Credentials.from_service_account_file(
            json_file_path)
        client = bigquery.Client(credentials=credentials, project=project_id)

        query_job = client.query(query)
        results = query_job.result()
        rows = [list(row.values()) for row in results]
        columns = [field.name for field in results.schema]

        return pd.DataFrame(rows, columns=columns)

    # Assume the configuration file is located at ~/.rgwfuncsrc
    config_path = os.path.expanduser('~/.rgwfuncsrc')
    with open(config_path, 'r') as f:
        config = json.load(f)

    db_presets = config.get('db_presets', [])
    db_preset = next(
        (preset for preset in db_presets if preset['name'] == db_preset_name),
        None)
    if not db_preset:
        raise ValueError(f"No matching db_preset found for {db_preset_name}")

    db_type = db_preset['db_type']

    if db_type == 'mssql':
        return query_mssql(db_preset, query)
    elif db_type == 'mysql':
        return query_mysql(db_preset, query)
    elif db_type == 'clickhouse':
        return query_clickhouse(db_preset, query)
    elif db_type == 'google_big_query':
        return query_google_big_query(db_preset, query)
    else:
        raise ValueError(f"Unsupported db_type: {db_type}")


def load_data_from_path(file_path: str) -> pd.DataFrame:
    """
    Load data from a file into a DataFrame based on the file extension.

    Parameters:
        file_path: The absolute path to the data file.

    Returns:
        A DataFrame containing the data loaded from the file.

    Raises:
        ValueError: If the file extension is unsupported.
    """

    def load_hdf5(file_path: str) -> pd.DataFrame:
        """Helper function to load HDF5 files and select a key if necessary."""
        with pd.HDFStore(file_path, mode='r') as store:
            available_keys = store.keys()
            if len(available_keys) == 1:
                df = pd.read_hdf(file_path, key=available_keys[0])
                print(f"Loaded key: {available_keys[0]}")
            else:
                while True:
                    print("Available keys:", available_keys)
                    key = input("Enter the key for the HDF5 dataset: ").strip()
                    if key in available_keys:
                        df = pd.read_hdf(file_path, key=key)
                        break
                    else:
                        print(
                            f"Key '{key}' is not in the available keys.")
        return df

    # Ensure the file path is absolute
    file_path = os.path.abspath(file_path)

    # Determine file type by extension
    file_extension = file_path.split('.')[-1].lower()

    # Load data based on file type
    if file_extension == 'csv':
        df = pd.read_csv(file_path, dtype=str)
        df.replace('', None, inplace=True)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file_path)
    elif file_extension == 'json':
        df = pd.read_json(file_path)
    elif file_extension == 'parquet':
        df = pd.read_parquet(file_path)
    elif file_extension in ['h5', 'hdf5']:
        df = load_hdf5(file_path)
    elif file_extension == 'feather':
        df = pd.read_feather(file_path)
    elif file_extension == 'pkl':
        df = pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    gc.collect()
    return df


def load_data_from_sqlite_path(sqlite_path: str, query: str) -> pd.DataFrame:
    """
    Execute a query on a SQLite database specified by its path and return the results as a DataFrame.

    Parameters:
        sqlite_path: The absolute path to the SQLite database file.
        query: The SQL query to execute.

    Returns:
        A DataFrame containing the query results.

    Raises:
        ValueError: If there is a problem executing the query.
    """

    # Ensure the file path is absolute
    sqlite_path = os.path.abspath(sqlite_path)

    try:
        with sqlite3.connect(sqlite_path) as conn:
            df = pd.read_sql_query(query, conn)
    except sqlite3.Error as e:
        raise ValueError(f"SQLite error: {e}")

    gc.collect()
    return df


def first_n_rows(df: pd.DataFrame, n: int) -> None:
    """
    Display the first n rows of the DataFrame.

    This function prints out the first `n` rows of a given DataFrame. Each row is formatted for clarity and printed as a dictionary. If the DataFrame is empty or `None`, it raises a ValueError.

    Parameters:
    - df (pd.DataFrame): The DataFrame to display rows from.
    - n (int): The number of rows to display from the start of the DataFrame.

    Raises:
    - ValueError: If the DataFrame is `None`.
    """
    if df is not None:
        first_n_rows = df.head(n).to_dict(orient="records")
        for row in first_n_rows:
            pprint(row, indent=4)
            print()
    else:
        raise ValueError(
            "No DataFrame to display. Please provide a DataFrame.")

    gc.collect()


def last_n_rows(df: pd.DataFrame, n: int) -> None:
    """
    Display the last n rows of the DataFrame.

    Prints the last `n` rows of a given DataFrame, formatted as dictionaries. Useful for end-segment analysis and verifying data continuity.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which to display rows.
    - n (int): The number of rows to display from the end of the DataFrame.

    Raises:
    - ValueError: If the DataFrame is `None`.
    """
    if df is not None:
        last_n_rows = df.tail(n).to_dict(orient="records")
        for row in last_n_rows:
            pprint(row, indent=4)
            print()
    else:
        raise ValueError(
            "No DataFrame to display. Please provide a DataFrame.")

    gc.collect()


def top_n_unique_values(df: pd.DataFrame, n: int, columns: List[str]) -> None:
    """
    Print the top `n` unique values for specified columns in the DataFrame.

    This method calculates and prints the top `n` unique frequency values for specified columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which to calculate top unique
    values.
    - n (int): Number of top values to display.
    - columns (List[str]): List of column names for which to display top unique values.

    Raises:
    - ValueError: If the DataFrame is `None`.
    """
    if df is not None:
        report = {}
        for column in columns:
            if column in df.columns:
                frequency = df[column].astype(str).value_counts(dropna=False)
                frequency = frequency.rename(
                    index={
                        'nan': 'NaN',
                        'NaT': 'NaT',
                        'None': 'None',
                        '': 'Empty'})
                top_n_values = frequency.nlargest(n)
                report[column] = {str(value): str(count)
                                  for value, count in top_n_values.items()}
                print(
                    f"Top {n} unique values for column '{column}':\n{
                        json.dumps(
                            report[column],
                            indent=2)}\n")
            else:
                print(f"Column '{column}' does not exist in the DataFrame.")
    else:
        raise ValueError(
            "No DataFrame to display. Please provide a DataFrame.")

    gc.collect()


def bottom_n_unique_values(
        df: pd.DataFrame,
        n: int,
        columns: List[str]) -> None:
    """
    Print the bottom `n` unique values for specified columns in the DataFrame.

    This method calculates and prints the bottom `n` unique frequency values for specified columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which to calculate bottom unique
    values.
    - n (int): Number of bottom unique frequency values to display.
    - columns (List[str]): List of column names for which to display bottom unique values.

    Raises:
    - ValueError: If the DataFrame is `None`.
    """
    if df is not None:
        report = {}
        for column in columns:
            if column in df.columns:
                frequency = df[column].astype(str).value_counts(dropna=False)
                frequency = frequency.rename(
                    index={
                        'nan': 'NaN',
                        'NaT': 'NaT',
                        'None': 'None',
                        '': 'Empty'})
                bottom_n_values = frequency.nsmallest(n)
                report[column] = {
                    str(value): str(count) for value,
                    count in bottom_n_values.items()}
                print(
                    f"Bottom {n} unique values for column '{column}':\n{
                        json.dumps(
                            report[column],
                            indent=2)}\n")
            else:
                print(f"Column '{column}' does not exist in the DataFrame.")
    else:
        raise ValueError(
            "No DataFrame to display. Please provide a DataFrame.")

    gc.collect()


def print_correlation(
        df: pd.DataFrame, column_pairs: List[Tuple[str, str]]) -> None:
    """
    Print correlation for multiple pairs of columns in the DataFrame.

    This function computes and displays the correlation coefficients for specified pairs of columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns to analyze.
    - column_pairs (List[Tuple[str, str]]): List of column pairs for which to compute correlations.
    """
    if df is not None:
        for col1, col2 in column_pairs:
            if col1 in df.columns and col2 in df.columns:
                try:
                    numeric_col1 = pd.to_numeric(df[col1], errors='coerce')
                    numeric_col2 = pd.to_numeric(df[col2], errors='coerce')

                    correlation = numeric_col1.corr(numeric_col2)
                    if pd.notnull(correlation):
                        print(
                            f"The correlation between '{col1}' and '{col2}' is {correlation}.")
                    else:
                        print(
                            f"Cannot calculate correlation between '{col1}' and '{col2}' due to insufficient numeric data.")
                except Exception as e:
                    print(f"Error processing cols '{col1}' and '{col2}': {e}")
            else:
                print(
                    f"One or both of the specified cols ('{col1}', '{col2}') do not exist in the DataFrame.")
    else:
        print("The DataFrame is empty.")

    gc.collect()


def print_memory_usage(df: pd.DataFrame) -> None:
    """
    Prints the memory usage of the DataFrame.

    This function computes the memory footprint of a DataFrame in megabytes and displays it, rounding to two decimal places for clarity.

    Parameters:
    - df (pd.DataFrame): The DataFrame for which the memory usage is computed.

    Raises:
    - ValueError: If the DataFrame is `None`.
    """
    if df is not None:
        memory_usage = df.memory_usage(deep=True).sum(
        ) / (1024 * 1024)  # Convert bytes to MB
        print(f"Memory usage of DataFrame: {memory_usage:.2f} MB")
    else:
        raise ValueError("No DataFrame to print. Please provide a DataFrame.")

    gc.collect()


def filter_dataframe(df: pd.DataFrame, filter_expr: str) -> pd.DataFrame:
    """
    Return a filtered DataFrame according to the given expression.

    This function filters rows of a DataFrame using a specified query expression, returning a new DataFrame containing only the rows that match the criteria.

    Parameters:
    - df (pd.DataFrame): The original DataFrame to be filtered.
    - filter_expr (str): A query string to be evaluated against the DataFrame.

    Returns:
    - pd.DataFrame: A new DataFrame containing the filtered rows.

    Raises:
    - ValueError: If the DataFrame is `None`.
    """
    if df is not None:
        try:
            filtered_df = df.query(filter_expr)
        except Exception:
            filtered_df = df[df.eval(filter_expr)]
    else:
        raise ValueError("No DataFrame to filter. Please provide a DataFrame.")

    gc.collect()

    return filtered_df


def filter_indian_mobiles(df: pd.DataFrame, mobile_col: str) -> pd.DataFrame:
    """
    Filter and return DataFrame rows containing valid Indian mobile numbers.

    This function processes a DataFrame to extract and retain rows where the specified column matches the typical format for Indian mobile numbers. An Indian mobile number is expected to be a digit-only string starting with 6, 7, 8, or 9, and should have at least 4 distinct digits.

    Parameters:
    - df (pd.DataFrame): The DataFrame to filter.
    - mobile_col (str): The name of the column in the DataFrame that contains mobile number data.

    Returns:
    - pd.DataFrame: A new DataFrame containing only rows with valid Indian mobile numbers.

    Raises:
    - ValueError: If the DataFrame is `None`.
    """
    if df is not None:
        filtered_df = df[
            df[mobile_col].apply(
                lambda x: (
                    str(x).isdigit() and
                    str(x).startswith(('6', '7', '8', '9')) and
                    len(set(str(x))) >= 4
                )
            )
        ]
    else:
        raise ValueError("No DataFrame to filter. Please provide a DataFrame.")

    gc.collect()

    return filtered_df


def print_dataframe(df: pd.DataFrame, source: Optional[str] = None) -> None:
    """
    Print the DataFrame and its column types. If a source path is provided, print it as well.

    Parameters:
        df: The DataFrame to print.
        source: Optional; The source path of the DataFrame for logging purposes.
    """
    if df is not None:
        print(df)
        columns_with_types = [
            f"{col} ({df[col].dtypes})" for col in df.columns]
        print("Columns:", columns_with_types)
        if source:
            print(f"Source: {source}")
    else:
        raise ValueError("No DataFrame to print. Please provide a DataFrame.")

    gc.collect()


def send_dataframe_via_telegram(df: pd.DataFrame, bot_name: str, message: Optional[str] = None, as_file: bool = True, remove_after_send: bool = True) -> None:
    """
    Send a DataFrame via Telegram using a specified bot configuration.

    Parameters:
        df: The DataFrame to send.
        bot_name: The name of the Telegram bot as specified in the configuration file.
        message: Custom message to send along with the DataFrame or file. Defaults to None.
        as_file: Boolean flag to indicate whether the DataFrame should be sent as a file (True) or as text (False). Defaults to True.
        remove_after_send: If True, removes the CSV file after sending. Defaults to True.

    Raises:
        ValueError: If the specified bot is not found or if no DataFrame is provided.
        Exception: If the message sending fails.

    Notes:
        The configuration file is assumed to be located at `~/.rgwfuncsrc`.
    """

    def get_config(config_path: str) -> dict:
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as file:
            return json.load(file)

    # Assume the configuration file is located at ~/.rgwfuncsrc
    config_path = os.path.expanduser('~/.rgwfuncsrc')
    config = get_config(config_path)

    bot_config = next(
        (bot for bot in config['telegram_bot_presets'] if bot['name'] == bot_name),
        None)
    if not bot_config:
        raise ValueError(f"No bot found with the name {bot_name}")

    if df is None:
        raise ValueError("No DataFrame to send. Please provide a DataFrame.")

    response = None
    if as_file:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"df_{timestamp}.csv"
        df.to_csv(file_name, index=False)
        try:
            with open(file_name, 'rb') as file:
                payload = {
                    'chat_id': bot_config['chat_id'],
                    'caption': message or ''}
                files = {'document': file}
                response = requests.post(
                    f"https://api.telegram.org/bot{
                        bot_config['bot_token']}/sendDocument",
                    data=payload,
                    files=files)
            if remove_after_send and os.path.exists(file_name):
                os.remove(file_name)
        except Exception as e:
            print(f"Failed to send document: {e}")
            raise
    else:
        df_str = df.to_string()
        payload = {
            'chat_id': bot_config['chat_id'],
            'text': (message + "\n\n" + df_str) if message else df_str,
            'parse_mode': 'HTML'
        }
        response = requests.post(
            f"https://api.telegram.org/bot{bot_config['bot_token']}/sendMessage", data=payload)

    if response and not response.ok:
        raise Exception(f"Error sending message: {response.text}")

    print("Message sent successfully.")


def send_data_to_email(
        df: pd.DataFrame,
        preset_name: str,
        to_email: str,
        subject: Optional[str] = None,
        body: Optional[str] = None,
        as_file: bool = True,
        remove_after_send: bool = True) -> None:
    """
    Send an email with an optional DataFrame attachment using the Gmail API via a specified preset.

    Parameters:
        df: The DataFrame to send.
        preset_name: The configuration preset name to use for sending the email.
        to_email: The recipient email address.
        subject: Optional subject of the email. Defaults to 'DataFrame CSV File' if not given.
        body: Optional message body of the email. Defaults to 'Please find the CSV file attached.' if not given.
        as_file: Boolean flag to decide whether to send the DataFrame as a file (True) or embed it in the email (False). Defaults to True.
        remove_after_send: If True, removes the CSV file after sending. Defaults to True.

    Raises:
        ValueError: If the preset is not found in the configuration.
        Exception: If the email preparation or sending fails.

    Notes:
        The configuration file is assumed to be located at `~/.rgwfuncsrc`.
    """

    def get_config(config_path: str) -> dict:
        with open(config_path, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in config file: {e}")

    def authenticate_service_account(
            service_account_credentials_path: str,
            sender_email_id: str) -> Any:
        credentials = service_account.Credentials.from_service_account_file(
            service_account_credentials_path,
            scopes=['https://mail.google.com/'],
            subject=sender_email_id
        )
        return build('gmail', 'v1', credentials=credentials)

    # Load configuration from ~/.rgwfuncsrc
    config_path = os.path.expanduser('~/.rgwfuncsrc')
    config = get_config(config_path)

    # Retrieve Gmail preset configuration
    gmail_config = next(
        (preset for preset in config['gmail_bot_presets'] if preset['name'] == preset_name),
        None)

    if not gmail_config:
        raise ValueError(f"No preset found with the name {preset_name}")

    sender_email = gmail_config['name']
    credentials_path = gmail_config['service_account_credentials_path']

    # Authenticate and get the Gmail service
    service = authenticate_service_account(credentials_path, sender_email)

    if as_file:
        # Create a temporary file for the DataFrame as CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file_name = tmp_file.name
            df.to_csv(tmp_file_name, index=False)

        # Create email with attachment
        try:
            message = MIMEMultipart()
            message['to'] = to_email
            message['from'] = sender_email
            message['subject'] = subject if subject else 'DataFrame CSV File'
            message.attach(
                MIMEText(
                    body if body else 'Please find the CSV file attached.'))

            with open(tmp_file_name, 'rb') as file:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(file.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename={
                        os.path.basename(tmp_file_name)}')
                message.attach(part)

            if remove_after_send and os.path.exists(tmp_file_name):
                os.remove(tmp_file_name)

        except Exception as e:
            raise Exception(f"Failed to prepare the document: {e}")

    else:
        # Create email body as plain text
        df_str = df.to_string()
        full_body = body + "\n\n" + df_str if body else df_str
        message = MIMEText(full_body)
        message['to'] = to_email
        message['from'] = sender_email
        message['subject'] = subject or 'DataFrame Content'

    # Sending the email
    try:
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        email_body = {'raw': raw}
        sent_message = service.users().messages().send(
            userId="me", body=email_body).execute()
        print(f"Email with Message Id {sent_message['id']} successfully sent.")
    except Exception as error:
        raise Exception(f"Error sending email: {error}")


def send_data_to_slack(
        df: pd.DataFrame,
        bot_name: str,
        message: Optional[str] = None,
        as_file: bool = True,
        remove_after_send: bool = True) -> None:
    """
    Send a DataFrame or message to Slack using a specified bot configuration.

    Parameters:
        df: The DataFrame to send.
        bot_name: The Slack bot configuration preset name.
        message: Custom message to send along with the DataFrame or file. Defaults to None.
        as_file: Boolean flag to decide whether to send the DataFrame as a file (True) or as text (False). Defaults to True.
        remove_after_send: If True, removes the CSV file after sending. Defaults to True.

    Raises:
        ValueError: If the specified bot is not found in the configuration.
        Exception: If the message sending fails.

    Notes:
        The configuration file is assumed to be located at `~/.rgwfuncsrc`.
    """

    def get_config(config_path: str) -> dict:
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as file:
            return json.load(file)

    # Load the Slack configuration from ~/.rgwfuncsrc
    config_path = os.path.expanduser('~/.rgwfuncsrc')
    config = get_config(config_path)

    bot_config = next(
        (bot for bot in config['slack_bot_presets'] if bot['name'] == bot_name),
        None)

    if not bot_config:
        raise ValueError(f"No bot found with the name {bot_name}")

    client = WebClient(token=bot_config['bot_token'])

    if as_file:
        # Create a temporary file for the DataFrame as CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            file_name = tmp_file.name
            df.to_csv(file_name, index=False)

        try:
            with open(file_name, 'rb') as file:
                response = client.files_upload(
                    channels=bot_config['channel_id'],
                    file=file,
                    filename=os.path.basename(file_name),
                    title="DataFrame Upload",
                    initial_comment=message or ''
                )
        finally:
            if remove_after_send and os.path.exists(file_name):
                os.remove(file_name)
    else:
        df_str = df.to_string()
        response = client.chat_postMessage(
            channel=bot_config['channel_id'],
            text=(message + "\n\n" + df_str) if message else df_str
        )

    # Check if the message was sent successfully
    if not response["ok"]:
        raise Exception(f"Error sending message: {response['error']}")

    print("Message sent successfully.")


def order_columns(df: pd.DataFrame, column_order_str: str) -> pd.DataFrame:
    """
    Reorder the columns of the DataFrame based on a string input.

    Parameters:
        df: The DataFrame whose columns will be reordered.
        column_order_str: A string specifying the desired order of columns,
        using ',' to separate columns.

    Returns:
        A new DataFrame with reordered columns.

    Raises:
        ValueError: If a specified column does not exist in the DataFrame.
    """
    if df is None:
        raise ValueError(
            "No DataFrame to reorder. Please provide a valid DataFrame.")

    columns = df.columns.tolist()
    parts = [part.strip() for part in column_order_str.split(',')]

    new_order = []
    seen = set()

    for part in parts:
        if part == '...':
            continue
        elif part in columns:
            new_order.append(part)
            seen.add(part)
        else:
            raise ValueError(f"Column '{part}' not found in DataFrame.")

    remaining = [col for col in columns if col not in seen]

    # Determine the position of '...' and arrange the columns
    if parts[0] == '...':
        new_order = remaining + new_order
    elif parts[-1] == '...':
        new_order = new_order + remaining
    else:
        pos = parts.index('...')
        new_order = new_order[:pos] + remaining + new_order[pos:]

    return df[new_order]


def append_ranged_classification_column(
        df: pd.DataFrame,
        ranges: str,
        target_col: str,
        new_col_name: str) -> pd.DataFrame:
    """
    Append a ranged classification column to the DataFrame.

    Parameters:
        df: The DataFrame to modify.
        ranges: A string representation of numeric ranges separated by commas.
        target_col: The column to analyze.
        new_col_name: The name of the new classification column.

    Returns:
        A new DataFrame with the classification column appended.
    """

    def pad_number(number, integer_length, decimal_length=0, decimal=False):
        if decimal:
            str_number = f"{number:.{decimal_length}f}"
            integer_part, decimal_part = str_number.split('.')
            padded_integer_part = integer_part.zfill(integer_length)
            return f"{padded_integer_part}.{decimal_part}"
        else:
            return str(int(number)).zfill(integer_length)

    range_list = ranges.split(',')
    has_decimals = any('.' in r for r in range_list)

    if has_decimals:
        range_list = [float(r) for r in range_list]

        max_decimal_length = max(
            len(str(r).split('.')[1])
            for r in range_list
            if '.' in str(r)
        )

        max_integer_length = max(
            len(str(int(float(r))))
            for r in range_list
        )

        labels = []

        for i in range(len(range_list) - 1):
            start = pad_number(
                range_list[i],
                max_integer_length,
                max_decimal_length,
                decimal=True
            )

            end = pad_number(
                range_list[i + 1],
                max_integer_length,
                max_decimal_length,
                decimal=True
            )

            label = f"{start} to {end}"
            labels.append(label)

    else:
        range_list = [int(r) for r in range_list]

        max_integer_length = max(
            len(str(r))
            for r in range_list
        )

        labels = [f"{pad_number(range_list[i],
                                max_integer_length)} to {pad_number(range_list[i + 1],
                                                                    max_integer_length)}" for i in range(len(range_list) - 1)]

    # Ensure the target column is numeric
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df[new_col_name] = pd.cut(
        df[target_col],
        bins=range_list,
        labels=labels,
        right=False,
        include_lowest=True)

    return df


def append_percentile_classification_column(
        df: pd.DataFrame,
        percentiles: str,
        target_col: str,
        new_col_name: str) -> pd.DataFrame:
    """
    Append a percentile classification column to the DataFrame.

    Parameters:
        df: The DataFrame to modify.
        percentiles: A string representation of percentile values separated by commas.
        target_col: The column to analyze.
        new_col_name: The name of the new classification column.

    Returns:
        A new DataFrame with the classification column appended.
    """

    def pad_number(number, integer_length, decimal_length=0, decimal=False):
        if decimal:
            str_number = f"{number:.{decimal_length}f}"
            integer_part, decimal_part = str_number.split('.')
            padded_integer_part = integer_part.zfill(integer_length)
            return f"{padded_integer_part}.{decimal_part}"
        else:
            return str(int(number)).zfill(integer_length)

    percentiles_list = percentiles.split(',')
    has_decimals = any('.' in p for p in percentiles_list)

    if has_decimals:
        percentiles_list = [float(p) for p in percentiles_list]
        max_decimal_length = max(
            len(str(p).split('.')[1]) for p in percentiles_list if '.' in str(p))
        max_integer_length = max(len(str(int(float(p))))
                                 for p in percentiles_list)

        labels = []

        for i in range(len(percentiles_list) - 1):
            start = pad_number(
                percentiles_list[i],
                max_integer_length,
                max_decimal_length,
                decimal=True)
            end = pad_number(
                percentiles_list[i + 1], max_integer_length, max_decimal_length, decimal=True)

            label = f"{start} to {end}"
            labels.append(label)
    else:
        percentiles_list = [int(p) for p in percentiles_list]

        max_integer_length = max(len(str(p)) for p in percentiles_list)

        labels = []

        for i in range(len(percentiles_list) - 1):
            start = pad_number(percentiles_list[i], max_integer_length)

            end = pad_number(percentiles_list[i + 1], max_integer_length)

            label = f"{start} to {end}"
            labels.append(label)

    # Ensure the target column is numeric
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    quantiles = [df[target_col].quantile(p / 100) for p in percentiles_list]

    df[new_col_name] = pd.cut(
        df[target_col],
        bins=quantiles,
        labels=labels,
        include_lowest=True)

    return df


def append_ranged_date_classification_column(
        df: pd.DataFrame,
        date_ranges: str,
        target_col: str,
        new_col_name: str) -> pd.DataFrame:
    """
    Append a ranged date classification column to the DataFrame.

    Parameters:
        df: The DataFrame to modify.
        date_ranges: A string representation of date ranges separated by commas.
        target_col: The date column to analyze.
        new_col_name: The name of the new date classification column.

    Returns:
        A new DataFrame with the date classification column appended.
    """

    date_list = [pd.to_datetime(date) for date in date_ranges.split(',')]

    labels = []

    for i in range(len(date_list) - 1):
        start_date = date_list[i].strftime('%Y-%m-%d')
        end_date = date_list[i + 1].strftime('%Y-%m-%d')
        label = f"{start_date} to {end_date}"
        labels.append(label)

    df[new_col_name] = pd.cut(
        pd.to_datetime(df[target_col]),
        bins=date_list,
        labels=labels,
        right=False)

    return df


def rename_columns(df: pd.DataFrame,
                   rename_pairs: Dict[str,
                                      str]) -> pd.DataFrame:
    """
    Rename columns in the DataFrame.

    Parameters:
        df: The DataFrame to modify.
        rename_pairs: A dictionary mapping old column names to new column names.

    Returns:
        A new DataFrame with columns renamed.
    """
    if df is None:
        raise ValueError(
            "No DataFrame to rename columns. Please provide a valid DataFrame.")

    return df.rename(columns=rename_pairs)


def cascade_sort(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Cascade sort the DataFrame by specified columns and order.

    Parameters:
        df: The DataFrame to sort.
        columns: A list of column names with sorting order, e.g.,
        ['Column1::ASC', 'Column2::DESC'].

    Returns:
        A new DataFrame sorted by specified columns.
    """
    if df is None:
        raise ValueError(
            "No DataFrame to sort. Please provide a valid DataFrame.")

    col_names = []
    asc_order = []

    # Parse the columns and sorting order
    for col in columns:
        if "::" in col:
            name, order = col.split("::")
            col_names.append(name)
            asc_order.append(order.upper() == "ASC")
        else:
            col_names.append(col)
            asc_order.append(True)

    # Ensure all specified columns exist
    for name in col_names:
        if name not in df.columns:
            raise ValueError(f"Column {name} not found in DataFrame")

    return df.sort_values(by=col_names, ascending=asc_order)


def append_xgb_labels(df: pd.DataFrame, ratio_str: str) -> pd.DataFrame:
    """
    Append XGB training labels based on a ratio string.

    Parameters:
        df: The DataFrame to modify.
        ratio_str: A string specifying the ratio of TRAIN:TEST or TRAIN:VALIDATE:TEST.

    Returns:
        A new DataFrame with XGB_TYPE labels appended.
    """
    if df is None:
        raise ValueError(
            "No DataFrame to add labels. Please provide a valid DataFrame.")

    ratios = list(map(int, ratio_str.split(':')))
    total_ratio = sum(ratios)
    total_rows = len(df)

    if len(ratios) == 2:
        train_rows = (ratios[0] * total_rows) // total_ratio
        test_rows = total_rows - train_rows
        labels = ['TRAIN'] * train_rows + ['TEST'] * test_rows
    elif len(ratios) == 3:
        train_rows = (ratios[0] * total_rows) // total_ratio
        validate_rows = (ratios[1] * total_rows) // total_ratio
        test_rows = total_rows - train_rows - validate_rows
        labels = ['TRAIN'] * train_rows + ['VALIDATE'] * \
            validate_rows + ['TEST'] * test_rows
    else:
        raise ValueError(
            "Invalid ratio string format. Use 'TRAIN:TEST' or 'TRAIN:VALIDATE:TEST'.")

    df_with_labels = df.copy()
    df_with_labels['XGB_TYPE'] = labels

    return df_with_labels


def append_xgb_regression_predictions(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: str,
        pred_col: str,
        boosting_rounds: int = 100,
        model_path: Optional[str] = None) -> pd.DataFrame:
    """
    Append XGB regression predictions to DataFrame. Assumes data is labeled by an 'XGB_TYPE' column.

    Parameters:
        df: DataFrame to modify.
        target_col: The target column for regression.
        feature_cols: Comma-separated string of feature columns.
        pred_col: Name of the prediction column.
        boosting_rounds: (Optional) Number of boosting rounds for training.
        model_path: (Optional) Path to save the trained model.

    Returns:
        DataFrame with predictions appended.
    """
    if df is None or 'XGB_TYPE' not in df.columns:
        raise ValueError(
            "DataFrame is not initialized or 'XGB_TYPE' column is missing.")

    features = feature_cols.replace(' ', '').split(',')

    # Convert categorical columns to 'category' dtype
    for col in features:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    train_data = df[df['XGB_TYPE'] == 'TRAIN']

    if 'VALIDATE' in df['XGB_TYPE'].values:
        validate_data = df[df['XGB_TYPE'] == 'VALIDATE']
    else:
        validate_data = None

    dtrain = xgb.DMatrix(
        train_data[features],
        label=train_data[target_col],
        enable_categorical=True)
    evals = [(dtrain, 'train')]

    if validate_data is not None:
        dvalidate = xgb.DMatrix(
            validate_data[features],
            label=validate_data[target_col],
            enable_categorical=True)
        evals.append((dvalidate, 'validate'))

    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=boosting_rounds,
        evals=evals,
        early_stopping_rounds=10 if validate_data is not None else None)

    # Make predictions for all data
    dall = xgb.DMatrix(df[features], enable_categorical=True)
    df[pred_col] = model.predict(dall)

    if model_path:
        model.save_model(model_path)

    columns_order = [col for col in df.columns if col not in [
        'XGB_TYPE', target_col, pred_col]] + ['XGB_TYPE', target_col, pred_col]
    df = df[columns_order]

    return df


def append_xgb_logistic_regression_predictions(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: str,
        pred_col: str,
        boosting_rounds: int = 100,
        model_path: Optional[str] = None) -> pd.DataFrame:
    """
    Append XGB logistic regression predictions to DataFrame. Assumes data is labeled by an 'XGB_TYPE' column.

    Parameters:
        df: DataFrame to modify.
        target_col: The target column for logistic regression.
        feature_cols: Comma-separated string of feature columns.
        pred_col: Name of the prediction column.
        boosting_rounds: (Optional) Number of boosting rounds for training.
        model_path: (Optional) Path to save the trained model.

    Returns:
        DataFrame with predictions appended.
    """
    if df is None or 'XGB_TYPE' not in df.columns:
        raise ValueError(
            "DataFrame is not initialized or 'XGB_TYPE' column is missing.")

    features = feature_cols.replace(' ', '').split(',')

    # Convert categorical columns to 'category' dtype
    for col in features:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    train_data = df[df['XGB_TYPE'] == 'TRAIN']

    validate_data = None
    if 'VALIDATE' in df['XGB_TYPE'].values:
        validate_data = df[df['XGB_TYPE'] == 'VALIDATE']

    dtrain = xgb.DMatrix(
        train_data[features],
        label=train_data[target_col],
        enable_categorical=True)
    evals = [(dtrain, 'train')]

    if validate_data is not None:
        dvalidate = xgb.DMatrix(
            validate_data[features],
            label=validate_data[target_col],
            enable_categorical=True)
        evals.append((dvalidate, 'validate'))

    params = {'objective': 'binary:logistic', 'eval_metric': 'auc'}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=boosting_rounds,
        evals=evals,
        early_stopping_rounds=10 if validate_data is not None else None)

    # Make predictions for all data
    dall = xgb.DMatrix(df[features], enable_categorical=True)
    df[pred_col] = model.predict(dall)

    if model_path:
        model.save_model(model_path)

    columns_order = [col for col in df.columns if col not in [
        'XGB_TYPE', target_col, pred_col]] + ['XGB_TYPE', target_col, pred_col]
    df = df[columns_order]

    return df


def print_n_frequency_cascading(
        df: pd.DataFrame,
        n: int,
        columns: str,
        order_by: str = "FREQ_DESC") -> None:
    """
    Print the cascading frequency of top n values for specified columns.

    Parameters:
        df: DataFrame to analyze.
        n: Number of top values to print.
        columns: Comma-separated column names to analyze.
        order_by: Order of frequency: ACS, DESC, FREQ_ASC, FREQ_DESC.
    """
    columns = [col.strip() for col in columns.split(",")]

    def generate_cascade_report(df, columns, limit, order_by):
        if not columns:
            return None

        current_col = columns[0]
        if current_col not in df.columns:
            return None

        # Convert the column to string representation
        df[current_col] = df[current_col].astype(str)
        frequency = df[current_col].value_counts(dropna=False)
        frequency = frequency.rename(
            index={
                'nan': 'NaN',
                'NaT': 'NaT',
                'None': 'None',
                '': 'Empty'})

        if limit is not None:
            frequency = frequency.nlargest(limit)

        sorted_frequency = sort_frequency(frequency, order_by)

        report = {}
        for value, count in sorted_frequency.items():
            if value in ['NaN', 'NaT', 'None', 'Empty']:
                filtered_df = df[df[current_col].isna()]
            else:
                filtered_df = df[df[current_col] == value]

            if len(columns) > 1:
                sub_report = generate_cascade_report(
                    filtered_df, columns[1:], limit, order_by)
                report[value] = {
                    "count": str(count), f"sub_distribution({
                        columns[1]})": sub_report if sub_report else {}}
            else:
                report[value] = {"count": str(count)}

        return report

    def sort_frequency(frequency, order_by):
        if order_by == "ASC":
            return dict(sorted(frequency.items(), key=lambda item: item[0]))
        elif order_by == "DESC":
            return dict(
                sorted(
                    frequency.items(),
                    key=lambda item: item[0],
                    reverse=True))
        elif order_by == "FREQ_ASC":
            return dict(sorted(frequency.items(), key=lambda item: item[1]))
        else:  # Default to "FREQ_DESC"
            return dict(
                sorted(
                    frequency.items(),
                    key=lambda item: item[1],
                    reverse=True))

    report = generate_cascade_report(df, columns, n, order_by)
    print(json.dumps(report, indent=2))


def print_n_frequency_linear(
        df: pd.DataFrame,
        n: int,
        columns: str,
        order_by: str = "FREQ_DESC") -> None:
    """
    Print the linear frequency of top n values for specified columns.

    Parameters:
        df: DataFrame to analyze.
        n: Number of top values to print.
        columns: Comma-separated column names to analyze.
        order_by: Order of frequency: ACS, DESC, FREQ_ASC, FREQ_DESC.
    """
    columns = [col.strip() for col in columns.split(",")]

    def generate_linear_report(df, columns, limit, order_by):
        report = {}

        for current_col in columns:
            if current_col not in df.columns:
                continue

            frequency = df[current_col].astype(str).value_counts(dropna=False)
            frequency = frequency.rename(
                index={
                    'nan': 'NaN',
                    'NaT': 'NaT',
                    'None': 'None',
                    '': 'Empty'})

            if limit is not None:
                frequency = frequency.nlargest(limit)

            sorted_frequency = sort_frequency(frequency, order_by)
            col_report = {str(value): str(count)
                          for value, count in sorted_frequency.items()}
            report[current_col] = col_report

        return report

    def sort_frequency(frequency, order_by):
        if order_by == "ASC":
            return dict(sorted(frequency.items(), key=lambda item: item[0]))
        elif order_by == "DESC":
            return dict(
                sorted(
                    frequency.items(),
                    key=lambda item: item[0],
                    reverse=True))
        elif order_by == "FREQ_ASC":
            return dict(sorted(frequency.items(), key=lambda item: item[1]))
        else:  # Default to "FREQ_DESC"
            return dict(
                sorted(
                    frequency.items(),
                    key=lambda item: item[1],
                    reverse=True))

    report = generate_linear_report(df, columns, n, order_by)
    print(json.dumps(report, indent=2))


def retain_columns(
        df: pd.DataFrame,
        columns_to_retain: List[str]) -> pd.DataFrame:
    """
    Retain specified columns in the DataFrame and drop the others.

    Parameters:
        df: DataFrame to modify.
        columns_to_retain: List of column names to retain.

    Returns:
        A new DataFrame with only the retained columns.
    """
    if not isinstance(columns_to_retain, list):
        raise ValueError("columns_to_retain should be a list of column names.")
    return df[columns_to_retain]


def mask_against_dataframe(
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        column_name: str) -> pd.DataFrame:
    """
    Retain only rows with common column values between two DataFrames.

    Parameters:
        df: DataFrame to modify.
        other_df: DataFrame to compare against.
        column_name: Column name to compare.

    Returns:
        A new DataFrame with rows whose column value exist in both DataFrames.
    """
    if column_name not in df.columns or column_name not in other_df.columns:
        raise ValueError("The specified column must exist in both DataFrames.")
    return df[df[column_name].isin(other_df[column_name])]


def mask_against_dataframe_converse(
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        column_name: str) -> pd.DataFrame:
    """
    Retain only rows with uncommon column values between two DataFrames.

    Parameters:
        df: The primary DataFrame to modify.
        other_df: The DataFrame to compare against.
        column_name: The column name to use for comparison.

    Returns:
        A new DataFrame with rows whose column values do not exist in
        'other_df'.
    """
    if column_name not in df.columns or column_name not in other_df.columns:
        raise ValueError("The specified column must exist in both DataFrames.")

    return df[~df[column_name].isin(other_df[column_name])]


def union_join(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a union join, concatenating the two DataFrames and dropping duplicates.

    Parameters:
        df1: First DataFrame.
        df2: Second DataFrame.

    Returns:
        A new DataFrame with the union of df1 and df2, without duplicates.

    Raises:
        ValueError: If the DataFrames do not have the same columns.
    """
    if set(df1.columns) != set(df2.columns):
        raise ValueError(
            "Both DataFrames must have the same columns for a union join")

    result_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
    return result_df


def bag_union_join(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a bag union join, concatenating the two DataFrames without dropping duplicates.

    Parameters:
        df1: First DataFrame.
        df2: Second DataFrame.

    Returns:
        A new DataFrame with the concatenated data of df1 and df2.

    Raises:
        ValueError: If the DataFrames do not have the same columns.
    """
    if set(df1.columns) != set(df2.columns):
        raise ValueError(
            "Both DataFrames must have the same columns for a bag union join")

    result_df = pd.concat([df1, df2], ignore_index=True)
    return result_df


def left_join(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        left_on: str,
        right_on: str) -> pd.DataFrame:
    """
    Perform a left join on two DataFrames.

    Parameters:
        df1: The left DataFrame.
        df2: The right DataFrame.
        left_on: Column name in df1 to join on.
        right_on: Column name in df2 to join on.

    Returns:
        A new DataFrame as the result of a left join.
    """
    return df1.merge(df2, how='left', left_on=left_on, right_on=right_on)


def right_join(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        left_on: str,
        right_on: str) -> pd.DataFrame:
    """
    Perform a right join on two DataFrames.

    Parameters:
        df1: The left DataFrame.
        df2: The right DataFrame.
        left_on: Column name in df1 to join on.
        right_on: Column name in df2 to join on.

    Returns:
        A new DataFrame as the result of a right join.
    """
    return df1.merge(df2, how='right', left_on=left_on, right_on=right_on)


def insert_dataframe_in_sqlite_database(db_path: str, tablename: str, df: pd.DataFrame) -> None:
    """
    Inserts a Pandas DataFrame into a SQLite database table.

    Parameters:
        db_path: str
            The file path to the SQLite database. If the database does not exist,
            it will be created.

        tablename: str
            The name of the table where the data will be inserted. If the table does
            not exist, it will be created based on the DataFrame's columns and types.

        df: pd.DataFrame
            The DataFrame containing the data to be inserted into the database.

    Functionality:
        - Checks if the specified table exists in the database.
        - Creates the table with appropriate column types if it doesn't exist.
        - Inserts the DataFrame's data into the table, appending to any existing data.

    Data Type Mapping:
        - Converts Pandas data types to SQLite types: 'int64' to 'INTEGER',
          'float64' to 'REAL', 'object' to 'TEXT', 'datetime64[ns]' to 'TEXT',
          and 'bool' to 'INTEGER'.

    Returns:
        None
    """

    def table_exists(cursor, table_name):
        cursor.execute(
            f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        return cursor.fetchone()[0] == 1

    dtype_mapping = {
        'int64': 'INTEGER',
        'float64': 'REAL',
        'object': 'TEXT',
        'datetime64[ns]': 'TEXT',
        'bool': 'INTEGER',
    }

    def map_dtype(dtype):
        return dtype_mapping.get(str(dtype), 'TEXT')

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        if not table_exists(cursor, tablename):
            columns_with_types = ', '.join(
                f'"{col}" {
                    map_dtype(dtype)}' for col,
                dtype in zip(
                    df.columns,
                    df.dtypes))
            create_table_query = f'CREATE TABLE "{tablename}" ({columns_with_types})'
            conn.execute(create_table_query)

        df.to_sql(tablename, conn, if_exists='append', index=False)


def sync_dataframe_to_sqlite_database(
        db_path: str,
        tablename: str,
        df: pd.DataFrame) -> None:
    """
    Processes and saves a DataFrame to an SQLite database, adding a timestamp column
    and replacing the existing table if needed. Creates the table if it does not exist.

    Parameters:
    - db_path (str): Path to the SQLite database file.
    - tablename (str): The name of the table in the database.
    - df (pd.DataFrame): The DataFrame to be processed and saved.
    """
    # Helper function to map pandas dtype to SQLite type
    def map_dtype(dtype):
        return dtype_mapping.get(str(dtype), 'TEXT')

    # Step 1: Add a timestamp column to the dataframe
    df['rgwfuncs_sync_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Define a simple mapping from pandas dtypes to SQLite types
    dtype_mapping = {
        'int64': 'INTEGER',
        'float64': 'REAL',
        'object': 'TEXT',
        'datetime64[ns]': 'TEXT',  # Dates are stored as text in SQLite
        'bool': 'INTEGER',  # SQLite does not have a separate Boolean storage class
    }

    # Step 2: Save df in SQLite3 db as '{tablename}_new'
    with sqlite3.connect(db_path) as conn:
        new_table_name = f"{tablename}_new"

        # Check if the new table already exists, create if not
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({new_table_name})")
        if cursor.fetchall() == []:  # Table does not exist
            # Create a table using the DataFrame's column names and types
            columns_with_types = ', '.join(
                f'"{col}" {
                    map_dtype(dtype)}' for col,
                dtype in zip(
                    df.columns,
                    df.dtypes))
            create_table_query = f'CREATE TABLE "{new_table_name}" ({columns_with_types})'
            conn.execute(create_table_query)

        # Insert data into the new table
        df.to_sql(new_table_name, conn, if_exists='replace', index=False)

        # Step 3: If '{tablename}_new' is not empty, delete table '{tablename}' (if it exists), and rename '{tablename}_new' to '{tablename}'
        # Check if the new table is not empty
        cursor.execute(f"SELECT COUNT(*) FROM {new_table_name}")
        count = cursor.fetchone()[0]

        if count > 0:
            # Drop the old table if it exists
            conn.execute(f"DROP TABLE IF EXISTS {tablename}")
            # Rename the new table to the old table name
            conn.execute(f"ALTER TABLE {new_table_name} RENAME TO {tablename}")
