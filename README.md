RGWML

***By Ryan Gerard Wilson (https://ryangerardwilson.com)***

# RGWFuncs

This library provides a variety of functions for manipulating and analyzing pandas DataFrames. 

--------------------------------------------------------------------------------

## Installation

Install the package using:
```bash
pip install rgwfuncs
```

--------------------------------------------------------------------------------

## Basic Usage

Import the library:
    ```
    import rgwfuncs
    ```

View available function docstrings in alphabetical order:
    ```
    rgwfuncs.docs()
    ```

View specific docstrings by providing a filter (comma-separated). For example, to display docstrings about "numeric_clean":
    ```
    rgwfuncs.docs(method_type_filter='numeric_clean')
    ```

To display all docstrings, use:
    ```
    rgwfuncs.docs(method_type_filter='*')
    ```

--------------------------------------------------------------------------------

## Function References and Syntax Examples

Below is a quick reference of available functions, their purpose, and basic usage examples.

### 1. docs
Print a list of available function names in alphabetical order. If a filter is provided, print the matching docstrings.

• Parameters:
  - `method_type_filter` (str): Optional, comma-separated to select docstring types, or '*' for all.

• Example:

    import rgwfuncs
    rgwfuncs.docs(method_type_filter='numeric_clean,limit_dataframe')

--------------------------------------------------------------------------------

### 2. `numeric_clean`
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

### 3. `limit_dataframe`
Limit the DataFrame to a specified number of rows.

• Parameters:
  - df (pd.DataFrame): The DataFrame to limit.
  - `num_rows` (int): The number of rows to retain.

• Returns:
  - pd.DataFrame: A new DataFrame limited to the specified number of rows.

• Example:
    ``` 
    from rgwfuncs import limit_dataframe
    import pandas as pd

    df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})
    df_limited = limit_dataframe(df, 5)
    print(df_limited)
    ```
--------------------------------------------------------------------------------

### 4. `from_raw_data`
Create a DataFrame from raw data.

• Parameters:
  - headers (list): A list of column headers.
  - data (list of lists): A two-dimensional list of data.

• Returns:
  - pd.DataFrame: A DataFrame created from the raw data.

• Example:
    ```
    from rgwfuncs import from_raw_data

    headers = ["Name", "Age"]
    data = [
        ["Alice", 30],
        ["Bob", 25],
        ["Charlie", 35]
    ]

    df = from_raw_data(headers, data)
    print(df)
    ```
--------------------------------------------------------------------------------

### 5. `append_rows`
Append rows to the DataFrame.

• Parameters:
  - df (pd.DataFrame): The original DataFrame.
  - rows (list of lists): Each inner list represents a row to be appended.

• Returns:
  - pd.DataFrame: A new DataFrame with appended rows.

• Example:
    ```
    from rgwfuncs import append_rows
    import pandas as pd

    df = pd.DataFrame({'Name': ['Alice'], 'Age': [30]})
    new_rows = [
        ['Bob', 25],
        ['Charlie', 35]
    ]
    df_appended = append_rows(df, new_rows)
    print(df_appended)
    ```
--------------------------------------------------------------------------------

### 6. `append_columns`
Append new columns to the DataFrame with None values.

• Parameters:
  - df (pd.DataFrame): The original DataFrame.
  - `col_names` (list or comma-separated string): The names of the columns to add.

• Returns:
  - pd.DataFrame: A new DataFrame with the new columns appended.

• Example:
    ```
    from rgwfuncs import append_columns
    import pandas as pd

    df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [30, 25]})
    df_new = append_columns(df, ['Salary', 'Department'])
    print(df_new)
    ```
--------------------------------------------------------------------------------

### 7. `update_rows`
Update specific rows in the DataFrame based on a condition.

• Parameters:
  - df (pd.DataFrame): The original DataFrame.
  - condition (str): A query condition to identify rows for updating.
  - updates (dict): A dictionary with column names as keys and new values as values.

• Returns:
  - pd.DataFrame: A new DataFrame with updated rows.

• Example:
    ```
    from rgwfuncs import update_rows
    import pandas as pd

    df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [30, 25]})
    df_updated = update_rows(df, "Name == 'Alice'", {'Age': 31})
    print(df_updated)
    ```
--------------------------------------------------------------------------------

### 8. `delete_rows`
Delete rows from the DataFrame based on a condition.

• Parameters:
  - df (pd.DataFrame): The original DataFrame.
  - condition (str): A query condition to identify rows for deletion.

• Returns:
  - pd.DataFrame: The DataFrame with specified rows deleted.

• Example:
    ```
    from rgwfuncs import delete_rows
    import pandas as pd

    df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [30, 25]})
    df_deleted = delete_rows(df, "Age < 28")
    print(df_deleted)
    ```
--------------------------------------------------------------------------------

### 9. `drop_duplicates`
Drop duplicate rows in the DataFrame, retaining the first occurrence.

• Parameters:
  - df (pd.DataFrame): The DataFrame from which duplicates will be dropped.

• Returns:
  - pd.DataFrame: A new DataFrame with duplicates removed.

• Example:
    ```
    from rgwfuncs import drop_duplicates
    import pandas as pd

    df = pd.DataFrame({'A': [1,1,2,2], 'B': [3,3,4,4]})
    df_no_dupes = drop_duplicates(df)
    print(df_no_dupes)
    ```
--------------------------------------------------------------------------------

### 10. `drop_duplicates_retain_first`
Drop duplicate rows based on specified columns, retaining the first occurrence.

• Parameters:
  - df (pd.DataFrame): The DataFrame from which duplicates will be dropped.
  - columns (str): Comma-separated string with column names used to identify duplicates.

• Returns:
  - pd.DataFrame: A new DataFrame with duplicates removed.

• Example:
    ```
    from rgwfuncs import drop_duplicates_retain_first
    import pandas as pd

    df = pd.DataFrame({'A': [1,1,2,2], 'B': [3,3,4,4]})
    df_no_dupes = drop_duplicates_retain_first(df, 'A')
    print(df_no_dupes)
    ```
--------------------------------------------------------------------------------

### 11. `drop_duplicates_retain_last`
Drop duplicate rows based on specified columns, retaining the last occurrence.

• Parameters:
  - df (pd.DataFrame): The DataFrame from which duplicates will be dropped.
  - columns (str): Comma-separated string with column names used to identify duplicates.

• Returns:
  - pd.DataFrame: A new DataFrame with duplicates removed.

• Example:
    ```
    from rgwfuncs import drop_duplicates_retain_last
    import pandas as pd

    df = pd.DataFrame({'A': [1,1,2,2], 'B': [3,3,4,4]})
    df_no_dupes = drop_duplicates_retain_last(df, 'A')
    print(df_no_dupes)
    ```

--------------------------------------------------------------------------------

### 12. `load_data_from_query`
Load data from a database query into a DataFrame based on a configuration preset.

• Parameters:
  - `db_preset_name` (str): Name of the database preset in the config file.
  - query (str): The SQL query to execute.
  - `config_file_name` (str): Name of the configuration file (default: "rgwml.config").

• Returns:
  - pd.DataFrame: A DataFrame containing the query result.

• Example:
    ```
    from rgwfuncs import load_data_from_query

    df = load_data_from_query(
        db_preset_name="MyDBPreset",
        query="SELECT * FROM my_table",
        config_file_name="rgwml.config"
    )
    print(df)
    ```

--------------------------------------------------------------------------------

### 13. `load_data_from_path`
Load data from a file into a DataFrame based on the file extension.

• Parameters:
  - `file_path` (str): The absolute path to the data file.

• Returns:
  - pd.DataFrame: A DataFrame containing the loaded data.

• Example:
    ```
    from rgwfuncs import load_data_from_path

    df = load_data_from_path("/absolute/path/to/data.csv")
    print(df)
    ```

--------------------------------------------------------------------------------

### 14. `load_data_from_sqlite_path`
Execute a query on a SQLite database file and return the results as a DataFrame.

• Parameters:
  - `sqlite_path` (str): The absolute path to the SQLite database file.
  - query (str): The SQL query to execute.

• Returns:
  - pd.DataFrame: A DataFrame containing the query results.

• Example:
    ```
    from rgwfuncs import load_data_from_sqlite_path

    df = load_data_from_sqlite_path("/path/to/database.db", "SELECT * FROM my_table")
    print(df)
    ```

--------------------------------------------------------------------------------

### 15. `first_n_rows`
Display the first n rows of the DataFrame (prints out in dictionary format).

• Parameters:
  - df (pd.DataFrame)
  - n (int): Number of rows to display.

• Example:
    ```
    from rgwfuncs import first_n_rows
    import pandas as pd

    df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
    first_n_rows(df, 2)
    ```

--------------------------------------------------------------------------------

### 16. `last_n_rows`
Display the last n rows of the DataFrame (prints out in dictionary format).

• Parameters:
  - df (pd.DataFrame)
  - n (int): Number of rows to display.

• Example:
    ```
    from rgwfuncs import last_n_rows
    import pandas as pd

    df = pd.DataFrame({'A': [1,2,3,4,5], 'B': [6,7,8,9,10]})
    last_n_rows(df, 2)
    ```

--------------------------------------------------------------------------------

### 17. `top_n_unique_values`
Print the top n unique values for specified columns in the DataFrame.

• Parameters:
  - df (pd.DataFrame): The DataFrame to evaluate.
  - n (int): Number of top values to display.
  - columns (list): List of columns for which to display top unique values.

• Example:
    ```
    from rgwfuncs import top_n_unique_values
    import pandas as pd

    df = pd.DataFrame({'Cities': ['NY', 'LA', 'NY', 'SF', 'LA', 'LA']})
    top_n_unique_values(df, 2, ['Cities'])
    ```

--------------------------------------------------------------------------------

### 18. `bottom_n_unique_values`
Print the bottom n unique values for specified columns in the DataFrame.

• Parameters:
  - df (pd.DataFrame)
  - n (int)
  - columns (list)

• Example:
    ```
    from rgwfuncs import bottom_n_unique_values
    import pandas as pd

    df = pd.DataFrame({'Cities': ['NY', 'LA', 'NY', 'SF', 'LA', 'LA']})
    bottom_n_unique_values(df, 1, ['Cities'])
    ```

--------------------------------------------------------------------------------

### 19. `print_correlation`
Print correlation for multiple pairs of columns in the DataFrame.

• Parameters:
  - df (pd.DataFrame)
  - `column_pairs` (list of tuples): E.g., `[('col1','col2'), ('colA','colB')]`.

• Example:
    ```
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
    ```

--------------------------------------------------------------------------------

### 20. `print_memory_usage`
Print the memory usage of the DataFrame in megabytes.

• Parameters:
  - df (pd.DataFrame)

• Example:
    ```
    from rgwfuncs import print_memory_usage
    import pandas as pd

    df = pd.DataFrame({'A': range(1000)})
    print_memory_usage(df)
    ```

--------------------------------------------------------------------------------

### 21. `filter_dataframe`
Return a new DataFrame filtered by a given query expression.

• Parameters:
  - df (pd.DataFrame)
  - `filter_expr` (str)

• Returns:
  - pd.DataFrame

• Example:
    ```
    from rgwfuncs import filter_dataframe
    import pandas as pd

    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [30, 20, 25]
    })

    df_filtered = filter_dataframe(df, "Age > 23")
    print(df_filtered)
    ```

--------------------------------------------------------------------------------

### 22. `filter_indian_mobiles`
Filter and return rows containing valid Indian mobile numbers in the specified column.

• Parameters:
  - df (pd.DataFrame)
  - `mobile_col` (str): The column name with mobile numbers.

• Returns:
  - pd.DataFrame

• Example:
    ```
    from rgwfuncs import filter_indian_mobiles
    import pandas as pd

    df = pd.DataFrame({'Phone': ['9876543210', '12345', '7000012345']})
    df_indian = filter_indian_mobiles(df, 'Phone')
    print(df_indian)
    ```

--------------------------------------------------------------------------------

### 23. `print_dataframe`
Print the entire DataFrame and its column types. Optionally print a source path.

• Parameters:
  - df (pd.DataFrame)
  - source (str, optional)

• Example:
    ```
    from rgwfuncs import print_dataframe
    import pandas as pd

    df = pd.DataFrame({'Name': ['Alice'], 'Age': [30]})
    print_dataframe(df, source='SampleData.csv')
    ```

--------------------------------------------------------------------------------

### 24. `send_dataframe_via_telegram`
Send a DataFrame via Telegram using a specified bot configuration.

• Parameters:
  - df (pd.DataFrame)
  - `bot_name` (str)
  - message (str)
  - `as_file` (bool)
  - `remove_after_send` (bool)

• Example:
    ```
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
    ```

--------------------------------------------------------------------------------

### 25. `send_data_to_email`
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
    ```
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
    ```

--------------------------------------------------------------------------------

### 26. `send_data_to_slack`
Send a DataFrame or message to Slack using a specified bot configuration.

• Parameters:
  - df (pd.DataFrame)
  - `bot_name` (str)
  - message (str)
  - `as_file` (bool)
  - `remove_after_send` (bool)

• Example:
    ```
    from rgwfuncs import send_data_to_slack

    df = ...  # Some DataFrame
    send_data_to_slack(
        df,
        bot_name='MySlackBot',
        message='Hello Slack!',
        as_file=True,
        remove_after_send=True
    )
    ```

--------------------------------------------------------------------------------

### 27. `order_columns`
Reorder the columns of a DataFrame based on a string input.

• Parameters:
  - df (pd.DataFrame)
  - `column_order_str` (str): Comma-separated column order.

• Returns:
  - pd.DataFrame

• Example:
    ```
    from rgwfuncs import order_columns
    import pandas as pd

    df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [30, 25], 'Salary': [1000, 1200]})
    df_reordered = order_columns(df, 'Salary,Name,Age')
    print(df_reordered)
    ```

--------------------------------------------------------------------------------

### 28. `append_ranged_classification_column`
Append a ranged classification column to the DataFrame.

• Parameters:
  - df (pd.DataFrame)
  - ranges (str): Ranges separated by commas (e.g., "0-10,11-20,21-30").
  - `target_col` (str): The column to classify.
  - `new_col_name` (str): Name of the new classification column.

• Returns:
  - pd.DataFrame

• Example:
    ```
    from rgwfuncs import append_ranged_classification_column
    import pandas as pd

    df = pd.DataFrame({'Scores': [5, 12, 25]})
    df_classified = append_ranged_classification_column(df, '0-10,11-20,21-30', 'Scores', 'ScoreRange')
    print(df_classified)
    ```

--------------------------------------------------------------------------------

### 29. `append_percentile_classification_column`
Append a percentile classification column to the DataFrame.

• Parameters:
  - df (pd.DataFrame)
  - percentiles (str): Percentile values separated by commas (e.g., "25,50,75").
  - `target_col` (str)
  - `new_col_name` (str)

• Returns:
  - pd.DataFrame

• Example:
    ```
    from rgwfuncs import append_percentile_classification_column
    import pandas as pd

    df = pd.DataFrame({'Values': [10, 20, 30, 40, 50]})
    df_classified = append_percentile_classification_column(df, '25,50,75', 'Values', 'ValuePercentile')
    print(df_classified)
    ```

--------------------------------------------------------------------------------

### 30. `append_ranged_date_classification_column`
Append a ranged date classification column to the DataFrame.

• Parameters:
  - df (pd.DataFrame)
  - `date_ranges` (str): Date ranges separated by commas, e.g., `2020-01-01_2020-06-30,2020-07-01_2020-12-31`
  - `target_col` (str)
  - `new_col_name` (str)

• Returns:
  - pd.DataFrame

• Example:
    ```
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
    ```

--------------------------------------------------------------------------------

### 31. `rename_columns`
Rename columns in the DataFrame.

• Parameters:
  - df (pd.DataFrame)
  - `rename_pairs` (dict): Mapping old column names to new ones.

• Returns:
  - pd.DataFrame

• Example:
    ```
    from rgwfuncs import rename_columns
    import pandas as pd

    df = pd.DataFrame({'OldName': [1,2,3]})
    df_renamed = rename_columns(df, {'OldName': 'NewName'})
    print(df_renamed)
    ```

--------------------------------------------------------------------------------

### 32. `cascade_sort`
Cascade sort the DataFrame by specified columns and order.

• Parameters:
  - df (pd.DataFrame)
  - columns (list): e.g. ["Column1::ASC", "Column2::DESC"].

• Returns:
  - pd.DataFrame

• Example:
    ```
    from rgwfuncs import cascade_sort
    import pandas as pd

    df = pd.DataFrame({
        'Name': ['Charlie', 'Alice', 'Bob'],
        'Age': [25, 30, 22]
    })

    sorted_df = cascade_sort(df, ["Name::ASC", "Age::DESC"])
    print(sorted_df)
    ```

--------------------------------------------------------------------------------

### 33. `append_xgb_labels`
Append XGB training labels (TRAIN, VALIDATE, TEST) based on a ratio string.

• Parameters:
  - df (pd.DataFrame)
  - `ratio_str` (str): e.g. "8:2", "7:2:1".

• Returns:
  - pd.DataFrame

• Example:
    ```
    from rgwfuncs import append_xgb_labels
    import pandas as pd

    df = pd.DataFrame({'A': range(10)})
    df_labeled = append_xgb_labels(df, "7:2:1")
    print(df_labeled)
    ```

--------------------------------------------------------------------------------

### 34. `append_xgb_regression_predictions`
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
    ```
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
    ```

--------------------------------------------------------------------------------

### 35. `append_xgb_logistic_regression_predictions`
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
    ```
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
    ```

--------------------------------------------------------------------------------

### 36. `print_n_frequency_cascading`
Print the cascading frequency of top n values for specified columns.

• Parameters:
  - df (pd.DataFrame)
  - n (int)
  - columns (str): Comma-separated column names.
  - `order_by` (str): `ASC`, `DESC`, `FREQ_ASC`, `FREQ_DESC`.

• Example:
    ```
    from rgwfuncs import print_n_frequency_cascading
    import pandas as pd

    df = pd.DataFrame({'City': ['NY','LA','NY','SF','LA','LA']})
    print_n_frequency_cascading(df, 2, 'City', 'FREQ_DESC')
    ```

--------------------------------------------------------------------------------

### 37. `print_n_frequency_linear`
Print the linear frequency of top n values for specified columns.

• Parameters:
  - df (pd.DataFrame)
  - n (int)
  - columns (str): Comma-separated columns.
  - `order_by` (str)

• Example:
    ```
    from rgwfuncs import print_n_frequency_linear
    import pandas as pd

    df = pd.DataFrame({'City': ['NY','LA','NY','SF','LA','LA']})
    print_n_frequency_linear(df, 2, 'City', 'FREQ_DESC')
    ```

--------------------------------------------------------------------------------

### 38. `retain_columns`
Retain specified columns in the DataFrame and drop the others.

• Parameters:
  - df (pd.DataFrame)
  - `columns_to_retain` (list or str)

• Returns:
  - pd.DataFrame

• Example:
    ```
    from rgwfuncs import retain_columns
    import pandas as pd

    df = pd.DataFrame({'A': [1,2], 'B': [3,4], 'C': [5,6]})
    df_reduced = retain_columns(df, ['A','C'])
    print(df_reduced)
    ```

--------------------------------------------------------------------------------

### 39. `mask_against_dataframe`
Retain only rows with common column values between two DataFrames.

• Parameters:
  - df (pd.DataFrame)
  - `other_df` (pd.DataFrame)
  - `column_name` (str)

• Returns:
  - pd.DataFrame

• Example:
    ```
    from rgwfuncs import mask_against_dataframe
    import pandas as pd

    df1 = pd.DataFrame({'ID': [1,2,3], 'Value': [10,20,30]})
    df2 = pd.DataFrame({'ID': [2,3,4], 'Extra': ['X','Y','Z']})

    df_masked = mask_against_dataframe(df1, df2, 'ID')
    print(df_masked)
    ```

--------------------------------------------------------------------------------

### 40. `mask_against_dataframe_converse`
Retain only rows with uncommon column values between two DataFrames.

• Parameters:
  - df (pd.DataFrame)
  - `other_df` (pd.DataFrame)
  - `column_name` (str)

• Returns:
  - pd.DataFrame

• Example:
    ```
    from rgwfuncs import mask_against_dataframe_converse
    import pandas as pd

    df1 = pd.DataFrame({'ID': [1,2,3], 'Value': [10,20,30]})
    df2 = pd.DataFrame({'ID': [2,3,4], 'Extra': ['X','Y','Z']})

    df_uncommon = mask_against_dataframe_converse(df1, df2, 'ID')
    print(df_uncommon)
    ```

--------------------------------------------------------------------------------

## Additional Info

For more information, refer to each function’s docstring by calling:
```
rgwfuncs.docs(method_type_filter='function_name')
```
or display all docstrings with:
```python
rgwfuncs.docs(method_type_filter='*')
```

--------------------------------------------------------------------------------

© 2025 Ryan Gerard Wilson. All rights reserved.

