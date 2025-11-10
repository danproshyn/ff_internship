import numpy as np
import pandas as pd


def get_series_first_mode_or_nan(s):
    return s.mode().iloc[0] if not s.mode().empty else np.nan


def read_transactions(path):
    transactions_dtypes = {
       'CLIENT_ID': 'uint64',
       'CAT_C2': 'category',
       'CAT_C3': 'category',
       'CAT_C4': 'category',
       'FL_C6': 'bool',
       'FL_C7': 'bool',
       'FL_C8': 'bool',
       'FL_C9': 'bool',
       'FL_C10': 'bool',
       'FL_C11': 'bool',
       'FL_C12': 'bool',
       'FL_C13': 'bool',
       'FL_C14': 'bool',
       'FL_C15': 'bool',
       'FLOAT_C16': 'float32',
       'FLOAT_C17': 'float32',
       'FLOAT_C18': 'float32',
       'INT_C19': 'int32',
       'FLOAT_C20': 'float32',
       'FLOAT_C21': 'float32'
    }

    transactions = pd.read_csv(
       path,
       sep=',',
       dtype=transactions_dtypes,
       parse_dates=["TRAN_DATE"],
    )

    # Rename columns to be lowercase
    transactions = transactions.rename(columns=str.lower)

    # Convert bool columns to 0 and 1
    transactions = transactions.astype({col: 'int8' for col in transactions.columns if col.startswith('fl_')})

    # Convert categorical columns to int
    transactions = transactions.astype({col: 'int32' for col in transactions.columns if col.startswith('cat_')})

    return transactions


def read_app_activity(path):
    app_activity_dtypes = {
       'CLIENT_ID': 'uint64',
       'DEVICE_ID': 'uint64',
       'CAT_C3': 'Int32',
       'CAT_C4': 'Int32',
       'CAT_C5': 'Int32',
       'CAT_C6': 'Int32',
       'CAT_C8': 'boolean',
       'CAT_C9': 'Int32',
       'CAT_C10': 'boolean',
       'FLOAT_C11': 'float32',
       'FLOAT_C12': 'float32',
       'FLOAT_C13': 'float32',
       'FLOAT_C14': 'float32',
       'FLOAT_C15': 'float32',
       'FLOAT_C16': 'float32',
       'FLOAT_C17': 'float32'
    }

    app_activity = pd.read_csv(
       path,
       sep=',',
       dtype=app_activity_dtypes,
       parse_dates=["ACTIVITY_DATE"],
    )

    # Rename columns to be lowercase
    app_activity = app_activity.rename(columns=str.lower)

    # Convert bool columns to 0 and 1
    app_activity = app_activity.astype({col: 'Int8' for col in ('cat_c8', 'cat_c10')})

    return app_activity


def handle_activity_null_values(df):
    # Drop columns with NULL-values > 40%
    df = df.drop(columns=['float_c13', 'float_c15', 'float_c16', 'float_c17'])

    # Drop rows with more than 2 NULL-values in categorical columns
    df = df.dropna(subset=['cat_c3', 'cat_c8', 'cat_c9', 'cat_c10'])

    # Fill the most frequent value into "cat_c4" column gaps
    cat_c4_fill_val = df['cat_c4'].value_counts().idxmax()
    df['cat_c4'] = df['cat_c4'].fillna(cat_c4_fill_val)

    # Replace nullable dtypes with not-null
    df = df.astype({col: 'int8' for col in df.select_dtypes(include=['Int8']).columns})
    df = df.astype({col: 'int32' for col in df.select_dtypes(include=['Int32']).columns})

    return df


def read_communications(path):
    communications_dtypes = {
       'CLIENT_ID': 'uint64',
       'CAT_C2': 'category',
       'CAT_C3': 'int32',
       'CAT_C4': 'int32',
       'CAT_C5': 'category',

    }

    communications = pd.read_csv(
       path,
       sep=',',
       dtype=communications_dtypes,
       parse_dates=["CONTACT_DATE"],
    )

    communications = communications.rename(columns=str.lower)
    communications = communications.dropna()

    return communications


def encode_comm_categories(df):
    for col in ('cat_c2', 'cat_c5'):
        mapping = {val: idx for idx, val in enumerate(np.sort(df[col].unique()))}
        df[col] = df[col].map(mapping).astype('int32')
    return df
