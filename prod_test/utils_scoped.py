import json
import re

import matplotlib.pyplot as plt
import pandas as pd
from feature_engine.creation import CyclicalFeatures
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


def read_clients(path, encode_bool=True):
    clients_dtypes = {
        'CLIENT_ID': 'uint64',
        'TARGET': 'bool',
        'IS_TRAIN': 'bool',
    }
    clients = pd.read_csv(
        path,
        dtype=clients_dtypes,
        parse_dates=['COMMUNICATION_MONTH'],
    )

    # Rename columns to be lowercase
    clients = clients.rename(columns=str.lower)

    # Convert bool columns to 0 and 1
    if encode_bool:
        clients = clients.astype({col: 'int8' for col in ('target', 'is_train')})

    return clients


def read_transactions(path, encode_bool=True, encode_category=True):
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
    if encode_bool:
        transactions = transactions.astype({col: 'int8' for col in transactions.columns if col.startswith('fl_')})

    # Convert categorical columns to int
    if encode_category:
        transactions = transactions.astype({col: 'int32' for col in transactions.columns if col.startswith('cat_')})

    return transactions


def read_app_activity(path, encode_bool=True, encode_category=True):
    app_activity_dtypes = {
       'CLIENT_ID': 'uint64',
       'DEVICE_ID': 'uint64',
       'CAT_C3': 'category',
       'CAT_C4': 'category',
       'CAT_C5': 'category',
       'CAT_C6': 'category',
       'CAT_C8': 'boolean',
       'CAT_C9': 'category',
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
    if encode_bool:
        app_activity = app_activity.astype({col: 'Int8' for col in ('cat_c8', 'cat_c10')})

    # Convert categorical columns to int
    if encode_category:
        app_activity = app_activity.astype({col: 'int32' for col in app_activity.select_dtypes(include=['category']).columns})

    return app_activity


def preprocess_app_activity_data(df):
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


def read_communications(path, encode_category=True):
    communications_dtypes = {
       'CLIENT_ID': 'uint64',
       'CAT_C2': 'category',
       'CAT_C3': 'category',
       'CAT_C4': 'category',
       'CAT_C5': 'category',

    }

    communications = pd.read_csv(
       path,
       sep=',',
       dtype=communications_dtypes,
       parse_dates=["CONTACT_DATE"],
    )

    communications = communications.rename(columns=str.lower)

    # Convert categorical columns to int
    if encode_category:
        communications = communications.astype({col: 'int32' for col in ('cat_c3', 'cat_c4')})

    return communications


def add_calendar_values(df, date_col, prefix=None):
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_month'] = df[date_col].dt.day
    df['is_weekend'] = df['day_of_week'].apply(lambda x: x >= 5)

    # Encode cyclical values
    time_cols = ['day_of_week', 'day_of_month']
    df = df.astype({col: 'int8' for col in time_cols})
    cyclical = CyclicalFeatures(variables=time_cols, drop_original=True)
    df = cyclical.fit_transform(df)

    # Add prefix
    if prefix:
        for col in ('day_of_week_sin', 'day_of_week_cos', 'day_of_month_sin', 'day_of_month_cos', 'is_weekend'):
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            df = df.rename(columns={col: prefix + col})

    return df


def write_df_dtype(df, path):
    with open(path, 'w') as f:
        json.dump(df.dtypes.astype(str).to_dict(), f)


def read_df_dtype(path):
    with open(path, 'r') as f:
        dtype = json.load(f)
    date_cols = [col for col, dtype in dtype.items() if dtype == 'datetime64[ns]']
    dtype = {col: dtype for col, dtype in dtype.items() if col not in date_cols}

    return dtype, date_cols


################################################################################
# Model utils
################################################################################

def split_dataset_v2(path, valid_size=0.2, replace_special_symbols=False):
    dataset = pd.read_csv(path)
    dataset.drop(columns=['client_id'], inplace=True)

    if replace_special_symbols:
        dataset = dataset.rename(columns=lambda x: re.sub(r'[,\n\[\]\{\}:"]', '__', x))

    train_dataset = dataset[dataset['is_train'] == 1].drop(columns=['is_train'])
    test_dataset = dataset[dataset['is_train'] == 0].drop(columns=['is_train'])

    X = train_dataset.drop(columns=['target'])
    X_test = test_dataset.drop(columns=['target'])
    y = train_dataset['target']
    y_test = test_dataset['target']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, stratify=y, random_state=42)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def calc_gini_coef(y_true, y_pred_proba):
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * roc_auc - 1


def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.plot(fpr, tpr, label='auc=' + str(roc_auc))
    plt.legend(loc='lower right')
    plt.title('ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


def plot_pr_curve(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.plot(recall, precision)
    plt.title('PR-curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
