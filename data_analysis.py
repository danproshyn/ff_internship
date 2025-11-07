import pandas as pd

# read client sample
client_sample_dtypes = {
   'CLIENT_ID': 'uint64',
   'TARGET': 'boolean',
   'IS_TRAIN': 'boolean',
}
clients_sample = pd.read_csv(
   'data/CLIENTS_SAMPLE.csv',
   sep=',',
   dtype=client_sample_dtypes
)

# read app activity
app_activity_dtypes = {
   'CLIENT_ID': 'uint64',
   'DEVICE_ID': 'uint64',
   'CAT_C3': 'category',
   'CAT_C4': 'category',
   'CAT_C5': 'category',
   'CAT_C6': 'category',
   'CAT_C8': 'category',
   'CAT_C9': 'category',
   'CAT_C10': 'category',
   'FLOAT_C11': 'float32',
   'FLOAT_C12': 'float32',
   'FLOAT_C13': 'float32',
   'FLOAT_C14': 'float32',
   'FLOAT_C15': 'float32',
   'FLOAT_C16': 'float32',
   'FLOAT_C17': 'float32'
}

app_activity = pd.read_csv(
   'data/APP_ACTIVITY.csv',
   sep=',',
   dtype=app_activity_dtypes,
   parse_dates=["ACTIVITY_DATE"],
)

# read communications
communications_dtypes = {
   'CLIENT_ID': 'uint64',
   "CAT_C2": "category",
   "CAT_C3": "category",
   "CAT_C4": "category",
   "CAT_C5": "category",

}

communications = pd.read_csv(
   'data/COMMUNICATIONS.csv',
   sep=',',
   dtype=communications_dtypes,
   parse_dates=["CONTACT_DATE"],
)

# read transactions
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
   'data/TRANSACTIONS.csv',
   sep=',',
   dtype=transactions_dtypes,
   parse_dates=["TRAN_DATE"],
)
