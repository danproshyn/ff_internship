import pandas as pd


# df = pd.read_csv('data/initial/CLIENTS.csv')
# df_sample = df.sample(frac=0.05)
# df_sample.to_csv('data/samples/CLIENTS_SAMPLE.csv', index=False)


# IN_FILENAME = 'data/initial/TRANSACTIONS.csv'
# OUT_FILENAME = 'data/samples/TRANSACTIONS_SAMPLE.csv'
# IN_FILENAME = 'data/initial/APP_ACTIVITY.csv'
# OUT_FILENAME = 'data/samples/APP_ACTIVITY_SAMPLE.csv'
IN_FILENAME = 'data/initial/COMMUNICATIONS.csv'
OUT_FILENAME = 'data/samples/COMMUNICATIONS_SAMPLE.csv'


client_ids = pd.read_csv('data/samples/CLIENTS_SAMPLE.csv')['CLIENT_ID'].values
df = pd.read_csv(IN_FILENAME)
res = pd.DataFrame()
for client_id, groups in df.groupby('CLIENT_ID'):
    if client_id in client_ids:
        res = pd.concat([res, groups], ignore_index=True)
res.to_csv(OUT_FILENAME, index=False)
