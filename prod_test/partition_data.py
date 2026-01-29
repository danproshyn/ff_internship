import os

from utils import read_clients, read_transactions, read_communications, read_app_activity


CHUNKS_NUM = 54

INPUT_DIR = 'data/raw/'
OUTPUT_DIR = 'data/raw_chunks/'

CLIENTS_PATH = os.path.join(INPUT_DIR, 'CLIENTS.csv')
TRANSACTIONS_PATH = os.path.join(INPUT_DIR, 'TRANSACTIONS.csv')
ACTIVITIES_PATH = os.path.join(INPUT_DIR, 'APP_ACTIVITY.csv')
COMMUNICATIONS_PATH = os.path.join(INPUT_DIR, 'COMMUNICATIONS.csv')


df_clients = read_clients(CLIENTS_PATH, encode_bool=False)
df_transactions = read_transactions(TRANSACTIONS_PATH, encode_bool=False, encode_category=False)
df_activities = read_app_activity(ACTIVITIES_PATH, encode_bool=False, encode_category=False)
df_communications = read_communications(COMMUNICATIONS_PATH, encode_category=False)


clients_chunk_size = int(df_clients.shape[0] / CHUNKS_NUM) + 1
clients_chunks = [
    df_clients.iloc[i:i+clients_chunk_size, :]
    for i in range(0, df_clients.shape[0], clients_chunk_size)
]

for count, cl_chunk in enumerate(clients_chunks, start=1):
    print(f'Processing chunk {count} of {len(clients_chunks)}...')

    tx_chunk = df_transactions[df_transactions['client_id'].isin(cl_chunk['client_id'])]
    act_chunk = df_activities[df_activities['client_id'].isin(cl_chunk['client_id'])]
    comm_chunk = df_communications[df_communications['client_id'].isin(cl_chunk['client_id'])]

    cl_chunk = cl_chunk.rename(columns=str.upper)
    tx_chunk = tx_chunk.rename(columns=str.upper)
    act_chunk = act_chunk.rename(columns=str.upper)
    comm_chunk = comm_chunk.rename(columns=str.upper)

    chunk_dir = os.path.join(OUTPUT_DIR, str(count))
    os.makedirs(chunk_dir, exist_ok=True)

    cl_chunk.to_csv(os.path.join(chunk_dir, 'CLIENTS.csv'), index=False)
    tx_chunk.to_csv(os.path.join(chunk_dir, 'TRANSACTIONS.csv'), index=False)
    act_chunk.to_csv(os.path.join(chunk_dir, 'APP_ACTIVITY.csv'), index=False)
    comm_chunk.to_csv(os.path.join(chunk_dir, 'COMMUNICATIONS.csv'), index=False)
