import pandas as pd


first_chunk = True
for chunk in pd.read_csv('data/initial/APP_ACTIVITY.csv', chunksize=100_000):
    sample = chunk.sample(frac=0.05)
    sample.to_csv('data/samples/APP_ACTIVITY_SAMPLE.csv', mode='a', index=False, header=first_chunk)
    first_chunk = False
