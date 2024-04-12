import pandas as pd
df = pd.read_parquet('phishingURL.parquet')
df.to_csv('phishingURL.csv')