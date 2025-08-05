import pandas as pd
df = pd.read_csv('data/Wednesday-workingHours.pcap_ISCX.csv')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
