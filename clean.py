import pandas as pd
df = pd.read_csv('data/Wednesday-workingHours.pcap_ISCX.csv')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.dropna(inplace=True)
df['Attack'] = df[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
df.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], inplace=True, errors='ignore')
