import pandas as pd

df = pd.read_csv("all_ecoff_values.csv")

df = df[df['(T)ECOFF'] != '-']
df = df[df['(T)ECOFF'] != 'ID']

df = df[~df['(T)ECOFF'].str.contains('(', regex=False)]

df['(T)ECOFF'] = pd.to_numeric(df['(T)ECOFF'])

df.to_csv("reduced_ecoff_values.csv", index=False)
