import os

import pandas as pd

import numpy as np

from tqdm import tqdm

frames = []
i = 0
for filename in tqdm(os.listdir("all_ecoff_values")):
    df = pd.read_csv(f"all_ecoff_values/{filename}")
    if df.columns[0] != "Unnamed: 0":
        print(filename, df.columns[0])
        i += 1
    df.rename(columns={"Unnamed: 0": "Agent"}, inplace=True)
    if len(df.columns) !=24:
        print(filename, len(df.columns))
    assert np.all(df.columns == ['Agent', '0.02', '0.004', '0.008',
                                 '0.016', '0.03', '0.06', '0.125',
                                 '0.25', '0.5', '1', '2', '4', '8',
                                 '16', '32', '64', '128', '256', '512',
                                 'Distributions', 'Observations',
                                 '(T)ECOFF', 'Confidence interval']), filename
    df['Agent'] = filename.replace(".csv", "")
    frames.append(df)

df = pd.concat(frames)
print(df.columns)

print(i, len(os.listdir("all_ecoff_values")))