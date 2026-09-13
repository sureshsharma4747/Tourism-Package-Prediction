import pandas as pd

DATA_PATH = "data/tourism.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["ProdTaken"].value_counts())