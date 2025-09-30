import pandas as pd

df = pd.read_csv("data/twitter_validation.csv")
print("🧾 CSV Columns:", df.columns.tolist())
print("🔍 First few rows:")
print(df.head())
