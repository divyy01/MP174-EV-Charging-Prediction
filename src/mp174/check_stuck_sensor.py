import pandas as pd

df = pd.read_csv("data/raw/train.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

numeric_columns = df.select_dtypes(include="number").columns

found = False

for col in numeric_columns:
    # Identify consecutive records with the same reading.
    groups = df[col].ne(df[col].shift()).cumsum()

    runs = df.groupby(groups).agg(
        value=(col, "first"),
        records=(col, "size"),
        start_date=("date", "min"),
        end_date=("date", "max"),
        unique_dates=("date", "nunique"),
    )

    # Look for readings repeated across at least 7 different dates.
    runs["duration_days"] = (runs["end_date"] - runs["start_date"]).dt.days
    runs = runs[(runs["unique_dates"] >= 7) & (runs["duration_days"] >= 14)]

    if not runs.empty:
        found = True
        print(f"\nCOLUMN: {col}")
        print(runs.to_string(index=False))

if not found:
    print("No repeated readings across 7 or more dates found.")
