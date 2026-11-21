import pandas as pd

df = pd.read_csv("data/raw/train.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

numeric_columns = df.select_dtypes(include="number").columns
all_dates = pd.date_range(df["date"].min(), df["date"].max())

min_days = 14

for col in numeric_columns:
    daily = df.groupby("date")[col].agg(nunique="nunique", value="first")

    start = None
    previous_date = None
    current_value = None
    count = 0

    print(f"\n========== {col} ==========")

    for date in all_dates:
        if date in daily.index and daily.loc[date, "nunique"] == 1:
            value = daily.loc[date, "value"]

            if (
                previous_date is not None
                and (date - previous_date).days == 1
                and value == current_value
            ):
                count += 1
            else:
                if count >= min_days:
                    print("Value:", current_value)
                    print("Start:", start.date())
                    print("End:", previous_date.date())
                    print("Consecutive days:", count)

                start = date
                current_value = value
                count = 1

            previous_date = date

        else:
            if count >= min_days:
                print("Value:", current_value)
                print("Start:", start.date())
                print("End:", previous_date.date())
                print("Consecutive days:", count)

            start = None
            previous_date = None
            current_value = None
            count = 0

    if count >= min_days:
        print("Value:", current_value)
        print("Start:", start.date())
        print("End:", previous_date.date())
        print("Consecutive days:", count)
