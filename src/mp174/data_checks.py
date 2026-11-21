import pandas as pd


def load_data(path):
    """Load the original dataset."""
    return pd.read_csv(path)


def check_missing_values(df):
    """Count missing values in every column."""
    return df.isnull().sum()


def check_duplicates(df):
    """Count exact duplicate rows and duplicate record IDs."""
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_record_ids": int(df["record_id"].duplicated().sum()),
    }


def check_negative_values(df):
    """Count negative values in numerical columns."""
    return (df.select_dtypes(include="number") < 0).sum()


def check_zero_energy(df):
    """Identify records with zero delivered energy."""
    return df[df["energy_kwh"] == 0]


def check_long_sessions(df):
    """Identify unusually long recorded sessions."""
    return df[df["session_minutes"] > 1000]


def check_unit_mixup(df):
    """Compare session duration statistics across station types."""
    return df.groupby("station_type")["session_minutes"].agg(["min", "mean", "max"])


def check_sentinel_candidates(df):
    """Find common sentinel values in numerical columns."""
    candidates = [-1, -999, 999, 9999, -9999, 99999]
    results = {}

    for col in df.select_dtypes(include="number").columns:
        counts = df[col].value_counts()

        matches = counts[counts.index.isin(candidates)]

        if not matches.empty:
            results[col] = matches.to_dict()

    return results


def check_spelling_variants(df):
    """Check category values and frequencies in categorical columns."""
    categorical_columns = ["vehicle", "station_type"]
    results = {}

    for col in categorical_columns:
        if col in df.columns:
            results[col] = df[col].value_counts().to_dict()

    return results


def check_target_distribution(df):
    """Sort target values to inspect unusually large values."""
    return df.sort_values("energy_kwh", ascending=False)[["record_id", "energy_kwh"]]


def check_leaking_column(df, holdout_df):
    """Identify training features missing from holdout, excluding the target."""
    target = "energy_kwh"

    return [
        col for col in df.columns if col != target and col not in holdout_df.columns
    ]


def check_stuck_sensor(df, min_days=14):
    """Find identical daily readings across consecutive calendar days."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    results = []

    numeric_columns = df.select_dtypes(include="number").columns
    all_dates = pd.date_range(df["date"].min(), df["date"].max())

    for col in numeric_columns:
        daily = df.groupby("date")[col].agg(nunique="nunique", value="first")

        start = None
        previous_date = None
        current_value = None
        count = 0

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
                        results.append(
                            {
                                "column": col,
                                "value": current_value,
                                "start_date": start.date(),
                                "end_date": previous_date.date(),
                                "days": count,
                            }
                        )

                    start = date
                    current_value = value
                    count = 1

                previous_date = date

            else:
                if count >= min_days:
                    results.append(
                        {
                            "column": col,
                            "value": current_value,
                            "start_date": start.date(),
                            "end_date": previous_date.date(),
                            "days": count,
                        }
                    )

                start = None
                previous_date = None
                current_value = None
                count = 0

        if count >= min_days:
            results.append(
                {
                    "column": col,
                    "value": current_value,
                    "start_date": start.date(),
                    "end_date": previous_date.date(),
                    "days": count,
                }
            )

    return results
