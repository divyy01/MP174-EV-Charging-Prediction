import pandas as pd


def load_data(path):
    """Load the original dataset."""
    return pd.read_csv(path)


def check_missing_values(df):
    """Count missing values in every column."""
    return df.isnull().sum()


def check_duplicates(df):
    """Count duplicate rows and duplicate record IDs."""
    return {
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_record_ids": df["record_id"].duplicated().sum(),
    }


def check_negative_values(df):
    """Check for negative numerical values."""
    return (df.select_dtypes(include="number") < 0).sum()


def check_zero_energy(df):
    """Return records having zero energy."""
    return df[df["energy_kwh"] == 0]


def check_long_sessions(df):
    """Identify sessions longer than 1000 recorded minutes."""
    return df[df["session_minutes"] > 1000]


def check_energy_limit(df):
    """Identify energy values exceeding the calculated theoretical limit."""
    duration = df["session_minutes"].copy()

    # Convert suspected seconds-based durations into minutes.
    duration = duration.where(duration <= 1000, duration / 60)

    max_energy = df["charger_kw"] * duration / 60

    return df[df["energy_kwh"] > max_energy]


def check_energy_billing_ratio(df):
    """Identify records with approximately 10x energy/billing ratios."""
    ratio = df["energy_kwh"] / df["kwh_billed_in_app"]

    return df[(ratio >= 8) & (ratio <= 12)]
