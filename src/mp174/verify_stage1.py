import pandas as pd
from mp174.data_checks import (
    load_data,
    check_duplicates,
    check_unit_mixup,
    check_sentinel_candidates,
    check_spelling_variants,
    check_target_distribution,
    check_leaking_column,
    check_stuck_sensor,
)

train = load_data("data/raw/train.csv")
later = load_data("data/raw/later.csv")
holdout = load_data("data/raw/holdout_features.csv")

print("\n===== DUPLICATE ROWS =====")
print(check_duplicates(train))

print("\n===== UNIT MIX-UP =====")
print(check_unit_mixup(train))

print("\n===== SENTINEL VALUES =====")
print(check_sentinel_candidates(train))

print("\n===== SPELLING VARIANTS =====")
print(check_spelling_variants(train))

print("\n===== TARGET DISTRIBUTION (TOP 15) =====")
print(check_target_distribution(train).head(15).to_string(index=False))

print("\n===== POSSIBLE LEAKING COLUMNS =====")
print(check_leaking_column(train, holdout))

print("\n===== STUCK SENSOR =====")
print(check_stuck_sensor(train))
