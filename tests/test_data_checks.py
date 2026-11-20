from src.mp174.data_checks import (
    load_data,
    check_missing_values,
    check_duplicates,
    check_negative_values,
    check_zero_energy,
    check_long_sessions,
    check_energy_billing_ratio,
)


def test_load_data():
    df = load_data("data/raw/train.csv")
    assert df.shape == (1049, 12)


def test_missing_values():
    df = load_data("data/raw/train.csv")
    assert check_missing_values(df).sum() == 0


def test_duplicates():
    df = load_data("data/raw/train.csv")
    result = check_duplicates(df)
    assert result["duplicate_rows"] == 49
    assert result["duplicate_record_ids"] == 49


def test_negative_values():
    df = load_data("data/raw/train.csv")
    assert check_negative_values(df).sum() == 0


def test_zero_energy():
    df = load_data("data/raw/train.csv")
    assert len(check_zero_energy(df)) == 1


def test_long_sessions():
    df = load_data("data/raw/train.csv")
    assert len(check_long_sessions(df)) == 294


def test_energy_billing_ratio():
    df = load_data("data/raw/train.csv")
    assert len(check_energy_billing_ratio(df)) == 9
