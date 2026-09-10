import pandas as pd
import pytest

from material_mlops.data.processing import PROCESSED_COLUMNS
from material_mlops.data.validation import (
    RAW_COLUMNS,
    validate_columns,
    validate_no_missing_values,
    validate_not_empty,
    validate_numeric,
)


def make_valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
        columns=RAW_COLUMNS,
    )


def test_validate_columns_accepts_valid_data():
    data = make_valid_data()

    validate_columns(data)


def test_validate_columns_rejects_invalid_columns():
    data = make_valid_data()
    data = data.rename(columns={RAW_COLUMNS[0]: "wrong_name"})

    with pytest.raises(ValueError):
        validate_columns(data)


def test_validate_no_missing_values_accepts_valid_data():
    data = make_valid_data()

    validate_no_missing_values(data)


def test_validate_no_missing_values_rejects_missing_data():
    data = make_valid_data()
    data.loc[0, "cement"] = None

    with pytest.raises(ValueError):
        validate_no_missing_values(data)


def test_validate_numeric_accepts_valid_data():
    data = make_valid_data()

    validate_numeric(data)


def test_validate_numeric_rejects_non_numeric_data():
    data = make_valid_data()
    data["cement"] = "not a number"

    with pytest.raises(ValueError):
        validate_numeric(data)


def test_validate_not_empty_rejects_empty_data():
    data = pd.DataFrame(columns=PROCESSED_COLUMNS)

    with pytest.raises(ValueError):
        validate_not_empty(data)
