import pandas as pd


RAW_COLUMNS = [
    "Cement (component 1)(kg in a m^3 mixture)",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
    "Fly Ash (component 3)(kg in a m^3 mixture)",
    "Water  (component 4)(kg in a m^3 mixture)",
    "Superplasticizer (component 5)(kg in a m^3 mixture)",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)",
    "Age (day)",
    "Concrete compressive strength(MPa, megapascals) ",
]


def validate_columns(data: pd.DataFrame) -> None:
    """Validate that the dataset contains the expected columns."""
    actual_columns = data.columns.tolist()

    if actual_columns != RAW_COLUMNS:
        raise ValueError(
            f"Unexpected columns.\n"
            f"Expected: {RAW_COLUMNS}\n"
            f"Actual: {actual_columns}"
        )


def validate_no_missing_values(data: pd.DataFrame) -> None:
    """Validate that the dataset contains no missing values."""
    if data.isna().any().any():
        missing = data.isna().sum()
        raise ValueError(f"Missing values found:\n{missing}")


def validate_numeric(data: pd.DataFrame) -> None:
    """Validate that all columns contain numeric data."""
    non_numeric = data.select_dtypes(exclude="number").columns

    if len(non_numeric) > 0:
        raise ValueError(f"Non-numeric columns found: {list(non_numeric)}")


def check_duplicates(data: pd.DataFrame) -> int:
    """Return the number of duplicate rows."""
    return int(data.duplicated().sum())


def validate_not_empty(data: pd.DataFrame) -> None:
    """Validate that the dataset contains at least one row."""
    if data.empty:
        raise ValueError("Dataset is empty.")
    

def validate_dataset(data: pd.DataFrame) -> None:
    """Run all dataset validation checks."""
    validate_not_empty(data)
    validate_columns(data)
    validate_no_missing_values(data)
    validate_numeric(data)

    duplicates = check_duplicates(data)

    if duplicates > 0:
        print(f"Warning: found {duplicates} duplicate rows.")
