import pandas as pd

PROCESSED_COLUMNS = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
    "compressive_strength",
]


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the raw dataset for model development."""
    processed = data.copy()

    processed = processed.drop_duplicates()
    processed.columns = PROCESSED_COLUMNS

    return processed


def split_features_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the processed dataset into features and target."""
    X = data.drop(columns="compressive_strength")
    y = data["compressive_strength"]

    return X, y
