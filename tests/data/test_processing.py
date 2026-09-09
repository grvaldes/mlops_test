import pandas as pd

from material_mlops.data.processing import (
    PROCESSED_COLUMNS,
    process_data,
    split_features_target,
)


def test_process_data_removes_duplicates():
    data = pd.DataFrame(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [2, 3, 4, 5, 6, 7, 8, 9, 10],
        ],
        columns=PROCESSED_COLUMNS,
    )

    processed = process_data(data)

    assert len(processed) == 2


def test_process_data_renames_columns():
    data = pd.DataFrame(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
    )

    data.columns = PROCESSED_COLUMNS

    processed = process_data(data)

    assert processed.columns.tolist() == PROCESSED_COLUMNS


def test_split_features_target():
    data = pd.DataFrame(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
        columns=PROCESSED_COLUMNS,
    )

    X, y = split_features_target(data)

    assert X.shape == (1, 8)
    assert y.shape == (1,)
    assert "compressive_strength" not in X.columns
