from pathlib import Path

import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw concrete dataset from an Excel file."""
    return pd.read_excel(path)
