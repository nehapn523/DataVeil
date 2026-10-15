
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# ADULT DATASET COLUMN NAMES
# ============================================================

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]


# ============================================================
# LOAD CSV DATASET
# ============================================================

def load_csv(file_path):
    """
    Load a CSV dataset from a given path.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        return df

    except Exception as e:
        raise ValueError(f"Unable to load CSV file: {e}")


# ============================================================
# LOAD UCI ADULT DATASET
# ============================================================

def load_adult_dataset(file_path):
    """
    Load the original UCI Adult dataset.

    The original adult.data file does not contain headers.
    Missing values are represented by '?'.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    try:
        df = pd.read_csv(
            file_path,
            header=None,
            names=COLUMNS,
            na_values=["?", " ?"],
            skipinitialspace=True,
            comment="|"
        )

        # Remove leading and trailing spaces from text values.
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        return df

    except Exception as e:
        raise ValueError(f"Unable to load Adult dataset: {e}")


# ============================================================
# CLEAN DATASET
# ============================================================

def clean_dataset(df, drop_missing=False):
    """
    Clean the dataset.

    Parameters:
        df: Input DataFrame
        drop_missing: If True, remove rows containing missing values.

    Returns:
        Cleaned DataFrame
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    cleaned_df = df.copy()

    # Clean column names.
    cleaned_df.columns = cleaned_df.columns.astype(str).str.strip()

    # Clean string columns.
    for col in cleaned_df.select_dtypes(include=["object"]).columns:
        cleaned_df[col] = cleaned_df[col].apply(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

        cleaned_df[col] = cleaned_df[col].replace(
            ["", "?", " ?"],
            np.nan
        )

    # Optionally remove rows with missing values.
    if drop_missing:
        cleaned_df = cleaned_df.dropna().reset_index(drop=True)

    return cleaned_df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df, strategy="drop"):
    """
    Handle missing values in a dataset.

    Strategies:
        drop      - Remove rows with missing values.
        mean      - Fill numeric columns with their mean.
        median    - Fill numeric columns with their median.
        mode      - Fill columns with their most frequent value.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if strategy not in ["drop", "mean", "median", "mode"]:
        raise ValueError(
            "Invalid strategy. Choose drop, mean, median, or mode."
        )

    result = df.copy()

    if strategy == "drop":
        return result.dropna().reset_index(drop=True)

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    categorical_columns = result.select_dtypes(
        exclude=[np.number]
    ).columns

    if strategy == "mean":
        for col in numeric_columns:
            result[col] = result[col].fillna(result[col].mean())

        for col in categorical_columns:
            if not result[col].mode().empty:
                result[col] = result[col].fillna(
                    result[col].mode().iloc[0]
                )

    elif strategy == "median":
        for col in numeric_columns:
            result[col] = result[col].fillna(result[col].median())

        for col in categorical_columns:
            if not result[col].mode().empty:
                result[col] = result[col].fillna(
                    result[col].mode().iloc[0]
                )

    elif strategy == "mode":
        for col in result.columns:
            modes = result[col].mode(dropna=True)

            if not modes.empty:
                result[col] = result[col].fillna(modes.iloc[0])

    return result


# ============================================================
# GET DATASET INFORMATION
# ============================================================

def get_dataset_info(df):
    """
    Return basic information about the dataset.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage": int(df.memory_usage(deep=True).sum())
    }


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df, file_path):
    """
    Save a DataFrame to a CSV file.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(file_path, index=False)

    return str(file_path)


# ============================================================
# LOAD DATASET AUTOMATICALLY
# ============================================================

def load_dataset(file_path, dataset_type="auto"):
    """
    Load a dataset based on its type.

    dataset_type:
        auto  - Detect Adult dataset from its filename.
        adult - Load original UCI Adult data.
        csv   - Load a regular CSV file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if dataset_type == "auto":
        if file_path.name.lower() in [
            "adult.data",
            "adult.test"
        ]:
            dataset_type = "adult"
        else:
            dataset_type = "csv"

    if dataset_type == "adult":
        return load_adult_dataset(file_path)

    elif dataset_type == "csv":
        return load_csv(file_path)

    else:
        raise ValueError(
            "Invalid dataset_type. Choose auto, adult, or csv."
        )
