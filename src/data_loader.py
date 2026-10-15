import pandas as pd
from pathlib import Path
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", 
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income" ]
def load_adult_data(path="data/adult.data"):
    """Load and clean the UCI Adult Income dataset."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}")
    df = pd.read_csv(
        file_path, header=None, names=COLUMNS, na_values=["?", " ?"],
        skipinitialspace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df = df.dropna().reset_index(drop=True)
    return df