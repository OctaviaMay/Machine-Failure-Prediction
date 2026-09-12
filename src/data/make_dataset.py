import numpy as np
import pandas as pd

from pathlib import Path

# Project root = two levels up from this file (src/data/ -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
INTERIM_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "interim_clean_data.csv"

def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df

def get_data_overview(df):
    rows = df.shape[0]
    columns = df.shape[1]

    return rows, columns

def check_data(df):
    # Checking missing value
    col_with_missing = list(df.columns[df.isnull().any()])
    # Checking duplicated value
    duplicated_rows = df[df.duplicated(keep=False)]

    if len(col_with_missing) > 0:
        missing_value_msg = f"Columns with missing value: {col_with_missing}"
    else:
        missing_value_msg = f"There is no missing value in all columns."
    if(len(duplicated_rows) > 0):
        duplicated_rows_msg = f"Number of duplicated rows:{len(duplicated_rows)}"
    else:
        duplicated_rows_msg = f"No duplicated rows found in dataset."

    return missing_value_msg, duplicated_rows_msg

def validate_data(df):
    col_num_unique_list = []
    col_cat_unique_list = []

    col_num = df.select_dtypes(include=['float64','int64']).columns.tolist()
    for col in col_num:
        col_num_unique = {'column': col, 'Total_Unique_Value':df[col].nunique(), 'Max': df[col].max(), 'Min':df[col].min()}
        col_num_unique_list.append(col_num_unique)

    col_cat = df.select_dtypes(include='object').columns.tolist()
    for col in col_cat:
        col_cat_unique = {'column':col, 'Total_Unique_Value': df[col].nunique(), 'Unique_Cat_List': df[col].unique()}
        col_cat_unique_list.append(col_cat_unique)

    return col_num_unique_list, col_cat_unique_list

def save_cleand_data(df):
    # Save to interim data folder
    df.to_csv(INTERIM_DATA_PATH, index = False)

    return df