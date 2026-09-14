import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
  average_precision_score,
  recall_score,
  precision_score,
  classification_report,
  confusion_matrix, ConfusionMatrixDisplay,RocCurveDisplay,
  PrecisionRecallDisplay)

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "interim_clean_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
SPLITTED_DATA_PATH = PROJECT_ROOT / "data" / "split" 

def load_data():
  df = pd.read_csv(PROCESSED_DATA_PATH)
  return df

def analysis_target(df : pd.DataFrame, target:str) -> str:
  class_ratio = 0.0
  if df[target].dtype == 'object' or df[target].nunique() <= 10:
    target_type = 'classification'
  else:
    target_type = 'regression'

  print(f'Machine Learning/Target Problem Type: {target_type}')
  if target_type == 'classification':
    class_counts = df[target].value_counts(normalize=True)
    class_ratio = class_counts.max()
    if round(class_ratio,2) >0.75:
      print(f'Target Class {target} Distribution Ratio: {round(class_ratio,2)}\nClass Imbalance Detected.')
      print("Recommended Model: ")
      print("- XGBoost or LightGBM with class weights or SMOTE")
      print("- RandomForestClassifier with `class_weight='balanced'`")
    else:
      print(f'Target Class {target} Distribution Ratio: {round(class_ratio,2)}\nTarget Class is fairly distributed.')
      print("Recommended Model: ")
      print("- Logistic Regression (simple, linear)")
      print("- Random Forest / Gradient Boosting (non-linear)")
      print("- Support Vector Machine (for small datasets)")

def identify_columns(target : str,
                     df: pd.DataFrame,
                     id_cols : list[str], 
                     fmode_cols : list[str]) -> list[str]:
  # Drop id column
  df = df.drop(id_cols, axis=1)

  cat_cols = df.select_dtypes(include='object').columns.tolist()
  print(f"Categorical Columns: {cat_cols}")

  num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
  # Remove UDI and target variables
  num_cols = [n for n in num_cols if n != target and n not in id_cols ] 
  # Remove five process failure modes which are not the features
  num_cols = [n for n in num_cols if n not in fmode_cols ]
  print(f"Numerical Columns: {num_cols}")

  return cat_cols, num_cols

def split_data(df:pd.DataFrame,
               target:str,
               test_size:float,
               cat_cols:list[str], 
               num_cols:list[str]) -> dict[pd.DataFrame]:
  cat_feature = cat_cols
  num_feature = num_cols
  features_cols = cat_feature + num_feature
  X = df[features_cols]
  y = df[target]

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size,stratify=y, random_state=42)

  return {
    'X_train' : X_train,
    'X_test' : X_test,
    'y_train' : y_train,
    'y_test' : y_test
  }

def save_splitted_test_data(X_train: pd.DataFrame,
                            X_test:pd.DataFrame,
                            y_train:pd.DataFrame,
                            y_test:pd.DataFrame):
    X_train.to_csv(SPLITTED_DATA_PATH / "X_train.csv", index = False)
    X_test.to_csv(SPLITTED_DATA_PATH / "X_test.csv", index=False)
    y_train.to_csv(SPLITTED_DATA_PATH / "y_train.csv", index=False)
    y_test.to_csv(SPLITTED_DATA_PATH / "y_test.csv", index=False)
    print(f"Test dataset is successfully saved.")