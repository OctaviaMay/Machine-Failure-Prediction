import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,label_binarize
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
import joblib

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "interim_clean_data.csv"

SPLITTED_DATA_PATH = PROJECT_ROOT / "data" / "split" / "binary"
SPLITTED_DATA_PATH_MULTI = PROJECT_ROOT / "data" / "split" / "multi"

MODELS_DIR = PROJECT_ROOT / "models" / "binary"
MODELS_DIR_MULTI = PROJECT_ROOT / "models" / "multi"

PREDICTED_DATA_PATH = PROJECT_ROOT / "data" / "predict" / "binary"
PREDICTED_DATA_PATH_MULTI = PROJECT_ROOT / "data" / "predict" / "multi"


SPLITTED_DATA_PATH.mkdir(exist_ok=True)
SPLITTED_DATA_PATH_MULTI.mkdir(exist_ok=True)

MODELS_DIR.mkdir(exist_ok=True)
MODELS_DIR_MULTI.mkdir(exist_ok=True)

PREDICTED_DATA_PATH.mkdir(exist_ok=True)
PREDICTED_DATA_PATH_MULTI.mkdir(exist_ok=True)


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


def analysis_multi_target(df: pd.DataFrame, multi_target:list[str])->pd.DataFrame:
  class_ratio = 0.0
  imbalance_class_count = 0
  arr_class_ratio =[]

  if df[multi_target[0]].dtype == 'object' or df[multi_target[0]].nunique() <= 10:
    target_type = 'classification'
  else:
    target_type = 'regression'

  print(f'Machine Learning/Target Problem Type: {target_type}')

  for target in multi_target:
    if target_type == 'classification':
        class_counts = df[target].value_counts(normalize=True)
        class_ratio = class_counts.max()
        temp_class_ratio = {'variable' : target, 'class_ratio' : class_ratio}
        arr_class_ratio.append(temp_class_ratio)

        if round(class_ratio,2) > 0.75:
          print(f'Target Class {target} Distribution Ratio: {round(class_ratio,2)}. Class Imbalance Detected.')
          imbalance_class_count += 1
        else:
          print(f'Target Class {target} Distribution Ratio: {round(class_ratio,2)}. Target Class is fairly distributed.')

  print("Recommended Model: ")
  if imbalance_class_count > 1:
    print("- XGBoost or LightGBM with class weights or SMOTE")
    print("- RandomForestClassifier with `class_weight='balanced'`")
  else:
    print("- Logistic Regression (simple, linear)")
    print("- Random Forest / Gradient Boosting (non-linear)")
    print("- Support Vector Machine (for small datasets)")

  return pd.DataFrame(arr_class_ratio)

   
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


def encode_multiclass_target(df: pd.DataFrame,
                             target : list[str])->tuple[int,str,pd.DataFrame]:
  # 0=No failure, 1=TWF, 2=HDF, 3=PWF, 4=OSF, 5=RNF
  conditions = [
      (df['TWF'] == 1),
      (df['HDF'] ==1),
      (df['PWF'] ==1),
      (df['OSF'] == 1),
      (df['RNF'] == 1)
  ]

  choices = ['1', '2', '3','4','5']

  df['MultiClass'] = np.select(conditions, choices, default='0')

  #  If a machine has two simultaneous failure modes (e.g., both TWF=1 and HDF=1) which is actually multi-label classification task
  overlap_records =  (df[target].sum(axis=1) > 1).sum()

  # after encoding, target variable should be new column named 'MultiClass'
  # original five process columns that earlier defined as target should be removed.
  df = df.drop(target, axis=1)
  target = 'MultiClass' # Assign new target after encoding

  return overlap_records,target,df



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
                            y_test:pd.DataFrame,
                            modeling_mode = ''):

    if modeling_mode == '':
      X_train.to_csv(SPLITTED_DATA_PATH / "X_train.csv", index = False)
      X_test.to_csv(SPLITTED_DATA_PATH / "X_test.csv", index=False)
      y_train.to_csv(SPLITTED_DATA_PATH / "y_train.csv", index=False)
      y_test.to_csv(SPLITTED_DATA_PATH / "y_test.csv", index=False)
    else:
        X_train.to_csv(SPLITTED_DATA_PATH_MULTI / "X_train.csv", index = False)
        X_test.to_csv(SPLITTED_DATA_PATH_MULTI / "X_test.csv", index=False)
        y_train.to_csv(SPLITTED_DATA_PATH_MULTI / "y_train.csv", index=False)
        y_test.to_csv(SPLITTED_DATA_PATH_MULTI / "y_test.csv", index=False)

    print(f"Test dataset is successfully saved.")


def develop_model(model : type,
                  param_grid: dict,
                  num_feature:list[str],
                  cat_feature : list[str],
                  X_train:pd.DataFrame,
                  y_train:pd.DataFrame) -> type:
    
    preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_feature),
            ('cat', OneHotEncoder(sparse_output=False,handle_unknown='ignore'), cat_feature)
        ]
    )

    pipeline = Pipeline([
        ('preprocess', preprocess),
        ('classifier', model)
    ])

    # print(type(model))

    # Set up GridSearchCV
    pipeline_grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=5, 
        scoring='average_precision',
        )

    # Fit GridSearchCV-model
    pipeline_grid_search.fit(X_train,y_train)

    # Access best model and get best score
    best_model = pipeline_grid_search.best_estimator_
    best_params = pipeline_grid_search.best_params_
    best_score = round(pipeline_grid_search.best_score_,3)
   
    return best_model , best_params, best_score


def get_models(num_feature : list[str],
               cat_feature : list[str],
               X_train:pd.DataFrame, 
               y_train : pd.DataFrame) -> list[dict]:
    
    models = []
    # base model - XGBClassifier
    # Control the balance of positive and negative weights, useful for unbalanced classes. 
    # A typical value to consider: sum(negative instances) / sum(positive instances)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model_xgb = xgb.XGBClassifier(
        random_state = 42,
        scale_pos_weight = ratio,
        objective="binary:logistic")

    param_grid = {
      'classifier__max_depth': [3, 5, 7],
      'classifier__learning_rate': [0.1, 0.01, 0.05],
      'classifier__n_estimators': [50, 100, 200]
    }

    pipeline_xgb, params_xgb, score_xgb = develop_model(model= model_xgb,
                                                        param_grid= param_grid,
                                                        num_feature= num_feature,
                                                        cat_feature=cat_feature,
                                                        X_train=X_train,
                                                        y_train=y_train)
    model_temp = {'model': 'XGBoost', 
                        'pipeline': pipeline_xgb, 
                        'best_params': params_xgb, 
                        'best_score': score_xgb}
    models.append(model_temp)

    # compare model 1 - Light GBM
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model_lgb = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose = -1
    )
    param_grid = {
      'classifier__max_depth': [3, 5, 7],
      'classifier__learning_rate' : [0.1],
      'classifier__n_estimators': [50, 100, 200]
    }
    pipeline_lgb, params_lgb, score_lgb = develop_model(model= model_lgb,
                                                        param_grid= param_grid,
                                                        num_feature= num_feature,
                                                        cat_feature=cat_feature,
                                                        X_train=X_train,
                                                        y_train=y_train)
    model_temp = {'model': 'LightGBM', 
                        'pipeline': pipeline_lgb, 
                        'best_params': params_lgb, 
                        'best_score': score_lgb}
    models.append(model_temp)

    # compare model 2 - RandomForest Classifier
    model_rdf = RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                max_depth=10,
                n_jobs=-1
              )
    param_grid = {
      'classifier__max_depth': [3, 5, 7],
      'classifier__n_estimators': [50, 100, 200]
    }
    pipeline_rdf, params_rdf, score_rdf = develop_model(model= model_rdf,
                                                        param_grid= param_grid,
                                                        num_feature= num_feature,
                                                        cat_feature=cat_feature,
                                                        X_train=X_train,
                                                        y_train=y_train)
    model_temp = {'model': 'RandomForestClassifier', 
                        'pipeline': pipeline_rdf, 
                        'best_params': params_rdf, 
                        'best_score': score_rdf}
    models.append(model_temp)

    return models

def develop_multiclass_model(model : type,
                  param_grid: dict,
                  num_feature:list[str],
                  cat_feature : list[str],
                  X_train:pd.DataFrame,
                  y_train:pd.DataFrame) -> type:
  
  preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_feature),
            ('cat', OneHotEncoder(sparse_output=True,handle_unknown='ignore'), cat_feature)
        ]
    )

  pipeline = Pipeline([
      ('preprocess', preprocess),
      ('classifier', model)
  ])

  # To ensure every fold contains representative samples of all failure classes.
  cv = StratifiedKFold(
      n_splits=5,
      shuffle=True,
      random_state=42
  )

  # Set up GridSearchCV
  pipeline_grid_search  = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',
        return_train_score=True,
        n_jobs = -1 # To uses all CPU cores and to reduce training time.
  )

  # Fit model
  pipeline_grid_search.fit(X_train,y_train)

  # Access best model and get best score
  best_model = pipeline_grid_search.best_estimator_
  best_params = pipeline_grid_search.best_params_
  best_score = round(pipeline_grid_search.best_score_,3)

  return best_model , best_params, best_score


def get_multiclass_model(model_name : str,
              num_feature : list[str],
              cat_feature : list[str],
              X_train:pd.DataFrame, 
              y_train : pd.DataFrame) -> dict:

  # base model - XGBClassifier
  if model_name == 'XGBClassifier':
    # Define parameter grid for GridSearchCV
    param_grid = {
      'classifier__max_depth': [3, 5, 7],
      'classifier__learning_rate': [0.1, 0.01, 0.05],
      'classifier__n_estimators': [50, 100, 200]
    }

    model_xgb = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=6,
        # subsample=0.8,
        # colsample_bytree=0.8,
        # eval_metric='mlogloss',
        random_state=42
    )

    pipeline_xgb, params_xgb, score_xgb = develop_multiclass_model(model= model_xgb,
                                                          param_grid= param_grid,
                                                          num_feature= num_feature,
                                                          cat_feature=cat_feature,
                                                          X_train=X_train,
                                                          y_train=y_train)

    return {'model': 'XGBoost', 
              'pipeline': pipeline_xgb, 
              'best_params': params_xgb, 
              'best_score': score_xgb}
    

  # compare model 1 - Light GBM
  if model_name == 'LGBMClassifier':
    param_grid = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate' : [0.1],
    'classifier__n_estimators': [50, 100, 200]
    }

    model_lgb = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity = -1
    )

    pipeline_lgb, params_lgb, score_lgb = develop_model(model= model_lgb,
                                                          param_grid= param_grid,
                                                          num_feature= num_feature,
                                                          cat_feature=cat_feature,
                                                          X_train=X_train,
                                                          y_train=y_train)
    return {'model': 'LightGBM', 
                      'pipeline': pipeline_lgb, 
                      'best_params': params_lgb, 
                      'best_score': score_lgb}
    

  # compare model 2 - RandomForestClassifier
  if model_name == 'RandomForestClassifier':
    param_grid = {
      'classifier__max_depth': [3, 5, 7],
      'classifier__n_estimators': [50, 100, 200]
      }

    model_rdf = RandomForestClassifier(
      class_weight='balanced',
      random_state=42,
      n_jobs=-1)

    pipeline_rdf, params_rdf, score_rdf = develop_model(model= model_rdf,
                                                            param_grid= param_grid,
                                                            num_feature= num_feature,
                                                            cat_feature=cat_feature,
                                                            X_train=X_train,
                                                            y_train=y_train)


    return {'model': 'RandomForestClassifier', 
                      'pipeline': pipeline_rdf, 
                      'best_params': params_rdf, 
                      'best_score': score_rdf}
    

def get_score(y_test : pd.DataFrame,
                   y_prob : pd.DataFrame) -> float:

    pr_auc = round(average_precision_score(y_test, y_prob),3)

    return pr_auc


def get_multiclass_model_score(model : type,
                  y_test : pd.DataFrame,
                  y_prob : pd.DataFrame) -> float:

    y_test_bin = label_binarize(y_test, classes=model.classes_) # All 6 targets(labels) need to be binarized in multi-class

    pr_auc = round(average_precision_score(
                  y_test_bin,
                  y_prob,
                  average = 'macro'),3)

    return pr_auc
    

def save_selected_models(model_pipelines, modeling_mode = ''):

  for pipeline in model_pipelines:
      m_pck_name = pipeline['model'] + "_pipeline.pkl"
      if modeling_mode == '':
        joblib.dump(pipeline['pipeline'], MODELS_DIR / m_pck_name)
      else:
        joblib.dump(pipeline['pipeline'], MODELS_DIR_MULTI / m_pck_name)

  print('Models successfully saved.')

def save_predicted_prob_value(lst_model_result: list[dict], modeling_mode = ''):

  for result in lst_model_result:
    
    if modeling_mode == '': 
      # for binary classification, predicted value and predicted probability will be saved.
      file_name_pv = result['model_name'] + "_predicted_value.csv"
      df_pv = pd.DataFrame(result['y_predict_value'])
      df_pv.to_csv(PREDICTED_DATA_PATH / file_name_pv, index=False)

      file_name_pb = result['model_name'] + "_predicted_prob.csv"
      df_pb = pd.DataFrame(result['y_predict_prob'])
      df_pb.to_csv(PREDICTED_DATA_PATH / file_name_pb, index=False)
    else:
      file_name = result['model_name'] + "_predicted_prob.csv"
      df = pd.DataFrame(result['y_predict_prob'])
      df.to_csv(PREDICTED_DATA_PATH_MULTI / file_name, index=False)

  print('Predicted value successfully saved.')
    
    
