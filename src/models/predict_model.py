import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
  average_precision_score,
  recall_score,
  precision_score,
  classification_report,
  confusion_matrix, ConfusionMatrixDisplay,RocCurveDisplay,
  PrecisionRecallDisplay)

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
TEST_DATA_PATH =PROJECT_ROOT / "data" / "split"
FEAT_IMP_PATH = PROJECT_ROOT / "assets" 

MODELS_DIR.mkdir(exist_ok=True)
TEST_DATA_PATH.mkdir(exist_ok=True)
FEAT_IMP_PATH.mkdir(exist_ok=True)

def run_prediction_evaluation(model, X_test, y_test):
  results =[]

  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)[:,1]
  pr_auc = round(average_precision_score(y_test, y_prob),3)

  result = { 'model': model,
              'y_pred': y_pred, 
              'PR_AUC': pr_auc, 
              }

  return result

# def get_feature_importance(pipeline : type ):
#     classifier = pipeline.named_steps['classifier']
#     feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
#     # Extract the actual feature name from the split list, making it a string
#     feature_names = [f.split('__')[-1] for f in feature_names]

#     if pipeline['model'] == 'LinearRegression':
#         impts = abs(classifier.coef_)
#     else:
#         impts = abs(classifier.feature_importances_)

#     fea_imp = pd.DataFrame({
#         "feature": feature_names,
#         "Importance": impts
#     }).sort_values("Importance", ascending=False)

#     print(fea_imp.sort_values("Importance", ascending=False).head(10))
#     plt.figure(figsize=(10,6))
#     sns.barplot(data = fea_imp,x=fea_imp['feature'],y=fea_imp['Importance'], alpha=0.5,color='gray')
#     plt.xticks(rotation='vertical')
#     plt.title(f"Features Importance in {classifier.__class__.__name__}")
#     plt.show()
#     return fea_imp

def get_feature_importance(pipeline):
  classifier = pipeline.named_steps['classifier']
  feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
  # Extract the actual feature name from the split list, making it a string
  feature_names = [f.split('__')[-1] for f in feature_names]

  impts = classifier.feature_importances_

  fea_imp = pd.DataFrame({
      "feature": feature_names,
      "Importance": impts
  }).sort_values("Importance", ascending=False)

  print(fea_imp.sort_values("Importance", ascending=False).head(10))
  sns.set(style='whitegrid')
  plt.figure(figsize=(10,6))
  sns.barplot(data = fea_imp,x=fea_imp['feature'],y=fea_imp['Importance'], alpha=0.5,color='gray')
  plt.xticks(rotation='vertical')
  plt.title(f"Features Importance in {classifier.__class__.__name__}")

  # save feature importance chart
  FILE_PATH = f"{FEAT_IMP_PATH}\\feature_importance_{classifier.__class__.__name__}.png"
  plt.savefig(FILE_PATH, dpi=150, bbox_inches ='tight')
  
  plt.show()
  return fea_imp