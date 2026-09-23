
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
  average_precision_score,
  precision_recall_curve,
  recall_score,
  precision_score,
  classification_report,
  confusion_matrix, ConfusionMatrixDisplay
  )

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BEST_MODEL_DIR = PROJECT_ROOT / "app" / "backend"

def evaluate_at_threshold(predefined_threshold:float,
                         y_test:pd.DataFrame,
                         y_proba:pd.DataFrame) -> dict:
    thresholds = np.linspace(0.0, 1.0, 101)

    best_threshold = None
    best_precision = -1

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rec = recall_score(y_test, y_pred)

        # only consider thresholds that meet recall target
        if rec >= predefined_threshold:
            prec = precision_score(y_test, y_pred, zero_division=0)

            # choose the one with best precision (less false alarms)
            if prec > best_precision:
                best_precision = prec
                best_rec = rec
                best_threshold = t

    # If no threshold meets recall >= threshold
    if best_threshold is None:
        print(f"No threshold reached Recall >= {predefined_threshold} on the test set.")
        print("Try feature engineering, more data, or a different model.")
        return (0,0)
    else:
        print(f"Best threshold for Recall >= {predefined_threshold}: {best_threshold:.3f}")
        print(f"Recall at this threshold: {best_rec:.3f}")
        print(f"Precision at this threshold: {best_precision:.3f}")

        # Final predictions at best threshold
        y_pred_best = (y_proba >= best_threshold).astype(int)

        # Confustion Matrix
        cm = confusion_matrix(y_test, y_pred_best)

        print("Confusion Matrix:")
        print(cm)

        print("Classification Report:")
        print(classification_report(y_test, y_pred_best, digits=3,zero_division=0))

        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))

        return {'best_recall' : best_rec,
                'best_precision' : best_precision,
                 'confusion_matrix' : cm,
                 'confusion_matrix_display' : cm_disp
                 }



def find_multiclass_best_threshold(pipeline:type,
                                   y_proba_multiclass:pd.DataFrame,
                                   y_true_multiclass:pd.DataFrame,
                                   recall_target:float,
                                   no_failure_class=0)->dict:
    # Assuming column 0 of y_proba_multiclass corresponds to 'No Failure'
    # Probability of 'Any Failure' = 1 - P('No Failure')
    NO_FAILURE_CLASS = 0
    no_failure_col_idx = list(pipeline.classes_).index(NO_FAILURE_CLASS)
    y_true_binary = (y_true_multiclass != NO_FAILURE_CLASS).astype(int)
    y_proba_binary = 1 - y_proba_multiclass.iloc[:, [0]]

    precisions, recalls, thresholds_pr  = precision_recall_curve(y_true_binary,y_proba_binary)

    # Find best precision where recall >= target
    mask = recalls[:-1] >= recall_target
    filtered_prec = np.where(mask, precisions[:-1], -1)
    best_idx = np.argmax(filtered_prec)
    best_threshold = thresholds_pr[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    print(f"Best threshold for Recall >= {recall_target}: {best_threshold:.3f}")
    print(f"Best Recall at this threshold: {best_recall:.3f}")
    print(f"Best Precision at this threshold: {best_precision:.3f}")

    return {'best_threshold':best_threshold, 
            'best_precision':best_precision,
            'best_recall':best_recall,
            'y_true_binary':y_true_binary,
            'y_proba_binary':y_proba_binary}

    

def evaluate_at_threshold_multiclass(y_proba_binary:pd.DataFrame,
                          y_true_binary:pd.DataFrame,
                          best_threshold:float):
    
    y_pred = (y_proba_binary >= best_threshold).astype(int)

    cm = confusion_matrix(y_true_binary, y_pred, labels=[0, 1, 2, 3, 4, 5])
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_true_binary, y_pred, digits=3, zero_division=0))

    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no_failure','TWF','HDF','PWF','OSF','RNF'])
    
    return {
                'confusion_matrix' : cm,
                'confusion_matrix_display' : cm_disp
                }



def save_best_model(best_model: type, modeling_mode = ''):

    model_name = best_model.named_steps['classifier'].__class__.__name__

    if modeling_mode == '':
        m_pck_name = model_name + "_binary.pkl"
    else:
        m_pck_name = model_name + "_multi.pkl"

    joblib.dump(best_model, BEST_MODEL_DIR / m_pck_name)

    print('Models successfully saved.')