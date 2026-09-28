
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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


def get_oof_proba(pipeline, X_train, y_train, n_splits=5, random_state=42):
    # Out-of-fold predicted probabilities on the training set.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_predict(pipeline, X_train, y_train,
                             cv=cv, method='predict_proba')


def find_threshold_for_recall(y_true, y_score, recall_target):
    # Highest-precision threshold that still meets the recall target.
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    mask = recalls[:-1] >= recall_target

    if not mask.any():
        raise ValueError(f"No threshold reaches recall >= {recall_target}")
    best_idx = np.argmax(np.where(mask, precisions[:-1], -1))

    print(f"[OOF] threshold={thresholds[best_idx]:.3f}  "
          f"recall={recalls[best_idx]:.3f}  precision={precisions[best_idx]:.3f}")
    
    return float(thresholds[best_idx])


def evaluate_fixed_threshold(y_true, y_score, threshold):
    # Evaluate once on the test set at a threshold chosen elsewhere.
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    print(f"[TEST] threshold={threshold:.3f}  recall={rec:.3f}  precision={prec:.3f}")
    print(cm)
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    
    return {'threshold': threshold, 'recall': rec, 'precision': prec,
            'confusion_matrix': cm,
            'confusion_matrix_display': ConfusionMatrixDisplay(cm, display_labels=[0, 1])}



def save_best_model(best_model: type, modeling_mode = ''):

    model_name = best_model.named_steps['classifier'].__class__.__name__

    if modeling_mode == '':
        m_pck_name = model_name + "_binary.pkl"
    else:
        m_pck_name = model_name + "_multi.pkl"

    joblib.dump(best_model, BEST_MODEL_DIR / m_pck_name)

    print('Models successfully saved.')