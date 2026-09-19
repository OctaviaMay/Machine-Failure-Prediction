
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
  confusion_matrix, ConfusionMatrixDisplay
  )





def get_threshold_result(threshold:float,
                         y_test:pd.DataFrame,
                         y_proba:pd.DataFrame) -> dict:
    thresholds = np.linspace(0.0, 1.0, 101)

    best_threshold = None
    best_precision = -1

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rec = recall_score(y_test, y_pred)

        # only consider thresholds that meet recall target
        if rec >= threshold:
            prec = precision_score(y_test, y_pred, zero_division=0)

            # choose the one with best precision (less false alarms)
            if prec > best_precision:
                best_precision = prec
                best_rec = rec
                best_threshold = t

    # If no threshold meets recall >= threshold
    if best_threshold is None:
        print(f"No threshold reached Recall >= {threshold} on the test set.")
        print("Try feature engineering, more data, or a different model.")
        return (0,0)
    else:
        print(f"Best threshold for Recall >= {threshold}: {best_threshold:.3f}")
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