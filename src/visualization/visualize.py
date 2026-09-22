import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay

from pathlib import Path

# Project root = two levels up from this file (src/visualization/ -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFUSION_MATRIX_PATH = PROJECT_ROOT / "assets" / "binary"
CONFUSION_MATRIX_PATH_MULTI = PROJECT_ROOT / "assets" / "multi"

RESULT_EVAL_PATH = PROJECT_ROOT / "assets" / "binary"
RESULT_EVAL_PATH_MULTI = PROJECT_ROOT / "assets" / "multi"

CONFUSION_MATRIX_PATH.mkdir(exist_ok=True)
CONFUSION_MATRIX_PATH_MULTI.mkdir(exist_ok=True)

RESULT_EVAL_PATH.mkdir(exist_ok=True)
RESULT_EVAL_PATH_MULTI.mkdir(exist_ok=True)

# -----------------------------
# Global styling
# -----------------------------
sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

palette = sns.color_palette("Set2") # Custom palette


def plot_mul_histogram(num_cols,df,color_p):
    for col in num_cols:
        plt.subplot(2, 3, num_cols.index(col) + 1)
        sns.histplot(df[col], kde=True,
            color=palette[color_p],
            edgecolor="white",
            linewidth=1,
            alpha=0.85)
        plt.title(f'Distribution of {col} [skewness:{round(df[col].skew(),3)}]')

    plt.tight_layout()
    plt.show()


def plot_bar(df_outlier, x, y, title):
    plt.figure(figsize=(8,6))
    sns.barplot(
        data=df_outlier,
        x=x,
        y=y,
        hue = x,
        width=0.2,
        legend=False
    )

    plt.title(title)
    plt.show()

def plot_bar_with_thredshold(df: pd.DataFrame,
                             cat_col_name:str,
                             target:str,
                             title:str
                             ):
    df_grp_cat = (
        df.groupby(cat_col_name)[target]
        .value_counts()
        .rename('count')
        .reset_index()
    )

    df_grp_cat = df_grp_cat.sort_values([cat_col_name,target,'count'], ascending=[True,True, False])
    plt.figure(figsize=(5,4))#10, 6
    plt.axhline(y=2000, color='red', linestyle='--', linewidth=2)
    plt.xticks(rotation='vertical')
    plt.title(title)
    sns.barplot(data=df_grp_cat,x = cat_col_name,y='count',hue=target)
    plt.show()

def plot_count(df, target, xlabel, title):
    ax = sns.countplot(df, x=target, hue=target, width=0.3 )
    for container in ax.containers:
        ax.bar_label(container)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

def plot_heatmap(df : pd.DataFrame, 
                 num_cols:list[str], 
                 target: str, 
                 title: str):
    num_cols_with_target = num_cols + [target]
    ax=sns.heatmap(df[num_cols_with_target].corr(),annot=True,cmap='coolwarm',fmt='.2f')
    ax.set_title(title,fontsize=12,fontweight='normal',pad=12)
    plt.show()

def plot_box(df: pd.DataFrame,num_cols:list[str], target ):
    plt.figure(figsize=(20,15))
    for col in [c for c in num_cols]:
        plt.subplot(2, 3, num_cols.index(col) + 1)
        sns.boxplot(x=df[target], y=df[col]).set(title=f'{target} Vs {col}',ylabel=f'{col}')

    plt.show()

def plot_multi_box(df: pd.DataFrame,num_cols:list[str], target ):
    for t in target:
        fig, axes = plt.subplots(1, len(num_cols), figsize=(25, 5))

        for i, col in enumerate(num_cols):
            sns.boxplot(
                x=df[t],
                y=df[col],
                ax=axes[i]
            )
            axes[i].set_title(f'{col} Vs Target {t}')

        plt.tight_layout()
        plt.show()

def plot_score_comparison_chart(model_list : list[str], pr_auc_list : list[float], cv_score_list:list[float]):
    bar_width = 0.35
    x = range(len(model_list))

    plt.figure(figsize=(7,5))
    rbar = plt.bar(x, pr_auc_list, width=bar_width, label='PR_AUC', color='skyblue')
    plt.bar_label(rbar, fmt= '%.3f')
    mbar = plt.bar([i + bar_width for i in x], cv_score_list, width=bar_width, label='CV_Score', color='orange')
    plt.bar_label(mbar, fmt = '%.3f')

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('PR AUC and Mean Average Precision Across CV Folds Comparison\n')
    plt.xticks([i + bar_width/2 for i in x], model_list)
    # Add a horizontal line spanning the entire plot width at y = 18
    # plt.axhline(y=8, color='red', linestyle='--', linewidth=1.5, label='Threshold')
    plt.ylim(0, 1.5)
    plt.legend()
    plt.show()

def plot_confusion_matrix(confusion_matrix:type,
                          model_name:str,
                          modeling_mode = ''):

    confusion_matrix.plot(cmap=plt.cm.Blues)
    if modeling_mode == '':
        file_name = f"{CONFUSION_MATRIX_PATH}\\ConfusionMatrix_{model_name}"
    else:
        file_name = f"{CONFUSION_MATRIX_PATH_MULTI}\\ConfusionMatrix_{model_name}"

    plt.savefig(file_name,dpi=1500,bbox_inches='tight')
    plt.show()


def plot_eval_comparison_chart(model_list:list[str],
                               precision_list:list[float],
                               recall_list:list[float],
                               modeling_mode = ''):
    bar_width = 0.35
    x = range(len(model_list))

    plt.figure(figsize=(7,5))
    pbar = plt.bar(x, precision_list, width=bar_width, label='Precision', color='skyblue')
    plt.bar_label(pbar, fmt= '%.3f')
    rbar = plt.bar([i + bar_width for i in x], recall_list, width=bar_width, label='Recall', color='lightgreen')
    plt.bar_label(rbar, fmt = '%.3f')

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Precision and Recall Comparison\n')
    plt.xticks([i + bar_width/2 for i in x], model_list)
    plt.ylim(0, 1)
    plt.legend()

    # Save result plot
    if modeling_mode == '':
        file_name = f"{RESULT_EVAL_PATH}\\results_comparison"
    else:
        file_name = f"{RESULT_EVAL_PATH_MULTI}\\results_comparison"

    plt.savefig(file_name,dpi=150,bbox_inches='tight')

    plt.show()

def plot_pre_recall_curve(pipeline_xgb : type,
                          pipeline_lgb : type,
                          X_test : pd.DataFrame,
                          y_test : pd.DataFrame,
                          modeling_mode = ''):
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot for XGBoost
    PrecisionRecallDisplay.from_estimator(
        pipeline_xgb,
        X_test,
        y_test,
        curve_kwargs = {'color':"darkorange",'linestyle':'--', 'name' : pipeline_xgb.named_steps['classifier'].__class__.__name__},
        ax=ax
    )

    # Plot for LightGBM
    PrecisionRecallDisplay.from_estimator(
        pipeline_lgb,
        X_test,
        y_test,
        curve_kwargs = {'color':"skyblue",'linestyle':'--', 'name':pipeline_lgb.named_steps['classifier'].__class__.__name__},
        ax=ax
    )

    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend()

    if modeling_mode == "":
        file_name = f"{RESULT_EVAL_PATH}//precision_recall_curge"
    else:
        file_name = f"{RESULT_EVAL_PATH_MULTI}//precision_recall_curge"

    plt.savefig(file_name,bbox_inches='tight',dpi=1500)
    plt.show()


