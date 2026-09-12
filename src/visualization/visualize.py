import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Project root = two levels up from this file (src/visualization/ -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

