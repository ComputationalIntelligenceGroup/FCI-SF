# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 12:23:49 2025

@author: chdem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from aeon.visualisation import plot_critical_difference

# -------------------
# 1. RAW DATA
# -------------------
data = {
    "Datasets": [
        "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
    "Algorithms": [
        "CSBS", "PRCDSF", "S-CDFSF", "CSSU", "FCI-SF",
    ],
    "Performance (Error)": [
        ["30", "66", "20", "22", "1"],
    ["30", "35", "34", "22", "18"],
    ["46", "73", "10", "22", "15"],
    ["26", "26", "25", "22", "4"],
    ["22", "109", "25", "22", "14"],
    ["60", "43", "30", "22", "4"],
    ["58", "52", "9", "22", "2"],
    ["44", "45", "27", "22", "8"],
    ["36", "53", "28", "22", "18"],
    ],
}

datasets = data["Datasets"]
algorithms = data["Algorithms"]
performance_data = data["Performance (Error)"]

# -------------------
# 2. BUILD DATAFRAME
# -------------------
rows = []
for dataset, performance in zip(datasets, performance_data):
    if len(performance) != len(algorithms):
        raise ValueError(
            f"Dataset '{dataset}' has {len(performance)} scores but "
            f"{len(algorithms)} algorithms – please fix this list."
        )

    row = {"Dataset": dataset}
    row.update({alg: perf for alg, perf in zip(algorithms, performance)})
    rows.append(row)

df = pd.DataFrame(rows)

# Convert string percentages to floats (0–1)
for alg in algorithms:
    df[alg] = (
        df[alg]
        .str.replace(",", ".", regex=False)
        .str.rstrip("%")
        .astype(float)
        / 100.0
    )

# -------------------
# 3. RANKS (LOWER = BETTER)
# -------------------
# Rank per dataset: 1 = lowest error = best algorithm
rankings_matrix = df[algorithms].rank(axis=1, method="min", ascending=True)

# Nicely formatted table with "error (rank)"
formatted_results = df[algorithms].copy()
for col in formatted_results.columns:
    formatted_results[col] = (
        formatted_results[col].round(3).astype(str)
        + " ("
        + rankings_matrix[col].astype(int).astype(str)
        + ")"
    )

sum_ranks = rankings_matrix.sum().round(3).rename("Sum Ranks")
average_ranks = rankings_matrix.mean().round(3).rename("Average Ranks")

formatted_results = pd.concat(
    [formatted_results, sum_ranks.to_frame().T, average_ranks.to_frame().T]
)

formatted_results.insert(
    0,
    "Dataset",
    df["Dataset"].tolist() + ["Sum Ranks", "Average Ranks"],
)

print("Error Table (fraction) with Ranking:")
print(formatted_results)

# Save table as image
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("tight")
ax.axis("off")
table = ax.table(
    cellText=formatted_results.values,
    colLabels=formatted_results.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(2.0, 2.0)
plt.subplots_adjust(
    left=0.2, bottom=0.2, right=0.8, top=1, wspace=0.2, hspace=0.2
)
plt.savefig("table_with_rankings.png", format="png", bbox_inches="tight", dpi=300)
plt.show()
print("Table saved as 'table_with_rankings.png'")

# -------------------
# 4. FRIEDMAN TEST
# -------------------
friedman_stat, p_value = friedmanchisquare(*rankings_matrix.T.values)
print(f"Friedman test statistic: {friedman_stat:.4f}, p-value = {p_value:.4g}")

# -------------------
# 5. CRITICAL DIFFERENCE DIAGRAM (LOWER = BETTER)
# -------------------
scores = df[algorithms].values  # error matrix: n_datasets x n_algorithms
classifiers = algorithms

print("Algorithms:", classifiers)
print("Errors matrix shape:", scores.shape)

plt.figure(figsize=(16, 12))

plot_critical_difference(
    scores,
    classifiers,
    lower_better=True,   # <-- important: lower error = better
    test="wilcoxon",     # or 'nemenyi'
    correction="holm",   # or 'bonferroni' or 'none'
)

ax = plt.gca()

# X-axis labels (algorithms)
for label in ax.get_xticklabels():
    label.set_fontsize(14)
    label.set_rotation(45)
    label.set_horizontalalignment("right")

ax.tick_params(axis="x", which="major", pad=20)
ax.tick_params(axis="y", labelsize=12)
plt.subplots_adjust(bottom=0.35)

plt.savefig(
    "critical_difference_diagram.png",
    format="png",
    bbox_inches="tight",
    dpi=300,
)
plt.show()
print("CD diagram saved as 'critical_difference_diagram.png'")
