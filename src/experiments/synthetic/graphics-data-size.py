# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 10:39:54 2025

@author: chdem
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 23:42:17 2025

@author: chdem
"""
import pandas as pd
from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt


TOTAL_VARIABLES = 50
PERCENT = 100
METRIC = "F1"

# Path to the CSV files
path = r"../clean_datasets/synthetic_data/data_by_size/*.csv"

# Lista de rutas a los archivos CSV
files = glob(path)


dfs = []

for f in files:
    try:
        df = pd.read_csv(f)   # aquí luego probamos opciones
        dfs.append(df)
       
    except pd.errors.ParserError as e:
        print(f"ERROR en archivo: {f}")
        print(e)


pattern = r'^(HD_|SHD_|F1_|COHEN_)'

dfs = [df.filter(regex=pattern) for df in dfs]

bad_suffixes = ("_a", "_c", "_t")

dfs = [
    df.loc[:, ~df.columns.str.endswith(bad_suffixes)]
    for df in dfs
]

dfs = [
    df.loc[:, ~df.columns.str.contains("FCI-STABLE")]
    for df in dfs
]


algorithms = ["CSBS", "PRCDSF", "S-CDFSF", "CSSU", "FCI-FS"]
metrics = ["HD", "SHD", "F1"]
percentages = ["20", "40", "60", "80", "100"]

for df in dfs:
    for alg in algorithms:
        for metric in metrics:
            # Collect all columns for this metric+algorithm+percentages
            cols = []

            for pct in percentages:
                # We look for the substring like "F1_CSBS_20" inside the column name,
                # which covers names like "F1_CSBS_20" AND "F1_CSBS_20_l"
                prefix = f"{metric}_{alg}_{pct}"
                matched = [c for c in df.columns if c.startswith(prefix)]
               
                cols.extend(matched)

            cols = list(dict.fromkeys(cols))  # remove duplicates, keep order

            if len(cols) == 0:
                # No columns found for this metric + algorithm in this df → skip
                continue

            # New column name, e.g. "SC_HD_CSBS", "SC_SHD_CSBS", "SC_F1_CSBS"
            new_col_name = f"SC_{metric}_{alg}"

            # Row-wise mean across all these percentage columns
            df[new_col_name] = df[cols].mean(axis=1)



algorithms = ["CSBS", "PRCDSF", "S-CDFSF", "CSSU", "FCI-FS"]
metrics = ["HD", "SHD"]
percentages = [20, 40, 60, 80, 100]

for df in dfs:
    for alg in algorithms:
        for metric in metrics:

            normalized_values_per_pct = []

            for pct in percentages:
                num_vars = TOTAL_VARIABLES * (pct / 100)

                # use startswith instead of "in"
                prefix = f"{metric}_{alg}_{pct}"
                cols = [c for c in df.columns if c.startswith(prefix)]

                if len(cols) == 0:
                    continue

                # mean for this percentage, then normalize
                value_for_pct = df[cols].mean(axis=1) / num_vars
                normalized_values_per_pct.append(value_for_pct)

            if len(normalized_values_per_pct) == 0:
                continue

            new_col_name = f"SCn_{metric}_{alg}"
            df[new_col_name] = (
                pd.concat(normalized_values_per_pct, axis=1).mean(axis=1)
            )
        # e.g. "SHD_CSBS_100", "SHD_CSBS_100_l", ...
        prefix = f"{METRIC}_{alg}_{PERCENT}"
        metric_cols_percent = [c for c in df.columns if c.startswith(prefix)]

        if not metric_cols_percent:
            continue  # no SHD_..._100 columns for this algorithm in this df

        # New column: one per algorithm
        new_col = f"{METRIC}{PERCENT}_{alg}"
        df[new_col] = df[metric_cols_percent].mean(axis=1)
            
            

        
"""
dfs = [
    df.loc[:, df.columns.str.startswith(("SC_", "SCn_"))]
    for df in dfs
]
"""



# --- you already have this ---
# dfs = [...]  # list of dataframes with only SC_* and SCn_* columns

algorithms = ["CSBS", "PRCDSF", "S-CDFSF", "CSSU", "FCI-FS"]
metrics    = ["SC_HD", "SC_SHD", "SC_F1", "SCn_HD", "SCn_SHD"]

pretty_metric_names = {
    "SC_HD":  "Self-compatibility HD",
    "SC_SHD": "Self-compatibility SHD endmarks",
    "SC_F1":  "Self-compatibility F1",
    "SCn_HD": "Weighted self-compatibility HD",
    "SCn_SHD":"Weighted self-compatibility SHD endmarks"
}

# --- new: unique markers ---
algorithm_markers = {
    "CSBS": "o",      # filled circle
    "PRCDSF": "s",    # square
    "S-CDFSF": "D",   # diamond
    "CSSU": "^",      # triangle up
    "FCI-FS": "v",    # triangle down
}

# optional: control fillstyle (none = hollow)
algorithm_fillstyle = {
    "CSBS": "full",
    "PRCDSF": "none",     # hollow square
    "S-CDFSF": "full",
    "CSSU": "none",       # hollow triangle
    "FCI-FS": "full",
}

# 1) Build summarized_dfs: mean, variance, count per column in each df
summarized_dfs = []
for df in dfs:
    summary = pd.DataFrame({
        "mean": df.mean(),
        "var": df.var(),
        "count": df.count()
    }).T   # rows: mean / var / count, columns: metrics
    summarized_dfs.append(summary)

# 2) Order datasets by size (number of rows), biggest first
order = [3, 5, 1, 2, 4, 0]

N = len(order)
x_values = [375*(2**i) for i in range(N)]  # biggest df gets smallest exponent

# 3. Plot each metric with one line per algorithm

for metric in metrics:
    plt.figure(figsize=(10, 6))

    for alg in algorithms:
        col_name = f"{metric}_{alg}"

        xs   = []  # positions (x-values)
        ys   = []  # means
        yerr = []  # CI 95%

        for pos, df_idx in enumerate(order):
            df = dfs[df_idx]
            #print(df_idx, files[df_idx])

            if col_name not in df.columns:
                continue

            
            series = df[col_name].dropna()
            
            
            n = series.count()
            if n == 0:
                continue

            mean = series.mean()
            var  = series.var(ddof=1)
            se   = np.sqrt(var / n)
            ci95 = 1.96 * se

            xs.append(x_values[pos])
            ys.append(mean)
            yerr.append(ci95)

        if xs:
            plt.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker=algorithm_markers[alg],     # <-- unique marker
                fillstyle=algorithm_fillstyle[alg],# <-- full or hollow
                capsize=4,
                markersize=7,
                label=alg,
            )

    plt.xlabel("Data size")
    plt.ylabel(pretty_metric_names[metric])    
    plt.xscale("log", base=2)

        # Force full precision tick labels
    plt.gca().set_xticks(x_values)
    plt.gca().set_xticklabels([f"{375}", f"{750}", f"{1500}", f"{3000}", f"{6000}", f"{12000}" ])
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
"""
# This code is here to check different values relative to the number of variables
metric = f"{METRIC}{PERCENT}"
pretty_name = f"{METRIC} ({PERCENT}%)"

plt.figure(figsize=(10, 6))

for alg in algorithms:
    col_name = f"{metric}_{alg}"

    xs   = []  # positions (x-values)
    ys   = []  # means
    yerr = []  # CI 95%

    for pos, df_idx in enumerate(order):
        df = dfs[df_idx]
        #print(df_idx, files[df_idx])

        if col_name not in df.columns:
            continue

        
        series = df[col_name].dropna()
        
        
        n = series.count()
        if n == 0:
            continue

        mean = series.mean()
        var  = series.var(ddof=1)
        se   = np.sqrt(var / n)
        ci95 = 1.96 * se

        xs.append(x_values[pos])
        ys.append(mean)
        yerr.append(ci95)

    if xs:
        plt.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker=algorithm_markers[alg],     # <-- unique marker
            fillstyle=algorithm_fillstyle[alg],# <-- full or hollow
            capsize=4,
            markersize=7,
            label=alg,
        )

plt.xlabel("Data size")
plt.ylabel(pretty_name)    
plt.xscale("log", base=2)

    # Force full precision tick labels
plt.gca().set_xticks(x_values)
plt.gca().set_xticklabels([f"{375}", f"{750}", f"{1500}", f"{3000}", f"{6000}", f"{12000}" ])

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

"""