# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 21:57:59 2025

@author: chdem
"""


import pandas as pd
from glob import glob
import numpy as np
import re
import matplotlib.pyplot as plt

TOTAL_VARIABLES = 50

# Path to the CSV files
path = r"../clean_datasets/synthetic_data/50_vart_all_alg/*.csv"

# List of CSV filenames
files = glob(path)

# List to collect results
summaries = []

for file in files:
    df = pd.read_csv(file)
    
    df.loc[:, ~df.columns.str.contains("FCI-STABLE")]

    # Compute mean and variance for numeric columns
    means = df.mean(numeric_only=True)
    variances = df.var(numeric_only=True)
    counts = df.count(numeric_only=True)  

    # Create a DataFrame with column name, mean, and variance
    stats_df = pd.DataFrame({
        'column': means.index,
        'mean': means.values,
        'variance': variances.values,
        "count": counts.values,
        'file': file.split('/')[-1]  # filename only
    })

    summaries.append(stats_df)
    

    

prefixes = ["numCI_", "numEdges_", "avgSepSize_", "execTime_", "HD_", "SHD_", "F1_", "COHEN_"]

# Convert to a single regex OR pattern (e.g. "^numCI_|^numEdges_|...")
pattern = "^(" + "|".join(prefixes) + ")"

filtered_summaries = []

for df in summaries:
    filtered_df = df[df["column"].str.match(pattern)]
    filtered_summaries.append(filtered_df.reset_index(drop=True))

# Replace the original list if you want:
summaries = filtered_summaries



ALGORITHMS = [ "CSBS", "PRCDSF", "S-CDFSF", "CSSU", "FCI-FS", "FCI-STABLE" ]
percent_values = [20, 40, 60, 80, 100]

# This will hold one element per original summary DataFrame
# Each element is: { alg_name: { percent: df } }
summaries_by_alg_and_pct = []

for df in summaries:
    alg_dict = {}

    for alg in ALGORITHMS:
        # First filter rows that belong to this algorithm
        # Look for "_ALG_" in the column name (e.g. "_CSBS_", "_FCI-FS_")
        alg_mask = df["column"].str.contains(f"_{alg}_", regex=False)
        df_alg = df[alg_mask].reset_index(drop=True)

        # Now split this algorithm-specific df by percentage
        pct_dict = {}
        for pct in percent_values:
            # Match "_20" or "_20_" etc, avoid "_200"
            pattern = f"_{pct}($|_)"
            pct_mask = df_alg["column"].str.contains(pattern, regex=True)
            df_pct = df_alg[pct_mask].reset_index(drop=True)

            pct_dict[pct] = df_pct

        alg_dict[alg] = pct_dict

    summaries_by_alg_and_pct.append(alg_dict)

# From the first summary (first CSV):
second_summary = summaries_by_alg_and_pct[1]




ALGORITHMS = ["CSBS", "PRCDSF", "S-CDFSF", "CSSU", "FCI-FS"]
percent_values = [20, 40, 60, 80, 100]
metric_prefixes = ["numCI", "numEdges", "avgSepSize", "execTime", "HD", "SHD", "F1", "COHEN"]

metric_labels = {
    "numCI": "Number of CI tests",
    "numEdges": "Number of edges",
    "avgSepSize": "Average separating set size",
    "execTime": "Execution time",
    "HD": "HD",
    "SHD": "SHD endmarks",
    "F1": "F1",
    "COHEN": "Cohen's Kappa"
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

TOTAL_VARIABLES = 50
second_summary = summaries_by_alg_and_pct[1]

for metric in metric_prefixes:
    plt.figure()
    metric_prefix = metric + "_"

    for alg in ALGORITHMS:
        x_vals = []
        y_vals = []
        y_errs = []

        for pct in percent_values:
            df_pct = second_summary[alg][pct]
            row = df_pct[df_pct["column"].str.startswith(metric_prefix)]

            if not row.empty:
                mean = row["mean"].iloc[0]
                var = row["variance"].iloc[0]
                n = row["count"].iloc[0]

                se = np.sqrt(var / n)
                ci_half = 1.96 * se

                num_vars = TOTAL_VARIABLES * pct / 100

                x_vals.append(num_vars)
                y_vals.append(mean)
                y_errs.append(ci_half)
                
        alg_label = alg if alg != "FCI-FS" else "FCI-SF"
            

        if x_vals:
            plt.errorbar(
                x_vals,
                y_vals,
                yerr=y_errs,
                marker=algorithm_markers[alg],     # <-- unique marker
                fillstyle=algorithm_fillstyle[alg],# <-- full or hollow
                capsize=4,
                markersize=7,
                label=alg_label,
            )

    plt.xlabel("Number of Observed Variables")
    plt.ylabel(metric_labels[metric])
    plt.legend()
    plt.grid(True)

plt.show()
