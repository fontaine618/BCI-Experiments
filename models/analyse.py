import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.style.use('seaborn-whitegrid')



# =============================================================================
# SETUP
dir_figures = "/home/simon/Documents/BCI/experiments/models/figures/"
dir_results = "/home/simon/Documents/BCI/experiments/models/predict/"
dir_swlda = "/home/simon/Documents/BCI/experiments/subjects/predict/"
os.makedirs(dir_figures, exist_ok=True)

# choose model
model_name = ["drcrm", "dcrm", "scrm", "drcsm"] #, "scrmfr"]

subject = "114"
# -----------------------------------------------------------------------------




# =============================================================================
# LOAD RESULTS

# load all KXXX.frt files
results = []
for mname in model_name:
    results.append(pd.read_csv(dir_results + f"K{subject}_{mname}.frt", index_col=0))
frt = pd.concat(results, ignore_index=True)

# load all KXXX.frtswlda files
frtswld = pd.read_csv(dir_swlda + f"K{subject}.frtswlda", index_col=0)

# concatenate
frt = pd.concat([frt, frtswld], ignore_index=True)
# -----------------------------------------------------------------------------




# =============================================================================
# PLOT RESULTS
filename = dir_figures + "frt.pdf"

# plot using seaborn
# each row is a subject
# x-axis is repetitions
# columsna re different metrics: acc, hamming, mean_entropy
# curves are the method

# first we need to melt the metrics
frt = frt.melt(
    id_vars=["subject", "repetition", "method"],
    value_vars=["acc", "hamming", "bce", "mean_entropy"],
    var_name="metric",
    value_name="value"
)

# plot
plt.cla()
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
g = sns.relplot(
    data=frt,
    x="repetition",
    y="value",
    hue="method",
    style="method",
    col="metric",
    col_wrap=2,
    kind="line",
    height=3,
    aspect=1.5,
    legend="full",
    facet_kws={
        "sharey": False,
        "sharex": True,
    }
)
g.set_titles("")
for ax, name in zip(g.axes.flatten(), ["Accuracy", "Hamming", "BCE", "Entropy"]):
    ax.set_ylabel(name)
g.set(xlabel="Repetition")
# save
plt.savefig(filename, bbox_inches="tight")
# -----------------------------------------------------------------------------
