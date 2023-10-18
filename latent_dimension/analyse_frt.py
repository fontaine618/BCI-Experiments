import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.style.use('seaborn-whitegrid')



# =============================================================================
# SETUP
dir_figures = "/home/simon/Documents/BCI/experiments/latent_dimension/figures/"
dir_results = "/home/simon/Documents/BCI/experiments/latent_dimension/predict/"
dir_swlda = "/home/simon/Documents/BCI/experiments/subjects/predict/"
os.makedirs(dir_figures, exist_ok=True)

Ks = list(range(2, 13))

subject = "114"
# -----------------------------------------------------------------------------




# =============================================================================
# LOAD RESULTS

# load all KXXX.frt files
results = []
for K in Ks:
    try:
        results.append(pd.read_csv(dir_results + f"K{subject}_dim{K}.frt", index_col=0))
    except:
        pass
frt = pd.concat(results, ignore_index=True)

# load all KXXX.frtswlda files
frtswld = pd.read_csv(dir_swlda + f"K{subject}.frtswlda", index_col=0)
frtswld["K"] = "swLDA"

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
    id_vars=["subject", "repetition", "K", "method"],
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
    hue="K",
    style="method",
    row="metric",
    # col_wrap=3,
    kind="line",
    height=3,
    aspect=1.5,
    legend="full",
    facet_kws={
        "sharey": "row",
        "sharex": True,
    }
)
g.set_titles("")
for row, name in zip(g.axes, ["Accuracy", "Hamming", "BCE", "Entropy"]):
    row[0].set_ylabel(name)
g.set(xlabel="Repetition")
# save
plt.savefig(filename, bbox_inches="tight")
# -----------------------------------------------------------------------------
