import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.style.use('seaborn-whitegrid')



# =============================================================================
# SETUP
dir_figures = "/home/simon/Documents/BCI/experiments/subjects/figures/"
dir_results = "/home/simon/Documents/BCI/experiments/subjects/predict/"
os.makedirs(dir_figures, exist_ok=True)

subjects = [
    "106", "107", "108", "111", "112", "113", "114", "115", "117", "118",
    # "119", "120", "121", "122", "123",
    # "143", "145", "146", "147", "151",
    # "152", "154",
    # "155", "156", "158", "159", "160", "166", "167",
    # "169", "171", "172", "177", "178", "179", "183",
    # "184", "185", "190", "191", "212"
]
# -----------------------------------------------------------------------------




# =============================================================================
# LOAD RESULTS

# load all KXXX.frt files
results = []
for subject in subjects:
    try:
        results.append(pd.read_csv(dir_results + f"K{subject}.frt", index_col=0))
    except:
        pass
frt = pd.concat(results, ignore_index=True)

# load all KXXX.frtswlda files
results = []
for subject in subjects:
    try:
        results.append(pd.read_csv(dir_results + f"K{subject}.frtswlda", index_col=0))
    except:
        pass
frtswld = pd.concat(results, ignore_index=True)

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
    col="subject",
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
for col, name in zip(g.axes[0], subjects):
    col.set_title(f"Subject {name}")
g.set(xlabel="Repetition")
# save
plt.savefig(filename, bbox_inches="tight")
# -----------------------------------------------------------------------------
