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

# load all KXXX.loo files
results = []
for subject in subjects:
    try:
        results.append(pd.read_csv(dir_results + f"K{subject}.loo", index_col=0))
    except:
        pass
loo = pd.concat(results, ignore_index=True)
# -----------------------------------------------------------------------------




# =============================================================================
# PLOT RESULTS

# plot using seaborn
# each panel is a subject
# x-axis is the repetition
# y-axis is the bce
# each line/color is a drop_component

metric = "mean_entropy"
metric_label = "Entropy"
filename = dir_figures + "loo_" + metric + ".pdf"

# plot
g = sns.FacetGrid(
    data=loo,
    col="subject",
    col_wrap=5,
    sharex=True,
    sharey=True,
    height=3,
    aspect=1,
    margin_titles=True,
)
g.map(
    sns.lineplot,
    "repetition",
    metric,
    "drop_component",
    palette="tab10",
    linewidth=1,
    alpha=0.5,
)
g.set_axis_labels("Repetition", metric_label)
g.set_titles(col_template="Subject {col_name}")
if metric == "bce":
    g.set(ylim=(-100., 0))
if metric == "acc":
    g.set(ylim=(0, 1))
g.add_legend()
# plt.show()
plt.savefig(filename, bbox_inches="tight")
# -----------------------------------------------------------------------------
