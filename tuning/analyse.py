import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# GRID EXPERIMENT: correlation and shrinkage
dir_results = "/home/simon/Documents/BCI/experiments/tuning/predict/"
dir_figures = "/home/simon/Documents/BCI/experiments/tuning/figures/"
K = 8
nreps = 7
seed = 0
cors = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
shrinkages = [3., 4., 5., 7.]

# source the scratch/swlda.py file first
# swlda_df contains the predictions

# load prediction results
pred_list = []
for cor in cors:
    for shrinkage in shrinkages:
        # file name sis wrong
        file = f"seed{seed}_nreps{15-nreps}_cor{cor}_shrinkage{shrinkage}"
        pred = pd.read_csv(dir_results + file + ".csv", index_col=0)
        pred["cor"] = cor
        pred["shrinkage"] = shrinkage
        pred_list.append(pred)
pred = pd.concat(pred_list, axis=0)

swlda_df["cor"] = "SWLDA"
for sh in pred["shrinkage"].unique():
    tmp = swlda_df
    tmp["shrinkage"] = sh
    pred = pd.concat([pred, tmp], axis=0)



# plot testing results
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.relplot(
    data=pred,
    x="repetition",
    y="bce",
    hue="cor",
    style="cor",
    row="shrinkage",
    col="dataset",
    kind="line",
    ax=ax,
    legend="full",
    palette="viridis",
    facet_kws={"legend_out": True},
    height=3
)
plt.savefig(dir_figures + "bce.pdf", bbox_inches="tight")


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.relplot(
    data=pred,
    x="repetition",
    y="acc",
    hue="cor",
    style="cor",
    row="shrinkage",
    col="dataset",
    kind="line",
    ax=ax,
    legend="full",
    palette="viridis",
    facet_kws={"legend_out": True},
    height=3
)
plt.savefig(dir_figures + "acc.pdf", bbox_inches="tight")

# =============================================================================
# HETEROGENEITY EXPERIMENT
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use("seaborn-v0_8-whitegrid")

dir_results = "/home/simon/Documents/BCI/experiments/tuning/predict/"
dir_figures = "/home/simon/Documents/BCI/experiments/tuning/figures/"
K = 8
nreps = 7
seed = 0
heterogeneities = ["_sparse", 1., 2., 3., 5., 7., 10., 15., 20.]

pred_list = []
for h in heterogeneities:
    file = f"heterogeneity{h}"
    pred = pd.read_csv(dir_results + file + ".csv", index_col=0)
    pred["heterogenetiy"] = h
    pred_list.append(pred)
pred = pd.concat(pred_list, axis=0)

# trnsform df to long format with accuracy and bce
pred = pred.melt(
    id_vars=["repetition", "dataset", "heterogenetiy"],
    value_vars=["acc", "bce"],
    var_name="metric",
    value_name="value"
)
pred["heterogenetiy"] = pred["heterogenetiy"].replace("_sparse", "Horseshoe")

# plot acc and bce as rows, training testing as columsn
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.relplot(
    data=pred,
    x="repetition",
    y="value",
    hue="heterogenetiy",
    style="heterogenetiy",
    row="metric",
    col="dataset",
    kind="line",
    ax=ax,
    legend="full",
    palette="viridis",
    facet_kws={"legend_out": True, "sharey": "row"},
    height=3
)
plt.savefig(dir_figures + "heterogeneity.pdf", bbox_inches="tight")


# =============================================================================
# XI_VAR EXPERIMENT
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import seaborn as sns
plt.style.use("seaborn-v0_8-whitegrid")

dir_results = "/home/simon/Documents/BCI/experiments/tuning/predict/"
dir_figures = "/home/simon/Documents/BCI/experiments/tuning/figures/"
K = 8
nreps = 7
seed = 0
xi_vars = [1e-6, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3.]

pred_list = []
for xi in xi_vars:
    file = f"xi_var{xi}"
    pred = pd.read_csv(dir_results + file + ".csv", index_col=0)
    pred["xi_var"] = xi
    pred_list.append(pred)
pred = pd.concat(pred_list, axis=0)

# trnsform df to long format with accuracy and bce
pred = pred.melt(
    id_vars=["repetition", "dataset", "xi_var"],
    value_vars=["acc", "bce"],
    var_name="metric",
    value_name="value"
)

# plot acc and bce as rows, training testing as columsn
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# color on log scale
sns.relplot(
    data=pred,
    x="repetition",
    y="value",
    hue="xi_var",
    style="xi_var",
    row="metric",
    col="dataset",
    kind="line",
    ax=ax,
    legend="full",
    palette="viridis",
    facet_kws={"legend_out": True, "sharey": "row"},
    height=3,
    hue_norm=LogNorm(vmin=0.003, vmax=3)
)
plt.savefig(dir_figures + "xi_var.pdf", bbox_inches="tight")

