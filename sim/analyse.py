import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import itertools as it
plt.style.use('seaborn-whitegrid')
torch.set_default_tensor_type(torch.cuda.FloatTensor)



# =============================================================================
# SETUP
dir_results = "/home/simon/Documents/BCI/experiments/sim/results/"
dir_figures = "/home/simon/Documents/BCI/experiments/sim/figures/"
os.makedirs(dir_figures, exist_ok=True)

# experiments
seeds = [1]
Kxs = [5, 8]
Kys = [3, 5]
Ks = range(1, 11)

# combinations
combinations = it.product(seeds, Kxs, Kys, Ks)
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD RESULTS
out = []
out_test = []
for i, (seed, Kx, Ky, K) in enumerate(list(combinations)):
    try:
        df = pd.read_csv(dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.icx", index_col=0).T
        df["seed"] = seed
        df["Kx"] = Kx
        df["Ky"] = Ky
        df["K"] = K
        df["Target"] = "p(x|y)"
        out.append(df)
        df = pd.read_csv(dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.icy", index_col=0).T
        df["seed"] = seed
        df["Kx"] = Kx
        df["Ky"] = Ky
        df["K"] = K
        df["Target"] = "p(y|x)"
        out.append(df)
        df = pd.read_csv(dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.test", index_col=0)
        df["Target"] = "p(y|x)"
        out_test.append(df)
    except FileNotFoundError:
        print(i, seed, Kx, Ky, K)
out = pd.concat(out)
out_test = pd.concat(out_test)
# -----------------------------------------------------------------------------



# =============================================================================
# PLOT RESULTS

# compute BF
out.sort_values(["seed", "Kx", "Ky", "Target", "K"], inplace=True)
log_bf_loo = out.groupby(["seed", "Kx", "Ky", "Target"]).apply(lambda x: x["log_bf_loo"].diff())
log_bf = out.groupby(["seed", "Kx", "Ky", "Target"]).apply(lambda x: x["log_bf"].diff())
# put back into dataframe
out["log_bf_loo"] = log_bf_loo.values.ravel()
out["log_bf"] = log_bf.values.ravel()



# make long
value = out.melt(
    id_vars=["seed", "Kx", "Ky", "K", "Target"],
    value_vars=["lppd", "elpd_loo", "elpd_waic", "log_bf"],
    var_name="metric",
    value_name="value",
)
se = out.melt(
    id_vars=["seed", "Kx", "Ky", "K", "Target"],
    value_vars=["elpd_loo_se", "elpd_waic_se"],
    var_name="metric",
    value_name="se",
)

# add in test metrics
value_test = out_test.melt(
    id_vars=["seed", "Kx", "Ky", "K", "Target"],
    value_vars=["bce", "acc"],
    var_name="metric",
    value_name="value",
)
se_test = out_test.melt(
    id_vars=["seed", "Kx", "Ky", "K", "Target"],
    value_vars=["bce_se"],
    var_name="metric",
    value_name="se",
)
value = pd.concat([value, value_test])
se = pd.concat([se, se_test])

metrics = ["lppd", "elpd_loo", "elpd_waic", "log_bf", "bce", "acc"]
metrics_display = ["LPPD", "PSIS-LOO", "WAIC", "BF", "Test BCE", "Accuracy"]

value["metric"] = value["metric"].replace({k: v for k, v in zip(metrics, metrics_display)})
value["Experiment"] = "Kx=" + value["Kx"].astype(str) + " | Ky=" + value["Ky"].astype(str)
value["Metric"] = value["Target"].astype(str) + ": " + value["metric"].astype(str)

# remove trailing _se from metric
se["metric"] = se["metric"].str.replace("_se", "")
se["metric"] = se["metric"].replace({k: v for k, v in zip(metrics, metrics_display)})
se["Experiment"] = "Kx=" + se["Kx"].astype(str) + " | Ky=" + se["Ky"].astype(str)
se["Metric"] = se["Target"].astype(str) + ": " + se["metric"].astype(str)

metrics = value["Metric"].unique()
nrows = len(metrics)
experiments = value["Experiment"].unique()
ncols = len(experiments)

plt.cla()
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*3+1, nrows*2+1), sharey=False, sharex="all")
for i, metric in enumerate(metrics):
    for j, experiment in enumerate(experiments):
        ax = axs[i, j]
        if i == 0:
            ax.set_title(experiment)
        if j == 0:
            ax.set_ylabel(metric)
        if i == nrows - 1:
            ax.set_xlabel("Number of components")
        values = value[(value["Metric"] == metric) & (value["Experiment"] == experiment)]
        ses = se[(se["Metric"] == metric) & (se["Experiment"] == experiment)]
        for s in values["seed"].unique():
            ax.plot(
                values[values["seed"] == s]["K"],
                values[values["seed"] == s]["value"],
                label=f"seed={s}",
                marker="o",
                color="blue" if values["Target"].unique()[0] == "p(x|y)" else "red",
                linestyle=["-", "--", ":"][s],
            )
            if ses.shape[0] > 0:
                ax.fill_between(
                    values[values["seed"] == s]["K"],
                    values[values["seed"] == s]["value"].values - ses[ses["seed"] == s]["se"].values,
                    values[values["seed"] == s]["value"].values + ses[ses["seed"] == s]["se"].values,
                    alpha=0.1,
                    color="blue" if values["Target"].unique()[0] == "p(x|y)" else "red",
                )
plt.tight_layout()
plt.savefig(dir_figures + "ic.pdf")

# -----------------------------------------------------------------------------