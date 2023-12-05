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
dir_results = "/home/simon/Documents/BCI/experiments/sim_K3/results/"
dir_figures = "/home/simon/Documents/BCI/experiments/sim_K3/figures/"
os.makedirs(dir_figures, exist_ok=True)

# experiments
seeds = range(3)
Kxs = [5, 8]
Kys = [3, 5]
Ks = range(1, 11)

# combinations
combinations = it.product(seeds, Kxs, Kys, Ks)
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD RESULTS
out = []
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
    except FileNotFoundError:
        print(i, seed, Kx, Ky, K)
out = pd.concat(out)
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
out = out.melt(
    id_vars=["seed", "Kx", "Ky", "K", "Target"],
    value_vars=["lppd", "elpd_loo", "elpd_waic", "log_bf"],
    var_name="metric",
    value_name="value",
)

metrics = ["lppd", "elpd_loo", "elpd_waic", "log_bf"]
metrics_display = ["LPPD", "PSIS-LOO", "WAIC", "BF"]

out["metric"] = out["metric"].replace({k: v for k, v in zip(metrics, metrics_display)})
out["Experiment"] = "Kx=" + out["Kx"].astype(str) + " | Ky=" + out["Ky"].astype(str)
out["Metric"] = out["Target"].astype(str) + ": " + out["metric"].astype(str)

metrics = out["Metric"].unique()
nrows = len(metrics)
experiments = out["Experiment"].unique()
ncols = len(experiments)

plt.cla()
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*3+1, nrows*2+1), sharey=False, sharex="all")
for i, metric in enumerate(metrics):
    for j, experiment in enumerate(experiments):
        ax = axs[i] # axs[i, j]
        if i == 0:
            ax.set_title(experiment)
        if j == 0:
            ax.set_ylabel(metric)
        if i == nrows - 1:
            ax.set_xlabel("Number of components")
        data = out[(out["Metric"] == metric) & (out["Experiment"] == experiment)]
        for s in data["seed"].unique():
            ax.plot(
                data[data["seed"] == s]["K"],
                data[data["seed"] == s]["value"],
                label=f"seed={s}",
                marker="o",
                color="blue" if data["Target"].unique()[0] == "p(x|y)" else "red",
            )
plt.tight_layout()
plt.savefig(dir_figures + "ic.pdf")

# -----------------------------------------------------------------------------