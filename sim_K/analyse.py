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
dir_results = "/home/simon/Documents/BCI/experiments/sim_K/results/"

# experiments
seeds = range(3)
Ktrues = [2, 5, 8]
Ks = range(2, 11)

# combinations
combinations = it.product(seeds, Ktrues, Ks)
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD RESULTS
out = []
for i, (seed, Ktrue, K) in enumerate(list(combinations)):
    try:
        # out.append(pd.read_csv(dir_results + f"dim{Ktrue}_seed{seed}_K{K}.ic", index_col=0))
        df = pd.read_csv(dir_results + f"dim{Ktrue}_K{K}_seed{seed}.icy", index_col=0).T
        df["seed"] = seed
        df["Ktrue"] = Ktrue
        out.append(df)
    except FileNotFoundError:
        print(i, seed, Ktrue, K)
out = pd.concat(out)
# -----------------------------------------------------------------------------



# =============================================================================
# PLOT RESULTS

# compute BF
out.sort_values(["seed", "Ktrue", "K"], inplace=True)
log_bf_loo = out.groupby(["seed", "Ktrue"]).apply(lambda x: x["log_bf_loo"].diff())
log_bf = out.groupby(["seed", "Ktrue"]).apply(lambda x: x["log_bf"].diff())
# put back into dataframe
out["log_bf_loo"] = log_bf_loo.values
out["log_bf"] = log_bf.values


metrics = ["lppd", "elpd_loo", "elpd_waic", "log_bf"]
metrics_display = ["LPPD", "PSIS-LOO", "WAIC", "BF"]

# make long
out = out.melt(
    id_vars=["seed", "Ktrue", "K"],
    value_vars=["lppd", "elpd_loo", "elpd_waic", "log_bf"],
    var_name="metric",
    value_name="value",
)

# plot
g = sns.catplot(
    data=out,
    x="K",
    y="value",
    col="Ktrue",
    row="metric",
    hue="seed",
    kind="point",
    sharey=False,
    sharex="all",
    height=3,
    aspect=1.5,
    legend=False,
)
g.set_axis_labels(x_var="Number of components", y_var="")
for row, n in enumerate(metrics_display):
    g.axes[row, 0].set_ylabel(n)
g.set_titles(template="True K={col_name}")
# plt.show()
plt.savefig(dir_results + "ics_y.pdf")

# -----------------------------------------------------------------------------