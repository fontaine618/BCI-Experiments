import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use("seaborn-v0_8-whitegrid")

dir_results = "/home/simon/Documents/BCI/experiments/tuning/predict/"
dir_figures = "/home/simon/Documents/BCI/experiments/tuning/figures/"
K = 8
nreps = 7
seed = 0
cors = [0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
shrinkages = [3., 4., 5., 7.]


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
train = pred[pred["dataset"] == "training"]
test = pred[pred["dataset"] == "testing"]

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

