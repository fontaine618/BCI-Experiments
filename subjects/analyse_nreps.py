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

subject = "114"
nreps = [3, 4, 5, 6, 7, 8, 9, 10]
# -----------------------------------------------------------------------------




# =============================================================================
# LOAD RESULTS

# load all KXXX.frt files
results = []
for trnreps in nreps:
    try:
        results.append(pd.read_csv(dir_results + f"K{subject}_trnreps{trnreps}.pred", index_col=0))
    except:
        pass
bffm = pd.concat(results, ignore_index=True)

# load all KXXX.frtswlda files
results = []
for trnreps in nreps:
    try:
        results.append(pd.read_csv(dir_results + f"K{subject}_trnreps{trnreps}.swlda", index_col=0))
    except:
        pass
swlda = pd.concat(results, ignore_index=True)

# concatenate
pred = pd.concat([bffm, swlda], ignore_index=True)
# -----------------------------------------------------------------------------


# =============================================================================
# PLOT RESULTS
filename = dir_figures + "nreps.pdf"

fig, axes = plt.subplots(6, 2, figsize=(10, 16), sharey="row", sharex="all")
# first row: training accuracy
ax = axes[0][0]
g = sns.lineplot(
    data=pred[(pred["dataset"]=="training") * (pred["method"]=="BFFM") *
              (pred["repetition"]==pred["training_reps"])],
    x="repetition",
    y="acc",
    hue="training_reps",
    style="training_reps",
    markers=True,
    ax=ax,
    legend="full",
    palette="colorblind"
)
ax.set_ylabel("Training accuracy")
ax.set_title("BFFM")

ax.legend(title="Training repetitions", loc="lower right")
ax = axes[0][1]
g = sns.lineplot(
    data=pred[(pred["dataset"]=="training") * (pred["method"]=="swLDA") *
              (pred["repetition"]==pred["training_reps"])],
    x="repetition",
    y="acc",
    hue="training_reps",
    style="training_reps",
    markers=True,
    ax=ax,
    legend="full",
    palette="colorblind"
)
ax.set_title("swLDA")
ax.legend(title="Training repetitions", loc="lower right")

# second row: testing accuracy
ax = axes[1][0]
g = sns.lineplot(
    data=pred[(pred["dataset"]=="testing") * (pred["method"]=="BFFM")],
    x="repetition",
    y="acc",
    hue="training_reps",
    style="training_reps",
    markers=True,
    ax=ax,
    legend="full",
    palette="colorblind"
)
ax.set_ylabel("Testing accuracy")
ax.legend(title="Training repetitions", loc="lower right")

ax = axes[1][1]
g = sns.lineplot(
    data=pred[(pred["dataset"]=="testing") * (pred["method"]=="swLDA")],
    x="repetition",
    y="acc",
    hue="training_reps",
    style="training_reps",
    markers=True,
    ax=ax,
    legend="full",
    palette="colorblind"
)
ax.legend(title="Training repetitions", loc="lower right")

# entropy of BFFM
ax = axes[2][0]
g = sns.lineplot(
    data=pred[(pred["dataset"]=="training") * (pred["method"]=="BFFM") *
              (pred["repetition"]==pred["training_reps"])],
    x="repetition",
    y="bce",
    hue="training_reps",
    style="training_reps",
    markers=True,
    ax=ax,
    legend="full",
    palette="colorblind"
)
ax.set_ylabel("BCE (training)")
ax.legend(title="Training repetitions", loc="lower right")

ax = axes[3][0]
g = sns.lineplot(
    data=pred[(pred["dataset"]=="testing") * (pred["method"]=="BFFM")],
    x="repetition",
    y="bce",
    hue="training_reps",
    style="training_reps",
    markers=True,
    ax=ax,
    legend="full",
    palette="colorblind"
)
ax.set_ylabel("BCE (testing)")
ax.legend(title="Training repetitions", loc="lower right")


# entropy of BFFM
ax = axes[4][0]
g = sns.lineplot(
    data=pred[(pred["dataset"]=="training") * (pred["method"]=="BFFM") *
              (pred["repetition"]==pred["training_reps"])],
    x="repetition",
    y="mean_entropy",
    hue="training_reps",
    style="training_reps",
    markers=True,
    ax=ax,
    legend="full",
    palette="colorblind"
)
g.set(yscale='log')
ax.set_ylabel("Mean entropy (training)")
ax.legend(title="Training repetitions", loc="lower left")

ax = axes[5][0]
g = sns.lineplot(
    data=pred[(pred["dataset"]=="testing") * (pred["method"]=="BFFM")],
    x="repetition",
    y="mean_entropy",
    hue="training_reps",
    style="training_reps",
    markers=True,
    ax=ax,
    legend="full",
    palette="colorblind"
)
g.set(yscale='log')
ax.set_ylabel("Mean entropy (testing)")
ax.legend(title="Training repetitions", loc="lower left")
ax.set_xlabel("Repetition")

axes[5][1].set_xlabel("Repetition")

plt.suptitle(f"Subject {subject}: TRN split into training/testing")
plt.tight_layout()
plt.savefig(filename, bbox_inches="tight")
# -----------------------------------------------------------------------------