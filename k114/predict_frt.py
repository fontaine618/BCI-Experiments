import sys
import os

import numpy as np
import torch
import time
import pickle

sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.bffmbci import BFFMResults
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical
from source.swlda.swlda import swlda_predict

plt.style.use("seaborn-v0_8-whitegrid")

from source.data.k_protocol import KProtocol

# =============================================================================
# SETUP
dir = "/home/simon/Documents/BCI/experiments/k114/"
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
dir_chains = dir + "chains/"
dir_results = dir + "predict/"
dir_figures = dir + "predict/"


# file
type = "FRT"
subject = "114"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8

# model
seed = 0
K = 8
n_iter = 20_000
nreps = 4
cor = 0.6
shrinkage = 5.
heterogeneity = 10.
xi_var = 0.01

factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
return_cumulative = True
n_samples = 100
nchars = 27
factor_samples = 10
file = f"seed{seed}.chain"
# -----------------------------------------------------------------------------



# =============================================================================
# testing accuracy
eeg = KProtocol(
    filename=filename,
    type=type,
    subject=subject,
    session=session,
    window=window,
    bandpass_window=bandpass_window,
    bandpass_order=bandpass_order,
    downsample=downsample,
)

order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx


torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file],
    warmup=10_000,
    thin=1
)
self = results.to_predict(n_samples=n_samples)

llk_long, chars = self.predict(
    order=order,
    sequence=sequence,
    factor_samples=factor_samples,
    character_idx=character_idx,
    factor_processes_method=factor_processes_method,
    drop_component=None
)

log_prob = self.aggregate(
    llk_long,
    sample_mean=sample_mean,
    which_first=which_first
)

wide_pred_one_hot = self.get_predictions(log_prob, True)

entropy = Categorical(logits=log_prob).entropy()

target_ = target[::nreps, :].unsqueeze(1).repeat(1, nreps, 1)
hamming = (wide_pred_one_hot != target_).double().sum(2).sum(0) / 2
acc = (wide_pred_one_hot == target_).all(2).double().sum(0)

target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
bce = (target36 * log_prob).sum(-1).mean(0)

df = pd.DataFrame({
    "hamming": hamming.cpu(),
    "acc": acc.cpu(),
    "max_entropy": entropy.max(0)[0].abs().cpu(),
    "mean_entropy": entropy.mean(0).abs().cpu(),
    "min_max_proba": log_prob.max(2)[0].min(0)[0].cpu().exp(),
    "bce": bce.cpu(),
    "dataset": "FRT",
    "repetition": range(1, nreps + 1),
    "sample_mean": sample_mean,
    "which_first": which_first,
}, index=range(1, nreps + 1))
print(df)

os.makedirs(dir_results, exist_ok=True)
df.to_csv(dir_results + f"FRT.csv")
print(df)
# -----------------------------------------------------------------------------


# =============================================================================
# swLDA predictions
Bmat = np.load(dir_chains + "swlda_bmat.npy")
# prepare data
response = eeg.stimulus.cpu().numpy()
type = eeg.stimulus_data["type"].values
trn = eeg.stimulus_data["repetition"] < 16
trn = eeg.stimulus_data.index[trn]
trnX = response[trn, ...]
trny = type[trn]
trnstim = eeg.stimulus_data.loc[trn]
# get predictions
trn_pred, trn_agg_pred, trn_cum_pred = swlda_predict(Bmat, trnX, trnstim, eeg.keyboard)
# get true
rowtrue = eeg.stimulus_data.loc[(eeg.stimulus_data["source"]<7) & (eeg.stimulus_data["type"]==1)]
rowtrue = rowtrue.groupby(["character"]).head(1).sort_values(["character"]).reset_index()
rowtrue = rowtrue[["character", "source"]].rename(columns={"source": "row"})
coltrue = eeg.stimulus_data.loc[(eeg.stimulus_data["source"]>6) & (eeg.stimulus_data["type"]==1)]
coltrue = coltrue.groupby(["character"]).head(1).sort_values(["character"]).reset_index()
coltrue = coltrue[["character", "source"]].rename(columns={"source": "col"})
true = rowtrue.merge(coltrue, on=["character"], how="outer").reset_index()
true["char"] = eeg.keyboard[true["row"]-1, true["col"]-7]

# metrics
trndf = trn_cum_pred.join(true.set_index("character"), on="character", how="left", rsuffix="_true", lsuffix="_pred")
trndf["hamming"] = (trndf["row_pred"] == trndf["row_true"]).astype(int) \
        + (trndf["col_pred"] == trndf["col_true"]).astype(int)
trndf["acc"] = (trndf["char_pred"] == trndf["char_true"]).astype(int)
trndf = trndf.groupby("repetition").agg({"hamming": "sum", "acc": "sum"})
trndf["dataset"] = "FRT"
trndf["method"] = "swLDA"
trndf.reset_index(inplace=True)
trndf["training_reps"] = 15
# -----------------------------------------------------------------------------


# =============================================================================
# Plot results
df = pd.read_csv(dir_results + f"FRT.csv", index_col=0)
df["method"] = "BFFM"

df = pd.concat([df, trndf])
df["acc"] /= 27

# select the metrics and convert to long format
metrics = ["acc"]
df = df.melt(
    id_vars=["repetition", "method"],
    value_vars=metrics,
    var_name="metric",
    value_name="value"
)

import seaborn as sns
sns.relplot(
    data=df,
    x="repetition",
    y="value",
    row="metric",
    hue="method",
    style="method",
    kind="line",
    legend="full",
    palette="viridis",
    facet_kws={"legend_out": True, "sharey": False},
    height=4,
)
plt.savefig(dir_figures + "FRT.pdf", bbox_inches="tight")
# -----------------------------------------------------------------------------



