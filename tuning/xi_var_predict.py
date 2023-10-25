import sys
import os
import torch
import time
import pickle

sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.bffmbci import BFFMResults
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical
from source.bffmbci.bffm import BFFModel

plt.style.use("seaborn-v0_8-whitegrid")

from source.data.k_protocol import KProtocol

# =============================================================================
# SETUP
dir = "/home/simfont/Documents/BCI/experiments/tuning/"
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_chains = dir + "chains/"
dir_results = dir + "predict/"

# file
type = "TRN"
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
cor = 0.8
trnreps = 7
xi_var = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3.][int(sys.argv[1])]

# prediction settings
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
return_cumulative = True
n_samples = 100
# -----------------------------------------------------------------------------

# =============================================================================
# LOAD CHAIN
file = f"xi_var{xi_var}.chain"
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file],
    warmup=10_000,
    thin=1
)
self = results.to_predict(n_samples=n_samples)
out = []
# -----------------------------------------------------------------------------


# =============================================================================
# training accuracy
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
eeg = eeg.repetitions(range(1, trnreps + 1))
nchars = eeg.stimulus_data["character"].nunique()
nreps = eeg.stimulus_data["repetition"].nunique()

order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx

llk_long, chars = self.predict(
    order=order,
    sequence=sequence,
    character_idx=character_idx,
    factor_processes_method=factor_processes_method
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
    "dataset": "training",
    "training_reps": trnreps,
    "repetition": range(1, nreps + 1),
    "xi_var": xi_var,
    "sample_mean": sample_mean,
    "which_first": which_first,
}, index=range(1, nreps + 1))
print(df)

out.append(df)
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
eeg = eeg.repetitions(range(trnreps + 1, 16))
nchars = eeg.stimulus_data["character"].nunique()
nreps = eeg.stimulus_data["repetition"].nunique()

order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx

llk_long, chars = self.predict(
    order=order,
    sequence=sequence,
    character_idx=character_idx,
    factor_processes_method=factor_processes_method
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
    "dataset": "testing",
    "training_reps": trnreps,
    "repetition": range(1, nreps + 1),
    "xi_var": xi_var,
    "sample_mean": sample_mean,
    "which_first": which_first,
}, index=range(1, nreps + 1))
print(df)

out.append(df)
# -----------------------------------------------------------------------------


df = pd.concat(out)
os.makedirs(dir_results, exist_ok=True)
df.to_csv(dir_results + f"xi_var{xi_var}.csv")
print(df)
