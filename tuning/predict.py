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

type = "TRN"
subject = "114"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8
factor_samples = 10
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
return_cumulative = True
n_samples = 100
nchars = 19

K = 8
nreps = 7
seed = 0
# cor = [0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99][int(sys.argv[1]) % 7]
# shrinkage = [3., 4., 5., 7., 10.][int(sys.argv[1]) // 7]
cor = [0.2, 0.3, 0.4, 0.5, 0.6][int(sys.argv[1]) % 5]
shrinkage = [3., 4., 5., 7., 10.][int(sys.argv[1]) // 5]
file = f"seed{seed}_nreps{nreps}_cor{cor}_shrinkage{shrinkage}.chain"

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
eeg = eeg.repetitions(range(1, nreps + 1))

order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx

llk_long, chars = self.predict(
    order=order,
    sequence=sequence,
    factor_samples=factor_samples,
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
    "training_reps": nreps,
    "repetition": range(1, nreps + 1),
    "cor": cor,
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
eeg = eeg.repetitions(range(nreps + 1, 16))
nreps = 15 - nreps

order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx

llk_long, chars = self.predict(
    order=order,
    sequence=sequence,
    factor_samples=factor_samples,
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
    "training_reps": nreps,
    "repetition": range(1, nreps + 1),
    "cor": cor,
    "sample_mean": sample_mean,
    "which_first": which_first,
}, index=range(1, nreps + 1))
print(df)

out.append(df)
# -----------------------------------------------------------------------------


df = pd.concat(out)
os.makedirs(dir_results, exist_ok=True)
df.to_csv(dir_results + f"seed{seed}_nreps{nreps}_cor{cor}_shrinkage{shrinkage}.csv")
print(df)
