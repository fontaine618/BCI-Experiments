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
dir = "/home/simfont/Documents/BCI/experiments/k114cor/"
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_chains = dir + "chains/"
dir_results = dir + "results/"
nreps = 7
cor = [0.9, 0.95, 0.97, 0.99, 0.995, 0.999][int(sys.argv[1])]
file = f"seed0_nreps{nreps}_cor{cor}.chain"

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
factor_processes_method = "maximize"
aggregation_method = "product"
return_cumulative = True
n_samples = 50
nchars = 19
# -----------------------------------------------------------------------------


# =============================================================================
out = []
torch.cuda.empty_cache()
results = BFFMResults.from_files(
	[dir_chains + file],
	warmup=10_000,
	thin=1
)
self = results.to_predict(n_samples=n_samples)
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
eeg = eeg.repetitions(range(1, nreps+1))

order = eeg.stimulus_order
sequences = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx


log_prob, wide_pred_one_hot, _ = self.predict(
	order=order,
	sequence=sequences,
	factor_samples=factor_samples,
	character_idx=character_idx,
	factor_processes_method=factor_processes_method,
	aggregation_method=aggregation_method,
	return_cumulative=return_cumulative
)

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
}, index=range(1, nreps + 1))
out.append(df)
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
eeg = eeg.repetitions(range(nreps+1, 16))
nreps = 15-nreps

order = eeg.stimulus_order.flip(0)
sequences = eeg.sequence.flip(0)
target = eeg.target.flip(0)
character_idx = eeg.character_idx.flip(0)

log_prob, wide_pred_one_hot, _ = self.predict(
	order=order,
	sequence=sequences,
	factor_samples=factor_samples,
	character_idx=character_idx,
	factor_processes_method=factor_processes_method,
	aggregation_method=aggregation_method,
	return_cumulative=return_cumulative
)

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
}, index=range(1, nreps + 1))
out.append(df)

df = pd.concat(out)
df.to_csv(dir_results + f"predict_cor{cor}.csv")
# -----------------------------------------------------------------------------


