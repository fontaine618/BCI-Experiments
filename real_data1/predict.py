import sys
sys.path.insert(1, '/home/simon/Documents/BCI/src')
import torch
import pickle
from src.results import MCMCResults
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical
from src.bffmbci.bffm import BFFModel

plt.style.use("seaborn-v0_8-whitegrid")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from src.data.k_protocol import KProtocol

# =============================================================================
# SETUP
dir = "/home/simon/Documents/BCI/experiments/real_data1/"
dir_chains = dir + "chains/K114_001_BCI_TRN/"
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
dir_figures = dir + "figures/"
dir_results = dir + "predict/"
filename = dir_data + "K114_001_BCI_TRN.mat"

chains = [0,]

type = "TRN"
subject = "114"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8

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

factor_samples = 1
factor_processes_method = "posterior_mean"
aggregation_method = "product"
return_cumulative = True
n_samples = 100
# -----------------------------------------------------------------------------


# =============================================================================
results = MCMCResults.from_files(
	[dir_chains + f"seed{chain}.chain" for chain in chains],
	warmup=10_000,
	thin=1
)

latent_dim = 5
nt = 25
# -----------------------------------------------------------------------------

i = 6
plt.cla()
plt.plot(bffmodel.variables["loading_processes"].data[i, ...].T.cpu())
plt.yscale("log")
plt.legend([f"loading process {i}" for i in range(latent_dim)])
plt.show()
order[i, ...]
target[i, ...]

# =============================================================================
# Training
nr = 15
nc = 19

order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target

self = results.to_predict(n_samples=n_samples)
# character_idx = torch.arange(0, nc).repeat_interleave(nr).int()
character_idx = eeg.character_idx

log_prob, wide_pred_one_hot, _ = self.predict(
	order=order,
	sequence=sequence,
	factor_samples=factor_samples,
	character_idx=character_idx,
	factor_processes_method=factor_processes_method,
	aggregation_method=aggregation_method,
	return_cumulative=return_cumulative
)

entropy = Categorical(logits=log_prob).entropy()

target_ = target[::nr, :].unsqueeze(1).repeat(1, nr, 1)
hamming = (wide_pred_one_hot != target_).double().sum(2).sum(0) / 2
acc = (wide_pred_one_hot == target_).all(2).double().sum(0)

target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
bce = (target36 * log_prob).sum(-1)._mean(0)

df = pd.DataFrame({
	"hamming": hamming.cpu(),
	"acc": acc.cpu(),
	"max_entropy": entropy.max(0)[0].abs().cpu(),
	"mean_entropy": entropy.mean(0).abs().cpu(),
	"min_max_proba": log_prob.max(2)[0].min(0)[0].cpu().exp(),
	"bce": bce.cpu()
}, index=range(1, nr+1))


df.to_csv(dir_results + f"train.csv")
# -----------------------------------------------------------------------------




# =============================================================================
# Testing
nr = 15
nc = 100

order = test_order
sequence = test_sequence
target = test_target

self = results.to_predict(n_samples=n_samples)
character_idx = torch.arange(0, nc).repeat_interleave(nr).int()

log_prob, wide_pred_one_hot, _ = self.predict(
	order=order,
	sequence=sequence,
	factor_samples=factor_samples,
	character_idx=character_idx,
	factor_processes_method=factor_processes_method,
	aggregation_method=aggregation_method,
	return_cumulative=return_cumulative
)

entropy = Categorical(logits=log_prob).entropy()

target_ = target[::nr, :].unsqueeze(1).repeat(1, nr, 1)
hamming = (wide_pred_one_hot != target_).double().sum(2).sum(0) / 2
acc = (wide_pred_one_hot == target_).all(2).double().sum(0)

target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_))
bce = (target36 * log_prob).sum(-1)._mean(0)

df = pd.DataFrame({
	"hamming": hamming.cpu(),
	"acc": acc.cpu(),
	"max_entropy": entropy.max(0)[0].abs().cpu(),
	"mean_entropy": entropy.mean(0).abs().cpu(),
	"min_max_proba": log_prob.max(2)[0].min(0)[0].cpu().exp(),
	"bce": bce.cpu()
}, index=range(1, nr+1))


df.to_csv(dir_results + f"test_nrep{n:02}.csv")
# -----------------------------------------------------------------------------




