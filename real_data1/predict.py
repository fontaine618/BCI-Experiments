import sys
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
import torch
import pickle
from src.bffmbci import BFFMResults
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
dir_results = dir + "predict/"

chains = [0, 1, 2, 3, 4, ]

type = "TRN"
subject = "114"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"
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
factor_processes_method = "maximize"
aggregation_method = "product"
return_cumulative = True
n_samples = 3
# -----------------------------------------------------------------------------


# =============================================================================
results = BFFMResults.from_files(
	[dir_chains + f"seed{chain}_mala.chain" for chain in chains],
	warmup=30_000,
	thin=1
)

latent_dim = 5
nt = 25
# -----------------------------------------------------------------------------


# =============================================================================
# Training
nr = 15
nc = 19

order = eeg.stimulus_order
sequences = eeg.sequence
target = eeg.target

self = results.to_predict(n_samples=n_samples)
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

target_ = target[::nr, :].unsqueeze(1).repeat(1, nr, 1)
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
	"bce": bce.cpu()
}, index=range(1, nr+1))


df.to_csv(dir_results + f"train_{factor_processes_method}_{n_samples}.csv")
# -----------------------------------------------------------------------------






# =============================================================================
# Testing
type = "FRT"
session = "003"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"

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
sequences = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx

self = results.to_predict(n_samples=n_samples)

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

target_ = target[::nr, :].unsqueeze(1).repeat(1, nr, 1)
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
	"bce": bce.cpu()
}, index=range(1, nr+1))


df.to_csv(dir_results + f"test_{factor_processes_method}_{n_samples}.csv")

preds = wide_pred_one_hot[:, -1, :]
i0 = preds.argmax(1)
i1 = preds.index_put_((torch.arange(preds.shape[0]), i0), torch.zeros_like(i0)).argmax(1)
tuples = [(i.item(), j.item()) for i, j in zip(i0, i1)]
# -----------------------------------------------------------------------------

subject = "111"
type = "FRT"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"
mat_obj = scipy.io.loadmat(filename)["Data"][0][0][0][0][0]
parameters = _array_to_dict(mat_obj[2][0][0])
parameters = {k: _array_to_dict(v[0][0]) for k, v in parameters.items()}
print(parameters["TextResult"]["Value"])
print(parameters["TextToSpell"]["Value"])



# =============================================================================
import numpy as np


order = eeg.stimulus_order
sequences = eeg.sequence
target = eeg.target

self = results.to_predict(n_samples=n_samples)


def llk(bffmodel: BFFModel):
	return bffmodel.variables["observations"].log_density

def maxllk(bffmodel: BFFModel):
	return bffmodel.variables["observations"].log_density + \
		bffmodel.variables["factor_processes"].log_density


statistics = dict(llk=llk, maxllk=maxllk)

obs, sam = self.posterior_checks(
	order,
	target,
	sequences,
	llk=llk
)

torch.Tensor(obs["llk"]) - torch.Tensor(sam["llk"])

plt.cla()
plt.scatter(x=np.array(obs["llk"]), y=np.array(sam["llk"]))
plt.axline((min(obs["llk"]), min(obs["llk"])), slope=1)
plt.xlabel("observed")
plt.ylabel("sampled")
plt.show()

bffmodel.variables["loading_processes"].data
bffmodel.variables["factor_processes"].data[0, :, :]
bffmodel.variables["smgp_scaling"].mixing_process.data
bffmodel.variables["sequence_data"].order.data

for _ in range(100):
	bffmodel.variables["factor_processes"].data = \
		bffmodel.variables["factor_processes"].posterior_mean
	print(bffmodel.variables["observations"].log_density)

bffmodel.variables["factor_processes"].log_density
# -----------------------------------------------------------------------------


