import sys
import os
import torch
import time
import pickle
import pandas as pd
import numpy as np
import scipy.special
sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.data.k_protocol import KProtocol
from source.bffmbci.bffm import DynamicRegressionCovarianceRegressionMean
from source.swlda.swlda import swlda, swlda_predict
from torch.distributions import Categorical

# =============================================================================
# SETUP
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
dir_chains = "/home/simon/Documents/BCI/experiments/subject/chains/"
dir_results = "/home/simon/Documents/BCI/experiments/subject/results/"
os.makedirs(dir_chains, exist_ok=True)
os.makedirs(dir_results, exist_ok=True)

# file
type = "TRN"
subject = "114" #str(sys.argv[1])
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8


train_reps = 3 #int(sys.argv[2])
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
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
# subset training reps
eeg.repetitions(list(range(1, train_reps+1)))
# -----------------------------------------------------------------------------


# =============================================================================
# TRAIN swLDA
response = eeg.stimulus.cpu().numpy()
trny = eeg.stimulus_data["type"].values
trnX = response
trnstim = eeg.stimulus_data


whichchannels, restored_weights, bias = swlda(
    responses=trnX,
    type=trny,
    sampling_rate=1000,
    response_window=[0, response.shape[1] - 1],
    decimation_frequency=1000,
    max_model_features=150,
    penter=0.1,
    premove=0.15
)

Bmat = torch.zeros((16, 25))
Bmat[restored_weights[:, 0] - 1, restored_weights[:, 1] - 1] = torch.Tensor(restored_weights[:, 3])
Bmat = Bmat.cpu()
# -----------------------------------------------------------------------------


# =============================================================================
# TEST
eeg = KProtocol(
    filename=dir_data + name + ".mat",
    type=type,
    subject=subject,
    session=session,
    window=window,
    bandpass_window=bandpass_window,
    bandpass_order=bandpass_order,
    downsample=downsample,
)
# subset training reps
eeg = eeg.repetitions(list(range(train_reps+1, 16)), True)

nchars = eeg.stimulus_data["character"].nunique()
nreps = eeg.stimulus_data["repetition"].nunique()

response = eeg.stimulus.cpu().numpy()
trny = eeg.stimulus_data["type"].values
trnX = response
trnstim = eeg.stimulus_data

ip = np.einsum("nte, et -> n", trnX, Bmat)


trnstim["log_proba"] = ip


log_prob = np.zeros((nchars, nreps, 36))

for c in trnstim["character"].unique():
    cum_log_proba = np.zeros((6, 6))
    for j, r in enumerate(trnstim["repetition"].unique()):
        idx = (trnstim["character"] == c) & (trnstim["repetition"] == r)
        log_proba = trnstim.loc[idx, "log_proba"].values
        stim = trnstim.loc[idx, "source"].values
        log_proba_mat = np.zeros((6, 6))
        for i, s in enumerate(stim):
            if s < 7:
                log_proba_mat[s-1, :] += log_proba[i]
            else:
                log_proba_mat[:, s-7] += log_proba[i]
        log_proba_mat -= scipy.special.logsumexp(log_proba_mat)
        cum_log_proba += log_proba_mat
        log_prob[c-1, j, :] = cum_log_proba.flatten().copy()

log_prob = torch.Tensor(log_prob)

Js = (6, 6)
combinations = torch.cartesian_prod(*[torch.arange(J) for J in Js])
to_add = torch.cumsum(torch.Tensor([0] + list(Js))[0:-1], 0).reshape(1, -1)
combinations = combinations + to_add
combinations = torch.nn.functional.one_hot(combinations.long(), sum(Js)).sum(1)

wide_pred = log_prob.argmax(2)
eeg.keyboard.flatten()[wide_pred.cpu()]
wide_pred_one_hot = combinations[wide_pred, :]
# -----------------------------------------------------------------------------






# =============================================================================
# METRICS

# entropy
entropy = Categorical(logits=log_prob).entropy()
mean_entropy = entropy.mean(0)

# accuracy & hamming
target_wide = eeg.target.view(nchars, nreps, -1).flip(0)  #NB: flip because testing
accuracy = (wide_pred_one_hot == target_wide).all(2).double().mean(0)
hamming = (wide_pred_one_hot != target_wide).double().sum(2).mean(0) / 2

# binary cross-entropy
ips = torch.einsum("...i,ji->...j", target_wide.double(), combinations.double())
idx = torch.argmax(ips, -1)

target_char = torch.nn.functional.one_hot(idx, 36)
bce = - (target_char * log_prob).sum(2).mean(0)

# save
df = pd.DataFrame({
    "hamming": hamming.cpu(),
    "acc": accuracy.cpu(),
    # "mean_entropy": entropy.mean(0).abs().cpu(),
    # "bce": bce.cpu(),
    "dataset": name + "_test",
    "repetition": range(1, nreps + 1),
    "training_reps": train_reps,
    "method": "swLDA",
}, index=range(1, nreps + 1))
df.to_csv(dir_results + f"K{subject}_{train_reps}reps_swlda.test")
# -----------------------------------------------------------------------------