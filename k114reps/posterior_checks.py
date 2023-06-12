import sys
import os
import torch
import time
import pickle
import numpy as np

sys.path.insert(1, '/home/simon/Documents/BCI/src')
sys.path.extend(['/home/simon/Documents/BCI', '/home/simon/Documents/BCI/src'])
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from src.bffmbci import BFFMResults
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical
from src.bffmbci.bffm import BFFModel

plt.style.use("seaborn-v0_8-whitegrid")

from src.data.k_protocol import KProtocol

# =============================================================================
# SETUP
dir = "/experiments/k114reps/"
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
dir_chains = dir + "chains/"
dir_results = dir + "results/"
nreps = 5
model = f"seed0_nreps{nreps}.chain"
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
n_samples = 10
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA AND RESULTS
results = BFFMResults.from_files(
    [dir_chains + model],
    warmup=10_000,
    thin=1
)

self = results.to_predict(n_samples=n_samples)
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
sequences = eeg.sequence
target = eeg.target
# -----------------------------------------------------------------------------


# =============================================================================
# STATISTICS

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
	**statistics
)
# -----------------------------------------------------------------------------


# =============================================================================
# PLOT
for sname in statistics:
    obs_ = np.array(obs[sname])
    sam_ = np.array(sam[sname])
    scatter = np.var(obs_) > 1e-10
    fig, ax = plt.subplots(figsize=(6, 6))
    if scatter:
        ax.scatter(obs_, sam_, alpha=0.5)
        x = np.mean(obs_)
        ax.axline((x, x), slope=1, color="black", linestyle="--")
    else:
        obs_ = np.mean(obs_)
        ax.axvline(obs_, color="black", linestyle="--")
        ax.hist(sam_, bins=20, density=True)
    pval = min(np.mean(obs_ > sam_), np.mean(obs_ < sam_)) * 2.
    ax.set_xlabel("Observed")
    ax.set_ylabel("Sampled")
    ax.set_title(f"{sname} (p={pval:.3f})")
    plt.tight_layout()
    plt.savefig(dir_results + f"posterior_check_{sname}.pdf")
# -----------------------------------------------------------------------------
