import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
import torch
import arviz as az
from source.bffmbci import BFFMResults
from source.data.k_protocol import KProtocol
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_figures = "/home/simfont/Documents/BCI/experiments/latent_dimension/figures/"
dir_results = "/home/simfont/Documents/BCI/experiments/latent_dimension/results/"
dir_chains = "/home/simfont/Documents/BCI/experiments/latent_dimension/chains/"
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
os.makedirs(dir_figures, exist_ok=True)
os.makedirs(dir_results, exist_ok=True)

Ks = list(range(2, 14))
K = int(sys.argv[1])

subject = "114"

# file
type = "TRN"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8


# prediction settings
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
return_cumulative = False
n_samples = 1000
factor_samples = 10
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
nchars = eeg.stimulus_data["character"].nunique()
nreps = eeg.stimulus_data["repetition"].nunique()

order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD RESULTS
file = f"K{subject}_dim{K}.chain"
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file],
    warmup=0,
    thin=1
)
self = results.to_predict(n_samples=n_samples)
# -----------------------------------------------------------------------------



# =============================================================================
# GET PREDICTIVE PROBABILITIES
llk_long, chars = self.predict(
    order=order,
    sequence=sequence,
    factor_samples=factor_samples,
    character_idx=character_idx,
    factor_processes_method=factor_processes_method,
    drop_component=None
)
# save
np.save(
    dir_results + f"K{subject}_dim{K}_mllk.npy",
    llk_long.cpu().numpy()
)
# -----------------------------------------------------------------------------


