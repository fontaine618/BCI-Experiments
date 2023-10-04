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
dir_results = dir + "posterior/"

os.makedirs(dir_results, exist_ok=True)

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
cor = 0.5
shrinkage = 7.
heterogeneity = [1., 2., 3., 5., 7., 10., 15., 20.]
xi_var = 1.
# -----------------------------------------------------------------------------


# =============================================================================
# get psoterior values
for h in heterogeneity:
    # load chain
    file = f"heterogeneity{h}.chain"
    torch.cuda.empty_cache()
    results = BFFMResults.from_files(
        [dir_chains + file],
        warmup=10_000,
        thin=1
    )
    h_chain = results.chains["heterogeneities"]
    l_chain = results.chains["loadings"]
    # get posterior mean
    h_mean = h_chain.mean(dim=(0, 1))
    l_mean = l_chain.mean(dim=(0, 1))
    # save to csv
    pd.DataFrame(h_mean.cpu().numpy()).to_csv(dir_results + f"heterogeneity{h}_hmean.csv")
    pd.DataFrame(l_mean.cpu().numpy()).to_csv(dir_results + f"heterogeneity{h}_lmean.csv")
# -----------------------------------------------------------------------------