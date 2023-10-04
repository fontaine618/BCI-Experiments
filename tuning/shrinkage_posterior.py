import sys
import os
import torch
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.bffmbci import BFFMResults
import pandas as pd

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
cor = 0.6
shrinkage = [3., 4., 5., 7., 10.]
file = f"seed{seed}_nreps{nreps}_cor{cor}_shrinkage{shrinkage}.chain"
# -----------------------------------------------------------------------------


# =============================================================================
# get psoterior values
for s in shrinkage:
    # load chain
    file = f"seed{seed}_nreps{nreps}_cor{cor}_shrinkage{s}.chain"
    torch.cuda.empty_cache()
    results = BFFMResults.from_files(
        [dir_chains + file],
        warmup=10_000,
        thin=1
    )
    s_chain = results.chains["shrinkage_factor"]
    l_chain = results.chains["loadings"]
    # get posterior mean
    s_mean = s_chain.mean(dim=(0, 1))
    l_mean = l_chain.mean(dim=(0, 1))
    # save to csv
    pd.DataFrame(s_mean.cpu().numpy()).to_csv(dir_results + f"shrinkage{s}_smean.csv")
    pd.DataFrame(l_mean.cpu().numpy()).to_csv(dir_results + f"shrinkage{s}_lmean.csv")
# -----------------------------------------------------------------------------