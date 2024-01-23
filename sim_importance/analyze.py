import sys
import os
import torch
import time
import pandas as pd
import pickle
import itertools as it
sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.data.k_protocol import KProtocol
from source.bffmbci.bffm import BFFModel
from source.bffmbci import BFFMResults, importance_statistic

# =============================================================================
# SETUP
dir_data = "/home/simon/Documents/BCI/experiments/sim_importance/data/"
dir_chains = "/home/simon/Documents/BCI/experiments/sim_importance/chains/"
dir_results = "/home/simon/Documents/BCI/experiments/sim_importance/results/"
os.makedirs(dir_chains, exist_ok=True)

# experiments
seeds = range(3)
Kxs = [8]
Kys = [5]
Ks = [8]

# combinations
combinations = it.product(seeds, Kxs, Kys, Ks)

# current experiment from command line
i = 0 #int(sys.argv[1])
seed, Kx, Ky, K = list(combinations)[i]

# file
file_true = f"Kx{Kx}_Ky{Ky}_seed{seed}"
file = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}"
file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.chain"
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
variables = pickle.load(open(dir_data + file_true + ".variables", "rb"))
Ltrue = variables["loadings"]
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD CHAIN
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file_chain],
    warmup=0,
    thin=1
)
L = results.chains["loadings"]
Lmean = L.mean((0, 1))
Lstd = L.std((0, 1))
Lmean/Lstd

# procrustes
Lref = L[0, 0, :, :]
U, S, V = torch.linalg.svd(L.mT @ Ltrue)
U = U @ V
Lrot = L @ U


Lmean = Lrot.mean((0, 1))
Lstd = Lrot.std((0, 1))
Lmean/Lstd

# varimax
Lref = _varimax(Lref)[0]
# reorder columns by column norm
Lref = Lref[:, torch.argsort(torch.norm(Lref, dim=0), descending=True)]

# -----------------------------------------------------------------------------


# =============================================================================
# LOAD POSTERIOR
posterior = pickle.load(open(dir_results + file + ".posterior", "rb"))
Lpost = posterior["loadings"]
torch.round(Lpost * 1) / 1
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD METRICS
importance = pd.read_csv(dir_results + file + "_importance.csv", index_col=0)
drop_one = pd.read_csv(dir_results + file + "_bcedrop.csv", index_col=0)
# -----------------------------------------------------------------------------