import pandas as pd
import numpy as np
import os
import sys
import torch
import arviz as az
import pickle
import itertools as it
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
from source.bffmbci import BFFMResults, importance_statistic
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_results = "/home/simfont/Documents/BCI/experiments/sim_importance/results/"
dir_chains = "/home/simfont/Documents/BCI/experiments/sim_importance/chains/"
dir_data = "/home/simfont/Documents/BCI/experiments/sim_importance/data/"
os.makedirs(dir_results, exist_ok=True)

# experiments
seeds = range(5)
Kxs = [8]
Kys = [5]
Ks = [8]

# combinations
combinations = it.product(seeds, Kxs, Kys, Ks)

# current experiment from command line
i = int(sys.argv[1])
seed, Kx, Ky, K = list(combinations)[i]

# file
file_true = f"Kx{Kx}_Ky{Ky}"
file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.chain"
file_out = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.posterior"
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD DATA
variables = pickle.load(open(dir_data + file_true + ".variables", "rb"))
Ltrue = variables["loadings"]
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD RESULTS
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file_chain],
    warmup=0,
    thin=1
)
# results.procrutres_align()
results.add_transformed_variables()
# -----------------------------------------------------------------------------





# =============================================================================
# COMPUTE PORTERIOR MEAN
posterior_mean = results.posterior_mean()
with open(dir_results + file_out, "wb") as f:
	pickle.dump(posterior_mean, f)
# -----------------------------------------------------------------------------


