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
seeds = range(3)
Kxs = [8]
Kys = [5]
Ks = [8]

# combinations
combinations = it.product(seeds, Kxs, Kys, Ks)

# current experiment from command line
i = int(sys.argv[1])
seed, Kx, Ky, K = list(combinations)[i]

# file
file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.chain"
file_true = f"Kx{Kx}_Ky{Ky}_seed{seed}"
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
results.procrutres_align(Ltrue)
# -----------------------------------------------------------------------------



# =============================================================================
# COMPUTE STATISTIC
stat = importance_statistic(results.chains)

df = pd.DataFrame(stat.cpu()).reset_index().rename(
    columns={
        "index": "component",
        0: "importance"
    }
)
df["Kx"] = Kx
df["Ky"] = Ky
df["seed"] = seed
df["K"] = K
df.to_csv(
    dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}_importance.csv"
)
# -----------------------------------------------------------------------------