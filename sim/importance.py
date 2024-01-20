import pandas as pd
import numpy as np
import os
import sys
import torch
import arviz as az
import itertools as it
sys.path.insert(1, '/home/simon/Documents/BCI/src')
from source.bffmbci import BFFMResults, importance_statistic
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_results = "/home/simon/Documents/BCI/experiments/sim/results/"
dir_chains = "/home/simon/Documents/BCI/experiments/sim/chains/"
dir_data = "/home/simon/Documents/BCI/experiments/sim/data/"
os.makedirs(dir_results, exist_ok=True)

# experiments
seeds = range(3)
Kxs = [5, 8]
Kys = [3, 5]
Ks = range(1, 11)

# combinations
combinations = it.product(seeds, Kxs, Kys, Ks)

# current experiment from command line
i = int(sys.argv[1])
seed, Kx, Ky, K = list(combinations)[i]

# file
file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.chain"
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD RESULTS
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file_chain],
    warmup=0,
    thin=1
)
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