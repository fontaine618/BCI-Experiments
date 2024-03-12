import numpy as np
import os
import sys
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
import torch
import itertools as it
from source.bffmbci import BFFMResults
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_results = "/home/simfont/Documents/BCI/experiments/sim/results/"
dir_chains = "/home/simfont/Documents/BCI/experiments/sim/chains/"
dir_data = "/home/simfont/Documents/BCI/experiments/sim/data/"
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
file_data = f"Kx{Kx}_Ky{Ky}_seed{seed}"
file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.chain"

# prediction settings
factor_processes_method = "analytical"
n_samples = 100

# dimensions
n_characters = 19
n_repetitions = 5
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
observations = torch.load(dir_data + file_data + ".observations")
order = torch.load(dir_data + file_data + ".order")
target = torch.load(dir_data + file_data + ".target")
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD RESULTS
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file_chain],
    warmup=0,
    thin=1
)
self = results.to_predict(n_samples=n_samples)
character_idx = torch.arange(n_characters).repeat_interleave(n_repetitions)
# -----------------------------------------------------------------------------



# =============================================================================
# GET PREDICTIVE PROBABILITIES
llk_long, chars = self.predict(
    order=order,
    sequence=observations,
    factor_samples=1,
    character_idx=character_idx,
    factor_processes_method=factor_processes_method,
    drop_component=None,
    batchsize=10
)
# save
np.save(
    dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}_mllk.npy",
    llk_long.cpu().numpy()
)
# -----------------------------------------------------------------------------


