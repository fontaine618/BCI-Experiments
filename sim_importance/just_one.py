import numpy as np
import os
import sys
import math
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
import torch
import pickle
import pandas as pd
import itertools as it
from source.bffmbci import BFFMResults
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
file_true = f"Kx{Kx}_Ky{Ky}_seed{seed}"
file_data = f"Kx{Kx}_Ky{Ky}_seed{seed}"
file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.chain"

# prediction settings
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
n_samples = 100

# dimensions
n_characters = 19
n_repetitions = 15
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
observations = torch.load(dir_data + file_data + ".observations")
order = torch.load(dir_data + file_data + ".order")
target = torch.load(dir_data + file_data + ".target")
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
self = results.to_predict(n_samples=n_samples)
character_idx = torch.arange(n_characters).repeat_interleave(n_repetitions)
# -----------------------------------------------------------------------------

out = {}
for k in range(-1, K):
    print(k)
    # =============================================================================
    # GET PREDICTIVE PROBABILITIES
    if k == -1:
        drop_components = list(range(K))
    else:
        drop_components = list(range(K))
        drop_components.remove(k)
    llk_long, chars = self.predict(
        order=order,
        sequence=observations,
        factor_samples=1,
        character_idx=character_idx,
        factor_processes_method=factor_processes_method,
        drop_component=drop_components,
        batchsize=10
    )
    # save
    np.save(
        dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}_mllk_just{k}.npy",
        llk_long.cpu().numpy()
    )
    # -----------------------------------------------------------------------------

    # =============================================================================
    # COMPUTE BCE
    nreps = n_repetitions
    log_prob = -torch.logsumexp(-llk_long, dim=3) + math.log(llk_long.shape[3])
    log_prob = log_prob - torch.logsumexp(log_prob, dim=2, keepdim=True)
    target_ = target[::nreps, :].unsqueeze(1).repeat(1, nreps, 1)
    target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
    bce = (target36 * log_prob).sum(-1).sum(-1).mean(0)
    out[str(k) if k >= 0 else "None"] = bce.item()
    print(out)
    # -----------------------------------------------------------------------------

df = pd.DataFrame(
    out, index=[0]
).T.reset_index().rename(
    columns={0: "bce", "index": "drop"}
)
df["Kx"] = Kx
df["Ky"] = Ky
df["seed"] = seed
df["K"] = K
df["diff_bce"] = df["bce"] - df.loc[df["drop"] == "None", "bce"].values[0]
df.to_csv(
    dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}_bcejust.csv"
)



