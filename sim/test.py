import numpy as np
import os
import sys
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
import torch
import itertools as it
import pandas as pd
from source.bffmbci import BFFMResults
from torch.distributions import Categorical
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
file_data = f"Kx{Kx}_Ky{Ky}_seed{seed + 1000}"
file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.chain"

# prediction settings
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
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
filename = dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}_mllk_test.npy"
# check if exists
if not os.path.isfile(filename):
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
        filename,
        llk_long.cpu().numpy()
    )
else:
    llk_long = torch.Tensor(np.load(filename))
# -----------------------------------------------------------------------------



# =============================================================================
# PREDICTIONS
llk_long2 = llk_long.reshape((-1, 36, n_samples)).unsqueeze(1)

log_prob = self.aggregate(
    llk_long2,
    sample_mean=sample_mean,
    which_first=which_first
)

wide_pred_one_hot = self.get_predictions(log_prob, True)
# -----------------------------------------------------------------------------



# =============================================================================
# METRICS
nreps = 1
entropy = Categorical(logits=log_prob).entropy()

target_ = target.unsqueeze(1)
hamming = (wide_pred_one_hot != target_).double().sum(2).mean(0) / 2
acc = (wide_pred_one_hot == target_).all(2).double().mean(0)

target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
bce = (target36 * log_prob).sum(-1).sum(0)
N = target36.shape[0]
bce_se = (target36 * log_prob).sum(-1).pow(2).sum(0).div(N).sub(bce.div(N).pow(2)).sqrt().mul(np.sqrt(N))

df = pd.DataFrame({
    "hamming": hamming.cpu(),
    "acc": acc.cpu(),
    "entropy": entropy.sum(0).cpu(),
    "bce": bce.cpu(),
    "bce_se": bce_se.cpu(),
    "repetition": range(1, nreps + 1),
    "sample_mean": sample_mean,
    "which_first": which_first,
    "method": "BFFM",
    "seed": seed,
    "Kx": Kx,
    "Ky": Ky,
    "K": K,
    "dataset": "test"
}, index=range(1, nreps + 1))
print(df)

df.to_csv(dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.test")
# -----------------------------------------------------------------------------


