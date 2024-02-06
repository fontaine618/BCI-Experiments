import numpy as np
import os
import sys
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
import torch
import itertools as it
import pandas as pd
import math
from source.bffmbci import BFFMResults
from torch.distributions import Categorical
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_results = "/home/simfont/Documents/BCI/experiments/sim_variants/results/"
dir_chains = "/home/simfont/Documents/BCI/experiments/sim_variants/chains/"
dir_data = "/home/simfont/Documents/BCI/experiments/sim_variants/data/"
os.makedirs(dir_results, exist_ok=True)

# experiments
seeds = range(1)
Kxs = [8]
Kys = [5]
models = ["LR-DCR", "LR-DC", "LR-SC"]
K = 8

# combinations
combinations = it.product(seeds, Kxs, Kys, models, models)

# current experiment from command line
i = int(sys.argv[1])
seed, Kx, Ky, mtrue, mfitted = list(combinations)[i]

# file
file_data = f"Kx{Kx}_Ky{Ky}_seed{1000+seed}_model{mtrue}"
file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}_model{mfitted}"
file_out = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}_model{mfitted}"

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
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD RESULTS
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file_chain + ".chain"],
    warmup=0,
    thin=1
)
self = results.to_predict(n_samples=n_samples)
character_idx = torch.arange(n_characters).repeat_interleave(n_repetitions)
# -----------------------------------------------------------------------------



# =============================================================================
# GET PREDICTIVE PROBABILITIES
filename = dir_results + file_out + ".npy"
if not os.path.isfile(filename):
    llk_long, chars = self.predict(
        order=order,
        sequence=observations,
        factor_samples=1,
        character_idx=character_idx,
        factor_processes_method=factor_processes_method,
        drop_component=None,
        batchsize=20
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
# COMPUTE BCE
nreps = n_repetitions
log_prob = -torch.logsumexp(-llk_long, dim=3) + math.log(llk_long.shape[3])
log_prob = log_prob - torch.logsumexp(log_prob, dim=2, keepdim=True)
target_ = target[::nreps, :].unsqueeze(1).repeat(1, nreps, 1)
target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
bce = (target36 * log_prob).sum(-1).sum(-1).mean(0)
bce_se = (target36 * log_prob).sum(-1).sum(-1).std(0) / np.sqrt(nreps)

df = pd.DataFrame({
    "bce": bce.item(),
    "bce_se": bce_se.item(),
    "sample_mean": sample_mean,
    "which_first": which_first,
    "method": "BFFM",
    "seed": seed,
    "Kx": Kx,
    "Ky": Ky,
    "K": K,
    "dataset": "test",
    "model_true": mtrue,
    "model_fitted": mfitted
}, index=[1])
df.to_csv(dir_results + file_out + ".test")
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
    "dataset": "test",
    "model_true": mtrue,
    "model_fitted": mfitted
}, index=range(1, nreps + 1))
print(df)

df.to_csv(dir_results + file_out + ".testagg")
# -----------------------------------------------------------------------------


