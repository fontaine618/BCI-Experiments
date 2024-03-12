import numpy as np
import os
import sys
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
import torch
import pandas as pd
import itertools as it
import torchmetrics
from source.bffmbci import BFFMResults
from source.data.k_protocol import KProtocol
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from torch.distributions import Categorical

# =============================================================================
# SETUP
type = "TRN"
subject = str(sys.argv[1])
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_results = f"/home/simfont/Documents/BCI/experiments/subject/results/K{subject}/"
dir_chains = f"/home/simfont/Documents/BCI/experiments/subject/chains/K{subject}/"
os.makedirs(dir_results, exist_ok=True)
filename = dir_data + name + ".mat"


# file
type = "TRN"
subject = str(sys.argv[1])
session = "001"
name = f"K{subject}_{session}_BCI_{type}"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8

# model
lite = False
seed = 0
K = 3 if lite else 8
V = "LR-SC" if lite else "LR-DCR"
cor = 0.50
n_iter = 20_000

# experiment
seeds = range(10)
train_reps = [3, 5, 8]
experiment = list(it.product(seeds, train_reps))
seed, train_reps = experiment[int(sys.argv[2])]

# prediction settings
factor_processes_method = "analytical"
n_samples = 100
sample_mean = "arithmetic"
which_first = "sequence"

# dimensions
n_characters = 19
n_repetitions = 15
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
eeg = KProtocol(
    filename=dir_data + name + ".mat",
    type=type,
    subject=subject,
    session=session,
    window=window,
    bandpass_window=bandpass_window,
    bandpass_order=bandpass_order,
    downsample=downsample,
)
# subset training reps
torch.manual_seed(seed)
reps = torch.randperm(15) + 1
training_reps = reps[:train_reps].cpu().tolist()
testing_reps = reps[train_reps:].cpu().tolist()
eeg = eeg.repetitions(testing_reps)

nchars = eeg.stimulus_data["character"].nunique()
nreps = eeg.stimulus_data["repetition"].nunique()
order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx
# -----------------------------------------------------------------------------




# =============================================================================
# LOAD RESULTS
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + f"K{subject}_trn{train_reps}_seed{seed}{'_lite' if lite else ''}.chain"],
    warmup=0,
    thin=1
)
self = results.to_predict(n_samples=n_samples)
# -----------------------------------------------------------------------------





# =============================================================================
# GET PREDICTIVE PROBABILITIES
filename = dir_results + f"K{subject}_trn{train_reps}_seed{seed}{'_lite' if lite else ''}_testmllk.npy"
llk_long = torch.Tensor(np.load(filename)) # n_chars x n_reps x 36 x n_samples
# -----------------------------------------------------------------------------





# =============================================================================
# PREDICTIONS
log_prob = self.aggregate(
    llk_long,
    sample_mean=sample_mean,
    which_first=which_first
) # n_chars x n_reps x 36
wide_pred_one_hot = self.get_predictions(log_prob, True) # n_chars x n_reps x 12
# -----------------------------------------------------------------------------





# =============================================================================
# METRICS

# entropy
entropy = Categorical(logits=log_prob).entropy()
mean_entropy = entropy.mean(0)

# accuracy & hamming
target_wide = target.view(nchars, nreps, -1)
accuracy = (wide_pred_one_hot == target_wide).all(2).double().mean(0)
hamming = (wide_pred_one_hot != target_wide).double().sum(2).mean(0) / 2

# binary cross-entropy
target_char = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_wide), 36)
bce = - (target_char * log_prob).sum(2).mean(0)

# auc
target_char_int = torch.argmax(target_char, -1)
auc = torch.Tensor([
    torchmetrics.functional.classification.multiclass_auroc(
        preds=log_prob[:, c, :],
        target=target_char_int[:, c],
        num_classes=36,
        average="weighted"
    ) for c in range(nreps)
])

# save
df = pd.DataFrame({
    "hamming": hamming.cpu(),
    "acc": accuracy.cpu(),
    "mean_entropy": entropy.mean(0).abs().cpu(),
    "bce": bce.cpu(),
    "auroc": auc.cpu(),
    "dataset": name + "_test",
    "repetition": range(1, nreps + 1),
    "aggregation": factor_processes_method,
    "sample_mean": sample_mean,
    "which_first": which_first,
    "training_reps": train_reps,
    "method": V,
    "K": K,
    "cor": cor
}, index=range(1, nreps + 1))
df.to_csv(dir_results + f"K{subject}_trn{train_reps}_seed{seed}{'_lite' if lite else ''}.test")
# -----------------------------------------------------------------------------


