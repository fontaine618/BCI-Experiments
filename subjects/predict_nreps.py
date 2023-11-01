import sys
import os
import torch
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.bffmbci import BFFMResults
import pandas as pd
from torch.distributions import Categorical
from source.data.k_protocol import KProtocol

# =============================================================================
# SETUP
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_chains = "/home/simfont/Documents/BCI/experiments/subjects/chains/"
dir_results = "/home/simfont/Documents/BCI/experiments/subjects/predict/"
os.makedirs(dir_results, exist_ok=True)

# file
type = "TRN"
subject = "114"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"
trn_reps = int(sys.argv[1])
file = f"K{subject}_trnreps{trn_reps}.chain"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8


# model
seed = 0
K = 8
n_iter = 20_000
cor = 0.7
shrinkage = 5.
heterogeneity = 3.
xi_var = 0.1
sparse = False

# settings
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
return_cumulative = True
n_samples = 100

out = []
# -----------------------------------------------------------------------------



# =============================================================================
# training accuracy
eeg = KProtocol(
    filename=filename,
    type=type,
    subject=subject,
    session=session,
    window=window,
    bandpass_window=bandpass_window,
    bandpass_order=bandpass_order,
    downsample=downsample,
)
eeg = eeg.repetitions(list(range(1, trn_reps+1)))
nchars = eeg.stimulus_data["character"].nunique()
nreps = eeg.stimulus_data["repetition"].nunique()

order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx

torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file],
    warmup=0,  # already warmed up
    thin=1
)
self = results.to_predict(n_samples=n_samples)

llk_long, chars = self.predict(
    order=order,
    sequence=sequence,
    character_idx=character_idx,
    factor_processes_method=factor_processes_method,
    drop_component=None
)

log_prob = self.aggregate(
    llk_long,
    sample_mean=sample_mean,
    which_first=which_first
)

wide_pred_one_hot = self.get_predictions(log_prob, True)

entropy = Categorical(logits=log_prob).entropy()

target_ = target[::nreps, :].unsqueeze(1).repeat(1, nreps, 1)
hamming = (wide_pred_one_hot != target_).double().sum(2).mean(0) / 2
acc = (wide_pred_one_hot == target_).all(2).double().mean(0)

target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
bce = (target36 * log_prob).sum(-1).mean(0)

df = pd.DataFrame({
    "hamming": hamming.cpu(),
    "acc": acc.cpu(),
    "mean_entropy": entropy.mean(0).abs().cpu(),
    "bce": bce.cpu(),
    "dataset": "training",
    "repetition": range(1, nreps + 1),
    "subject": subject,
    "sample_mean": sample_mean,
    "which_first": which_first,
    "drop_component": "None",
    "method": "BFFM",
    "training_reps": trn_reps
}, index=range(1, nreps + 1))
print(df)
print(df[["acc", "bce", "mean_entropy"]])
out.append(df)
# -----------------------------------------------------------------------------


# =============================================================================
# training accuracy
eeg = KProtocol(
    filename=filename,
    type=type,
    subject=subject,
    session=session,
    window=window,
    bandpass_window=bandpass_window,
    bandpass_order=bandpass_order,
    downsample=downsample,
)
eeg = eeg.repetitions(list(range(trn_reps+1, 16)), True)
nchars = eeg.stimulus_data["character"].nunique()
nreps = eeg.stimulus_data["repetition"].nunique()

order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx

torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file],
    warmup=0,  # already warmed up
    thin=1
)
self = results.to_predict(n_samples=n_samples)

llk_long, chars = self.predict(
    order=order,
    sequence=sequence,
    character_idx=character_idx,
    factor_processes_method=factor_processes_method,
    drop_component=None
)

log_prob = self.aggregate(
    llk_long,
    sample_mean=sample_mean,
    which_first=which_first
)

wide_pred_one_hot = self.get_predictions(log_prob, True)

entropy = Categorical(logits=log_prob).entropy()

target_ = target[::nreps, :].unsqueeze(1).repeat(1, nreps, 1)
target_ = target_.flip(0)
hamming = (wide_pred_one_hot != target_).double().sum(2).mean(0) / 2
acc = (wide_pred_one_hot == target_).all(2).double().mean(0)

target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
bce = (target36 * log_prob).sum(-1).mean(0)

df = pd.DataFrame({
    "hamming": hamming.cpu(),
    "acc": acc.cpu(),
    "mean_entropy": entropy.mean(0).abs().cpu(),
    "bce": bce.cpu(),
    "dataset": "testing",
    "repetition": range(1, nreps + 1),
    "subject": subject,
    "sample_mean": sample_mean,
    "which_first": which_first,
    "drop_component": "None",
    "method": "BFFM",
    "training_reps": trn_reps
}, index=range(1, nreps + 1))
print(df)
print(df[["acc", "bce", "mean_entropy"]])
out.append(df)
# -----------------------------------------------------------------------------





# =============================================================================
df = pd.concat(out)
os.makedirs(dir_results, exist_ok=True)
df.to_csv(dir_results + f"K{subject}_trnreps{trn_reps}.pred")
print(df)
# -----------------------------------------------------------------------------

