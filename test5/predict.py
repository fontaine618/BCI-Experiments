import sys
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
import torch
import pickle
from src.results import BFFMResults
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical

plt.style.use("seaborn-v0_8-whitegrid")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SETUP
dir = "/home/simfont/Documents/BCI/experiments/test5/"
dir_train = dir + "data/train/"
dir_test = dir + "data/test/"
dir_chains = dir + "chains/"
dir_figures = dir + "figures/"
dir_results = dir + "results/posterior_mean/"

with open(dir_train + "order.pkl", "rb") as f:
	train_order = pickle.load(f)
with open(dir_train + "target.pkl", "rb") as f:
	train_target = pickle.load(f)
with open(dir_train + "sequence.pkl", "rb") as f:
	train_sequence = pickle.load(f)
with open(dir_train + "settings.pkl", "rb") as f:
	train_settings = pickle.load(f)
with open(dir_train + "prior_parameters.pkl", "rb") as f:
	train_prior_parameters = pickle.load(f)
with open(dir_train + "values.pkl", "rb") as f:
	train_values = pickle.load(f)

with open(dir_test + "order.pkl", "rb") as f:
	test_order = pickle.load(f)
with open(dir_test + "target.pkl", "rb") as f:
	test_target = pickle.load(f)
with open(dir_test + "sequence.pkl", "rb") as f:
	test_sequence = pickle.load(f)

n = int(sys.argv[1])

factor_samples = 1
factor_processes_method = "posterior_mean"
aggregation_method = "product"
return_cumulative = True
n_samples = 100
# -----------------------------------------------------------------------------



# =============================================================================
results = BFFMResults.from_files(
	[dir_chains + f"nrep{n:02}.chain"],
	warmup=10_000,
	thin=1
)
# -----------------------------------------------------------------------------






# =============================================================================
# Training
nr = 15
nc = 20

order = train_order
sequence = train_sequence
target = train_target

self = results.to_predict(n_samples=n_samples)
character_idx = torch.arange(0, nc).repeat_interleave(nr).int()

log_prob, wide_pred_one_hot, _ = self.predict(
	order=order,
	sequence=sequence,
	factor_samples=factor_samples,
	character_idx=character_idx,
	factor_processes_method=factor_processes_method,
	aggregation_method=aggregation_method,
	return_cumulative=return_cumulative
)

entropy = Categorical(logits=log_prob).entropy()

target_ = target[::nr, :].unsqueeze(1).repeat(1, nr, 1)
hamming = (wide_pred_one_hot != target_).double().sum(2).sum(0) / 2
acc = (wide_pred_one_hot == target_).all(2).double().sum(0)

target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_))
bce = (target36 * log_prob).sum(-1)._mean(0)

df = pd.DataFrame({
	"hamming": hamming.cpu(),
	"acc": acc.cpu(),
	"max_entropy": entropy.max(0)[0].abs().cpu(),
	"mean_entropy": entropy.mean(0).abs().cpu(),
	"min_max_proba": log_prob.max(2)[0].min(0)[0].cpu().exp(),
	"bce": bce.cpu()
}, index=range(1, nr+1))


df.to_csv(dir_results + f"train_nrep{n:02}.csv")
# -----------------------------------------------------------------------------




# =============================================================================
# Testing
nr = 15
nc = 100

order = test_order
sequence = test_sequence
target = test_target

self = results.to_predict(n_samples=n_samples)
character_idx = torch.arange(0, nc).repeat_interleave(nr).int()

log_prob, wide_pred_one_hot, _ = self.predict(
	order=order,
	sequence=sequence,
	factor_samples=factor_samples,
	character_idx=character_idx,
	factor_processes_method=factor_processes_method,
	aggregation_method=aggregation_method,
	return_cumulative=return_cumulative
)

entropy = Categorical(logits=log_prob).entropy()

target_ = target[::nr, :].unsqueeze(1).repeat(1, nr, 1)
hamming = (wide_pred_one_hot != target_).double().sum(2).sum(0) / 2
acc = (wide_pred_one_hot == target_).all(2).double().sum(0)

target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_))
bce = (target36 * log_prob).sum(-1)._mean(0)

df = pd.DataFrame({
	"hamming": hamming.cpu(),
	"acc": acc.cpu(),
	"max_entropy": entropy.max(0)[0].abs().cpu(),
	"mean_entropy": entropy.mean(0).abs().cpu(),
	"min_max_proba": log_prob.max(2)[0].min(0)[0].cpu().exp(),
	"bce": bce.cpu()
}, index=range(1, nr+1))


df.to_csv(dir_results + f"test_nrep{n:02}.csv")
# -----------------------------------------------------------------------------




