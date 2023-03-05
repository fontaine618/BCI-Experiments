import sys
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
import torch
import pickle
from src.results import MCMCResults
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SETUP
dir = "/home/simfont/Documents/BCI/experiments/test5/"
dir_train = dir + "data/train/"
dir_test = dir + "data/test/"
dir_chains = dir + "chains/"
dir_figures = dir + "figures/"
dir_results = dir + "results/"

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
# -----------------------------------------------------------------------------



# =============================================================================
results = MCMCResults.from_files(
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
return_cumulative = True

self = results[n].to_predict(n_samples=10)
factor_samples = 10
factor_processes_method = "analytical"
aggregation_method = "product"
# character_idx = torch.arange(0, nc).repeat(nr).int()
character_idx = torch.arange(0, nc).repeat_interleave(nr).int()

log_prob, wide_pred_one_hot, chars = self.predict(
	order=order,
	sequence=sequence,
	factor_samples=factor_samples,
	character_idx=character_idx,
	factor_processes_method=factor_processes_method,
	aggregation_method=aggregation_method,
	return_cumulative=return_cumulative
)

target_ = target[::nr, :].unsqueeze(1).repeat(1, nr, 1)
hamming = (wide_pred_one_hot != target_).double().sum(2).sum(0) / 2
acc = (wide_pred_one_hot == target_).all(2).double().sum(0)

df = pd.DataFrame({
	"hamming": hamming,
	"acc": acc
}, index=range(1, nr))


df.to_csv(dir_results + f"train_nrep{n:02}.csv")
# -----------------------------------------------------------------------------




# =============================================================================
# Testing
nr = 15
nc = 100

order = test_order
sequence = test_sequence
target = test_target
return_cumulative = True

self = results[n].to_predict(n_samples=10)
factor_samples = 10
factor_processes_method = "analytical"
aggregation_method = "product"
# character_idx = torch.arange(0, nc).repeat(nr).int()
character_idx = torch.arange(0, nc).repeat_interleave(nr).int()

log_prob, wide_pred_one_hot, chars = self.predict(
	order=order,
	sequence=sequence,
	factor_samples=factor_samples,
	character_idx=character_idx,
	factor_processes_method=factor_processes_method,
	aggregation_method=aggregation_method,
	return_cumulative=return_cumulative
)

target_ = target[::nr, :].unsqueeze(1).repeat(1, nr, 1)
hamming = (wide_pred_one_hot != target_).double().sum(2).sum(0) / 2
acc = (wide_pred_one_hot == target_).all(2).double().sum(0)

df = pd.DataFrame({
	"hamming": hamming,
	"acc": acc
}, index=range(1, nr))


df.to_csv(dir_results + f"test_nrep{n:02}.csv")
# -----------------------------------------------------------------------------




