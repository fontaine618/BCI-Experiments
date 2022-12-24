import torch
import numpy as np
import pickle
from src.results.mcmc_results import MCMCResults
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SETUP
chains = [0, 1, 2]

# paths
dir = "/home/simon/Documents/BCI/experiments/test2/"
dir_data = dir + "data/"
dir_chains = dir + "chains/"
dir_figures = dir + "figures/"

with open(dir_data + "order.pkl", "rb") as f:
	order = pickle.load(f)
with open(dir_data + "target.pkl", "rb") as f:
	target = pickle.load(f)
with open(dir_data + "sequence.pkl", "rb") as f:
	sequence = pickle.load(f)
with open(dir_data + "settings.pkl", "rb") as f:
	settings = pickle.load(f)
with open(dir_data + "prior_parameters.pkl", "rb") as f:
	prior_parameters = pickle.load(f)
with open(dir_data + "true_values.pkl", "rb") as f:
	true_values = pickle.load(f)
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD CHAINS
results = {
	seed: MCMCResults.load(
		dir_chains + f"seed{seed}.chain"
	)
	for seed in chains
}
# -----------------------------------------------------------------------------


# =============================================================================
# COMPUTE METRICS EVERY 1000 ITERATIONS
metrics = {
	seed: result.moving_metrics(true_values, 1000)
	for seed, result in results.items()
}

# get all metrics
var_metric_list = set()
for moving_metrics, meta in metrics.values():
	for metrics_dict in moving_metrics.values():
		for var, var_metrics in metrics_dict.items():
			for metric in var_metrics.keys():
				var_metric_list.add((var, metric))
# -----------------------------------------------------------------------------


# =============================================================================
# PLOT METRIC FIGURES
for var, metric in var_metric_list:
	fig, ax = plt.subplots()
	for chain_id, (moving_metrics, meta) in metrics.items():
		x = [(m["lower"] + m["upper"])/2 for k, m in meta.items()]
		x = np.array(x)
		y = [m[var][metric] for k, m in moving_metrics.items()]
		y = np.array(y)
		ax.plot(x, y, label=chain_id, c=f"C{chain_id}")
	ax.set_title(var)
	ax.set_ylabel(metric)
	ax.legend(title="chain")
	if not (("likelihood" in var) or ("similarity" in metric)):
		ax.set_yscale("log")
	fig.savefig(f"{dir_figures}/{var}.{metric}.pdf")
	plt.close(fig)
# -----------------------------------------------------------------------------




