import torch
import pandas as pd
import numpy as np
from src.models.bffmbci.bffm import BFFModel
from src.results_old.mcmc_results import MCMCResults
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# re-generate true model
latent_dim = 3

torch.manual_seed(0)
model = BFFModel.generate_from_dimensions(
	latent_dim=latent_dim,
	n_channels=7,
	stimulus_to_stimulus_interval=10,
	stimulus_window=55,
	n_stimulus=(3, 3),
	n_sequences=51,
	nonnegative_smgp=True,
	heterogeneities=3.,
	shrinkage_factor=(2., 10.)
)
true_values = model.current_values()
true_llk = model.variables["observations"].log_density
true_values["observation_log_likelihood"] = true_llk

# load chains
results = {
	chain_id: MCMCResults.load(
		f"/home/simon/Documents/BCI/experiments/test/chains/seed{chain_id}.chain"
	) for chain_id in [0, 2, 4]
}

# compute metrics every 100 iterations
metrics100 = {
	chain_id: results[chain_id].moving_metrics(true_values, 100) for chain_id in results
}

# get all variables, metric
metrics = set()
for moving_metrics, meta in metrics100.values():
	for metrics_dict in moving_metrics.values():
		for var, var_metrics in metrics_dict.items():
			for metric in var_metrics.keys():
				metrics.add((var, metric))


# plot metrics
dir_figures = "/home/simon/Documents/BCI/experiments/test/figures"
for var, metric in metrics:
	fig, ax = plt.subplots()
	for chain_id, (moving_metrics, meta) in metrics100.items():
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