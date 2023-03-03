import torch
import numpy as np
import arviz as az
import pickle
from src.results import MCMCResults, _add_transformed_variables, _flatten_dict
# from src.results_old import MCMCResults
# from src.results_old import MCMCMultipleResults
from src.bffmbci import BFFModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SETUP
chains = [0, 1, 2, 3, 4]

# paths
dir = "/home/simon/Documents/BCI/experiments/test4/"
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
results = MCMCResults.from_files(
	[dir_chains + f"seed{chain}.chain" for chain in chains],
	warmup=10_000,
	thin=1
)
results.align()
results.add_transformed_variables()
# -----------------------------------------------------------------------------




# =============================================================================
data = results.to_arviz()
rhat = az.rhat(data)
ess = az.ess(data)
true_values = _flatten_dict(true_values)
_add_transformed_variables(true_values)
# -----------------------------------------------------------------------------



# =============================================================================
# Plot RHAT
xmin = 0.95
xmax = 2.3

for k, v in rhat.data_vars.items():
	fig, ax = plt.subplots()
	vv = v.values
	if v.shape:
		if v.shape[0] == 3:
			vv = vv.T
		if len(v.shape) == 3:
			vv = np.moveaxis(vv, 2, 0).reshape(3, -1).T
	else:
		vv = vv.reshape(1, -1)
	df = pd.DataFrame(vv)
	sns.histplot(df, ax=ax, bins=np.linspace(xmin, xmax, 28),
				 multiple="stack", shrink=0.8, edgecolor="white")
	ax.set_xlim(xmin, xmax)
	ax.set_title(k)
	ax.set_ylabel("$\widehat{R}$")
	fig.savefig(f"{dir_figures}rhat/{k}.pdf")
	plt.close(fig)

# plot loadings
for k in range(3):
	fig, ax = plt.subplots()
	axs = az.plot_forest(data, var_names="loadings.norm_one", coords={"loadings.norm_one_dim_1": k},
						 show=False, ax=ax)
	fig.savefig(f"{dir_figures}/posterior/loadings_norm_one_{k}.pdf")
	plt.cla()

	fig, ax = plt.subplots()
	axs = az.plot_forest(data, var_names="loadings.times_shrinkage", coords={"loadings.times_shrinkage_dim_1": k},
						 show=False, ax=ax)
	fig.savefig(f"{dir_figures}/posterior/loadings_times_shrinkage_{k}.pdf")
	plt.cla()

	fig, ax = plt.subplots()
	axs = az.plot_forest(data, var_names="loadings", coords={"loadings_dim_1": k},
						 show=False, ax=ax)
	fig.savefig(f"{dir_figures}/posterior/loadings_{k}.pdf")
	plt.cla()

	fig, ax = plt.subplots()
	axs = az.plot_forest(data, var_names="smgp_factors.target_signal", coords={"smgp_factors.target_signal_dim_0": k},
						 show=False, ax=ax)
	fig.savefig(f"{dir_figures}/posterior/smgp_factors_target_signal_{k}.pdf")
	plt.cla()

	fig, ax = plt.subplots()
	axs = az.plot_forest(data, var_names="smgp_factors.nontarget_process", coords={"smgp_factors.nontarget_process_dim_0": k},
						 show=False, ax=ax)
	fig.savefig(f"{dir_figures}/posterior/smgp_factors_nontarget_process_{k}.pdf")
	plt.cla()

	fig, ax = plt.subplots()
	axs = az.plot_forest(data, var_names="smgp_factors.mixing_process", coords={"smgp_factors.mixing_process_dim_0": k},
						 show=False, ax=ax)
	fig.savefig(f"{dir_figures}/posterior/smgp_factors_mixing_process_{k}.pdf")
	plt.cla()

	fig, ax = plt.subplots()
	axs = az.plot_forest(data, var_names="smgp_scaling.target_signal", coords={"smgp_scaling.target_signal_dim_0": k},
						 show=False, ax=ax)
	fig.savefig(f"{dir_figures}/posterior/smgp_scaling_target_signal_{k}.pdf")
	plt.cla()

	fig, ax = plt.subplots()
	axs = az.plot_forest(data, var_names="smgp_scaling.nontarget_process", coords={"smgp_scaling.nontarget_process_dim_0": k},
						 show=False, ax=ax)
	fig.savefig(f"{dir_figures}/posterior/smgp_scaling_nontarget_process_{k}.pdf")
	plt.cla()

	fig, ax = plt.subplots()
	axs = az.plot_forest(data, var_names="smgp_scaling.mixing_process", coords={"smgp_scaling.mixing_process_dim_0": k},
						 show=False, ax=ax)
	fig.savefig(f"{dir_figures}/posterior/smgp_scaling_mixing_process_{k}.pdf")
	plt.cla()



fig, ax = plt.subplots()
axs = az.plot_forest(data, var_names="shrinkage_factor", show=False, ax=ax)
fig.savefig(f"{dir_figures}/posterior/shrinkage_factor.pdf")
plt.cla()
# -----------------------------------------------------------------------------







# =============================================================================
# New prediction class
self = results.to_predict(n_samples=10)
factor_samples = 10
factor_processes_method = "analytical"
aggregation_method = "integral"
character_idx = torch.arange(0, 19).repeat(15).int()
log_prob, wide_pred_one_hot, chars = self.predict(
	order=order,
	sequence=sequence,
	factor_samples=factor_samples,
	character_idx=character_idx,
	factor_processes_method=factor_processes_method,
	aggregation_method=aggregation_method,
	return_cumulative=True
)

nr = settings["n_repetitions"]
nc = settings["n_characters"]
target_ = target[:nc, :].unsqueeze(1).repeat(1, nr, 1)

# number of col/row errors
hamming = (wide_pred_one_hot != target_).double().sum(2).sum(0) / 2
# total accuracy
acc = (wide_pred_one_hot == target_).all(2).double().sum(0)

x = np.arange(1, 16)



fig, axs = plt.subplots(2, 2, sharex="all", figsize=(8, 6))
# hamming
axs[0, 0].plot(x, 38 - hamming.cpu())
axs[0, 0].set_ylabel("Nb. correct rows/cols")
axs[0, 0].set_title(f"factor method={factor_processes_method}\nagg method={aggregation_method}\n"
					f"factor_samples={factor_samples}\nn_posterior=10")
axs[0, 0].set_xticks(np.arange(1, 16, 2))
axs[0, 0].axhline(38, color="k", linestyle="--")

# accuracy
axs[1, 0].plot(x, acc.cpu())
axs[1, 0].set_ylabel("Correct predictions")
axs[1, 0].axhline(19, color="k", linestyle="--")

# accuracy
axs[0, 1].plot(x, log_prob[0, :, :].cpu())
axs[0, 1].set_ylabel("log probability")

# accuracy
axs[1, 1].plot(x, log_prob[16, :, :].cpu())
axs[1, 1].set_ylabel("log probability")
axs[1, 1].set_title("Sequence 16")


plt.tight_layout()
fig.savefig(f"{dir_figures}/prediction/{factor_processes_method}_{aggregation_method}_{factor_samples}.pdf")
# -----------------------------------------------------------------------------





# =============================================================================
# Plot LLK
results = MCMCResults.from_files(
	[dir_chains + f"seed{chain}.chain" for chain in chains],
	warmup=0,
	thin=10
)

fig, ax = plt.subplots()
df = pd.DataFrame(results.chains["log_likelihood.observations"].cpu().T)
sns.lineplot(
	data=df
)
ax.set_ylim(-460_000, -454_000)
ax.set_xticks(np.arange(0, 2000, 500), np.arange(0, 20000, 5000))
ax.axhline(y=true_values["observation_log_likelihood"], color="black")
ax.set_title("Obs. log-likelihood")
fig.savefig(f"{dir_figures}obsrevation_log_likelihood.pdf")
# -----------------------------------------------------------------------------






# =============================================================================
# LOAD CHAINS
# TODO Update this
results = {
	seed: MCMCResults.load(
		dir_chains + f"seed{seed}.chain"
	)
	for seed in chains
}

self = MCMCMultipleResults(results)
# -----------------------------------------------------------------------------


# =============================================================================
# COMPUTE METRICS EVERY 1000 ITERATIONS
metrics = {
	seed: result.moving_metrics(true_values, 1)
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


