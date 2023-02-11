import torch
import numpy as np
import arviz as az
import pickle
from src.results_old import MCMCResults
from src.results_old import MCMCMultipleResults
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
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

self = MCMCMultipleResults(results)

# # fix broken
#
# import types
# import io
#
# # Could derive from Unpickler, but that may be a C class, so our tracebacks would be less helpful
# class MagicUnpickler(pickle._Unpickler):
#     def __init__(self, fp):
#         super().__init__(fp)
#         self._magic_classes = {}
#
#     def find_class(self, module, name):
#         try:
#             return super().find_class(module, name)
#         except ModuleNotFoundError:
#             return self._create_magic_class(module, name)
#
#     def _create_magic_class(self, module, name):
#         cache_key = (module, name)
#         if cache_key not in self._magic_classes:
#             cls = type(f'<<Emulated Class {module}:{name}>>', (types.SimpleNamespace,), {})
#             self._magic_classes[cache_key] = cls
#         return self._magic_classes[cache_key]
#
#
# for seed in chains:
# 	filename = dir_chains + f"seed{seed}.chain"
# 	with open(filename, "rb") as f:
# 		results_old = MagicUnpickler(f).load()
# 		results_old = MCMCResults(results_old.variables, warmup=results_old.warmup, thin=results_old.thin)
# 		results_old.save(filename)


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




# =============================================================================
# arViz
import pandas as pd

# rhat after drop burnin
rhat = az.rhat(self.chains_cat.sel(draw=slice(100000, None, 10)))

rhat_value = []
rhat_varname = []
for varname, var in rhat.items():
	if varname == "observation_log_likelihood":
		continue
	values = var.values.flatten().tolist()
	rhat_value.extend(values)
	rhat_varname.extend([varname] * len(values))
rhat_varname.extend(["all"] * len(rhat_value))
rhat_value.extend(rhat_value)
rhat_pd = pd.DataFrame({
	"rhat": rhat_value,
	"variable": rhat_varname
})
axs = rhat_pd.hist(by="variable", figsize=(12, 12))
plt.tight_layout()
plt.savefig(f"{dir_figures}/rhat/all.pdf")

for vname in [
	"smgp_scaling.target_signal",
	"smgp_scaling.mixing_process",
	"smgp_scaling.nontarget_process",
	"smgp_scaling.target_process",
	"smgp_factors.target_signal",
	"smgp_factors.mixing_process",
	"smgp_factors.nontarget_process",
	"smgp_factors.target_process",
]:
	plt.clf()
	plt.figure(figsize=(8, 4))
	plt.plot(rhat[vname].T)
	plt.title(vname)
	plt.xlabel("time since stimulus onset")
	plt.ylabel("rhat")
	plt.tight_layout()
	plt.savefig(f"{dir_figures}/rhat/{vname}.pdf")
# -----------------------------------------------------------------------------




