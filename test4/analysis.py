import torch
import numpy as np
import arviz as az
import pickle
from src.results import MCMCResults, _add_transformed_variables, _flatten_dict
# from src.results_old import MCMCResults
# from src.results_old import MCMCMultipleResults
from src.bffmbci import BFFMPredict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SETUP
chains = [0, 1, 2]

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
	warmup=0,
	thin=10
)
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
# -----------------------------------------------------------------------------



# =============================================================================
# Plot LLK
fig, ax = plt.subplots()
df = pd.DataFrame(results.chains["log_likelihood.observations"].cpu().T)
sns.lineplot(
	data=df
)
ax.set_ylim(-460_000, -454_000)
ax.set_xticks(np.arange(0, 25_000, 5_000), np.arange(0, 250_000, 50_000))
ax.axhline(y=true_values["observation_log_likelihood"], color="black")
ax.set_title("Obs. log-likelihood")
fig.savefig(f"{dir_figures}obsrevation_log_likelihood.pdf")

# plot loadings
for k in range(3):
	fig, ax = plt.subplots()
	axs = az.plot_forest(data, var_names="loadings", coords={"loadings_dim_1": k},
						 show=False, ax=ax)
	fig.savefig(f"{dir_figures}/posterior/loadings_{k}.pdf")
# -----------------------------------------------------------------------------





# =============================================================================
# Prediction
self = results.to_predict(n_samples=1000)
factor_samples = 10
log_probs, pred_one_hot = self.predict(order, sequence, factor_samples)

# Hamming (2.43 -> about one swap)
(pred_one_hot != target).double().sum(1).mean().item()

# Accuracy (14.5%)
(pred_one_hot == target).all(1).double().mean().item()

# Recall (39.25%)
(pred_one_hot == target)[target==1].double().mean().item()

# At least one correct (64%)
(pred_one_hot == target)[:, :3].all(1).double().mean().item() + \
(pred_one_hot == target)[:, 3:].all(1).double().mean().item() - \
(pred_one_hot == target).all(1).double().mean().item()

# aggregate over repetitions
nr = settings["n_repetitions"]
nc = settings["n_characters"]
log_probs_rep = torch.cat([
	log_probs[i::nc, :].unsqueeze(0)
	for i in range(nc)
])
agg_log_probs = log_probs_rep.cumsum(1)
agg_pred_id = agg_log_probs.argmax(2)
agg_pred_one_hot = self.combinations[agg_pred_id, :]
target_ = target[:nc, :].unsqueeze(1).repeat(1, nr, 1)

# number of col/row errors
(agg_pred_one_hot != target_).double().sum(2).sum(0) / 2
# total accuracy
(agg_pred_one_hot == target_).all(2).double().sum(0)

character_idx = torch.arange(nc).repeat(nr)
# -----------------------------------------------------------------------------





















# tmp
llks = dict()
for chain in chains:
	with open(dir_chains + f"seed{chain}.chain", "rb") as f:
		result = pickle.load(f)
		llks[chain] = result["log_likelihood"]

tllk = true_values["observation_log_likelihood"]

fig, ax = plt.subplots()
for chain, llk in llks.items():
	ax.plot(llk["observations"], label=chain, c=f"C{chain}")
ax.set_title("observation log-likelihood")
ax.legend(title="chain")
ax.axhline(tllk, c="k", ls="--")
ax.set_ylim(-120000, -110000)
fig.savefig(f"{dir_figures}/llk.pdf")
plt.close(fig)


fig, ax = plt.subplots()
for chain, llk in llks.items():
	ax.plot(np.abs(np.array(llk["observations"]) - tllk), label=chain, c=f"C{chain}")
ax.set_title("observation log-likelihood absdiff")
ax.legend(title="chain")
ax.set_yscale("log")
fig.savefig(f"{dir_figures}/llk_absdiff.pdf")
plt.close(fig)




file = "/home/simon/Documents/BCI/experiments/test3/chains/seed0.chain"












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




