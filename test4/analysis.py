import torch
import numpy as np
import arviz as az
import pickle
from src.results import BFFMResults, add_transformed_variables, _flatten_dict
# from src.results_old import MCMCResults
# from src.results_old import MCMCMultipleResults
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SETUP
chains = [0, 2, 3, 4]

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
results = BFFMResults.from_files(
	[dir_chains + f"seed{chain}.chain" for chain in chains],
	warmup=10_000,
	thin=1
)

true_values = _flatten_dict(true_values)
add_transformed_variables(true_values)

results.align(loadings=true_values["loadings"])
results.add_transformed_variables()
# -----------------------------------------------------------------------------




# =============================================================================
data = results.to_arviz()
rhat = az.rhat(data)
ess = az.ess(data)
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
# Plot ESS
xmin = 1
xmax = 10_000

for k, v in ess.data_vars.items():
	fig, ax = plt.subplots()
	vv = v.values
	if v.shape:
		if v.shape[0] == settings["latent_dim"]:
			vv = vv.T
		if len(v.shape) == 3:
			vv = np.moveaxis(vv, 2, 0).reshape(settings["latent_dim"], -1).T
	else:
		vv = vv.reshape(1, -1)
	df = pd.DataFrame(vv)
	sns.histplot(df, ax=ax, bins=np.logspace(np.log10(xmin), np.log10(xmax), 21),
				 multiple="stack", shrink=0.8, edgecolor="white")
	ax.set_xlim(xmin, xmax)
	ax.set_title(k)
	ax.set_ylabel("ESS")
	ax.set_xscale("log")
	fig.savefig(f"{dir_figures}ess/{k}.pdf")
	plt.close(fig)
# -----------------------------------------------------------------------------


# =============================================================================
# Plot posterior
from collections import defaultdict
plt.rcParams['text.usetex'] = True

def vname_to_expr(vname, dim=None):
	out = defaultdict(lambda: vname, {
		"smgp_factors.nontarget_process": f"$\\alpha^z_{{0,{dim}}}$",
		"smgp_factors.mixing_process": f"$\\zeta^z_{{{dim}}}$",
		"smgp_factors.target_process": f"$\\alpha^z_{{1,{dim}}}$",
		"smgp_factors.difference_process": f"$\\beta^z_{{1,{dim}}}-\\beta^z_{{0,{dim}}}$",
		"smgp_factors.target_signal": f"$\\beta^z_{{1,{dim}}}$",

		"smgp_scaling.nontarget_process": f"$\\alpha^\\xi_{{0,{dim}}}$",
		"smgp_scaling.mixing_process": f"$\\zeta^\\xi_{{{dim}}}$",
		"smgp_scaling.target_process": f"$\\alpha^\\xi_{{1,{dim}}}$",
		"smgp_scaling.target_signal": f"$\\beta^\\xi_{{1,{dim}}}$",
		"smgp_scaling.nontarget_process_centered": f"$\\alpha^\\xi_{{0,{dim}}}$ (centered)",
		"smgp_scaling.target_multiplier_process":
			f"$1+\\zeta^\\xi_{{{dim}}}\\delta^\\xi_{{1,{dim}}}/\\alpha^\\xi_{{0,{dim}}}$",

		"heterogeneities": f"$\\phi_{{\cdot{dim}}}$",
		"loadings": f"$\\Theta_{{\cdot{dim}}}$",
		"loadings.norm_one": f"$\\Theta_{{\cdot{dim}}} / \\Vert\\Theta_{{\cdot{dim}}}\\Vert_2$",
		"loadings.times_shrinkage": f"$\\tau_{dim}^{{1/2}}\\Theta_{{\cdot{dim}}}$",

		"observation_variance": "$\\sigma_e^2$",
		"shrinkage_factor": "$\\tau_k$",
		"log_likelihood.observations": "log-likelihood",
		"scaling_factor": "scaling factor",
	})[vname]
	return out


for vname in [
	"smgp_factors.nontarget_process",
	"smgp_factors.target_process",
	"smgp_factors.mixing_process",
	"smgp_factors.target_signal",
	"smgp_factors.difference_process",

	"smgp_scaling.nontarget_process",
	"smgp_scaling.target_process",
	"smgp_scaling.mixing_process",
	"smgp_scaling.target_signal",
	"smgp_scaling.nontarget_process_centered",
	"smgp_scaling.target_multiplier_process"
]:
	plt.cla()
	fig, axs = plt.subplots(3, 1, sharex="all", sharey="all", figsize=(5, 10))
	for k, ax in enumerate(axs):
		az.plot_forest(data, var_names=vname, coords={f"{vname}_dim_0": k}, show=False, ax=ax)
		title = vname_to_expr(vname, k+1)
		ax.set_title(title)
		ax.set_yticklabels(range(20, 0, -1))
	plt.tight_layout()
	fig.savefig(f"{dir_figures}/posterior/{vname}.pdf")

for vname in [
	"heterogeneities",
	"loadings",
	"loadings.norm_one",
	"loadings.times_shrinkage",
]:
	plt.cla()
	fig, axs = plt.subplots(3, 1, sharex="all", sharey="all", figsize=(5, 10))
	for k, ax in enumerate(axs):
		az.plot_forest(data, var_names=vname, coords={f"{vname}_dim_1": k}, show=False, ax=ax)
		title = vname_to_expr(vname, k+1)
		ax.set_title(title)
		ax.set_yticklabels(range(15, 0, -1))
	plt.tight_layout()
	fig.savefig(f"{dir_figures}/posterior/{vname}.pdf")

for vname in [
	"observation_variance",
	"shrinkage_factor",
	"log_likelihood.observations",
	"scaling_factor"
]:
	plt.cla()
	fig, ax = plt.subplots(figsize=(5, 5))
	az.plot_forest(data, var_names=vname, show=False, ax=ax)
	title = vname_to_expr(vname)
	ax.set_title(title)
	ax.set_yticklabels([])
	plt.tight_layout()
	fig.savefig(f"{dir_figures}/posterior/{vname}.pdf")
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
results = BFFMResults.from_files(
	[dir_chains + f"seed{chain}.chain" for chain in chains],
	warmup=0,
	thin=10
)

fig, ax = plt.subplots()
df = pd.DataFrame(results.chains["log_likelihood.observations"].cpu().T)
sns.lineplot(
	data=df
)
ax.set_ylim(-456_000, -454_000)
ax.set_xticks(np.arange(0, 2000, 500), np.arange(0, 20000, 5000))
ax.axhline(y=true_values["observation_log_likelihood"], color="black")
ax.set_title("Obs. log-likelihood")
fig.savefig(f"{dir_figures}obsrevation_log_likelihood.pdf")
# -----------------------------------------------------------------------------



