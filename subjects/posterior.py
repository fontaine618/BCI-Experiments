import torch
import numpy as np
import arviz as az
import pickle
from source.bffmbci import BFFMResults
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import networkx as nx

plt.style.use("seaborn-v0_8-whitegrid")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SETUP

dir = "/home/simon/Documents/BCI/experiments/k114/"
dir_chains = dir + "chains/"
dir_figures = dir + "posterior/"

# type = "TRN"
# subject = "114"
# session = "001"
# name = f"K{subject}_{session}_BCI_{type}"
# filename = dir_data + name + ".mat"
window = 800.0
sampling_rate = 256.
# bandpass_window = (0.1, 15.0)
# bandpass_order = 2
downsample = 8
# factor_samples = 10
# factor_processes_method = "analytical"
# sample_mean = "harmonic"
# which_first = "sample"
# return_cumulative = True
# n_samples = 15
# nchars = 19
#
# nreps = 7
# seed = 0
# cor = 0.5
# shrinkage = 5.
file = f"seed0.chain"
K = 8
channel_names = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                       'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
# -----------------------------------------------------------------------------


# =============================================================================

torch.cuda.empty_cache()
results = BFFMResults.from_files(
	[dir_chains + file],
	warmup=10_000,
	thin=1
)

results.align()
results.add_transformed_variables()
# -----------------------------------------------------------------------------


# =============================================================================
# Plot posterior
# data = results.to_arviz()
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
		"smgp_scaling.difference_process": f"$\\beta^\\xi_{{1,{dim}}}-\\beta^\\xi_{{0,{dim}}}$",

		"nontarget_mean_process.componentwise": f"$\\beta^z_{{0,{dim}}}\\exp(\\beta^\\xi_{{0,{dim}}})$",
		"target_mean_process.componentwise": f"$\\beta^z_{{1,{dim}}}\\exp(\\beta^\\xi_{{1,{dim}}})$",
		"difference_mean_process.componentwise": f"$\\beta^z_{{1,{dim}}}\\exp(\\beta^\\xi_{{1,{dim}}})-\\beta^z_{{0,{dim}}}\\exp(\\beta^\\xi_{{0,{dim}}})$",

		"nontarget_mean_process.channelwise" :
			f"$\\beta^z_{{0\\cdot}}\\exp(\\beta^\\xi_{{0\cdot}})\\cdot\\Theta_{{{dim}\\cdot}}$",
		"target_mean_process.channelwise" :
			f"$\\beta^z_{{1\\cdot}}\\exp(\\beta^\\xi_{{1\cdot}})\\cdot\\Theta_{{{dim}\\cdot}}$",
		"difference_mean_process.channelwise" :
			f"$[\\beta^z_{{1\\cdot}}\\exp(\\beta^\\xi_{{1\cdot}})"
			f"-\\beta^z_{{0\\cdot}}\\exp(\\beta^\\xi_{{0\cdot}})]"
			f"\\cdot\\Theta_{{{dim}\\cdot}}$",

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

nt = results.chains["smgp_factors.nontarget_process"].shape[-1]
time_factor = sampling_rate / downsample
time = torch.arange(nt).cpu() * time_factor

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
	"smgp_scaling.difference_process",

	"nontarget_mean_process.componentwise",
	"target_mean_process.componentwise",
	"difference_mean_process.componentwise",
]:
	plt.cla()
	fig, axs = plt.subplots(K, 1, sharex="all", sharey="all", figsize=(5, 3*K))
	for k, ax in enumerate(axs):
		X = results.chains[vname][:, :, k, :].cpu()
		X = X.reshape(-1, X.shape[-1])
		low = torch.quantile(X, 0.05, axis=0)
		mean = X.mean(0)
		high = torch.quantile(X, 0.95, axis=0)
		ax.fill_between(time, low, high, alpha=0.5)
		ax.plot(time, mean)
		title = vname_to_expr(vname, k+1)
		ax.set_title(title)
		ax.axhline(0, color="k", linestyle="--")
	plt.tight_layout()
	fig.savefig(f"{dir_figures}{vname}.pdf")



for vname in [
	"heterogeneities",
	"loadings",
	"loadings.norm_one",
]:
	plt.cla()
	fig, axs = plt.subplots(1, K, sharex="all", sharey="all", figsize=(3*K, 8))
	for k, ax in enumerate(axs):
		X = results.chains[vname][:, :, :, k].cpu()
		X = X.reshape(-1, X.shape[-1])
		low = torch.quantile(X, 0.05, axis=0)
		mean = X.mean(0)
		high = torch.quantile(X, 0.95, axis=0)
		# plot pointestimate with 90% CI
		ax.scatter(mean, range(1, 17))
		yerr = torch.stack([mean-low, high-mean])
		ax.errorbar(mean, range(1, 17), xerr=yerr, fmt="none", capsize=5)
		title = vname_to_expr(vname, k+1)
		ax.set_title(title)
		ax.set_yticks(range(1, 17))
		ax.set_yticklabels(channel_names, rotation=90)
		ax.axvline(0, color="k", linestyle="--")
	plt.tight_layout()
	fig.savefig(f"{dir_figures}{vname}.pdf")


for vname in [
	"shrinkage_factor",
]:
	X = results.chains[vname].cpu()
	X = X.reshape(-1, X.shape[-1])
	K = X.shape[-1]
	low = torch.quantile(X, 0.05, axis=0)
	mean = X.mean(0)
	high = torch.quantile(X, 0.95, axis=0)
	plt.cla()
	fig, ax = plt.subplots(figsize=(5, 8))
	ax.scatter(mean, range(1, K+1))
	yerr = torch.stack([mean-low, high-mean])
	ax.errorbar(mean, range(1, K+1), xerr=yerr, fmt="none", capsize=5)
	title = vname_to_expr(vname)
	ax.set_title(title)
	ax.set_yticks(range(1, K+1))
	ax.set_yticklabels(range(1, K+1))
	ax.axvline(0, color="k", linestyle="--")
	plt.tight_layout()
	plt.show()
	fig.savefig(f"{dir_figures}{vname}.pdf")
# -----------------------------------------------------------------------------




# =============================================================================
# Plot Networks

pos = {
    'F3': (1, 4),
    'Fz': (2, 4),
    'F4': (3, 4),
    'T7': (0, 3),
    'C3': (1, 3),
    'Cz': (2, 3),
    'C4': (3, 3),
    'T8': (4, 3),
    'CP3': (1, 2),
    'CP4': (3, 2),
    'P3': (1, 1),
    'Pz': (2, 1),
    'P4': (3, 1),
    'PO7': (1, 0),
    'Oz': (2, 0),
    'PO8': (3, 0),
}


for vname in [
	"nontarget_mean_process.channelwise",
	"target_mean_process.channelwise",
	"difference_mean_process.channelwise",
]:
	plt.cla()
	fig, axs = plt.subplots(5, 5, sharex="all", sharey="all", figsize=(16, 16))
	for e, ename in enumerate(channel_names):
		col, row = pos[ename]
		ax = axs[4-row, col]
		X = results.chains[vname][:, :, e, :].cpu()
		X = X.reshape(-1, X.shape[-1])
		low = torch.quantile(X, 0.05, axis=0)
		mean = X.mean(0)
		high = torch.quantile(X, 0.95, axis=0)
		ax.fill_between(time, low, high, alpha=0.5)
		ax.plot(time, mean)
		ax.set_title(ename)
		ax.axhline(0, color="k", linestyle="--")
	plt.tight_layout()
	fig.savefig(f"{dir_figures}{vname}.pdf")



plt.cla()
fig, axs = plt.subplots(1, K, sharex="all", sharey="all", figsize=(3*K, 4))
for k, ax in enumerate(axs):
	component = results.chains["loadings"][0, :, :, k].mean(0).reshape(-1, 1)
	# point color given by sign
	colors = ["b" if c > 0 else "r" for c in component]
	# point size given by magnitude
	sizes = component.abs().pow(1.).cpu().numpy() * 10
	x = [pos[c][0] for c in channel_names]
	y = [pos[c][1] for c in channel_names]
	ax.scatter(x, y, c=colors, s=sizes)
	title = vname_to_expr("loadings", k+1)
	ax.set_title(title)
	ax.set_aspect('equal', 'box')
	# remove grid
	ax.grid(False)
	# remove ticks
	ax.set_xticks([])
	ax.set_yticks([])

plt.tight_layout()
fig.savefig(f"{dir_figures}loadings_network.pdf")



for k in range(K):
	component = results.chains["loadings"][0, :, :, k].mean(0).reshape(-1, 1)
	network = component @ component.T
	edgelist = torch.tril_indices(16, 16, offset=-1)
	weights = network[edgelist[0], edgelist[1]]
	G = nx.from_edgelist(edgelist.T.cpu().numpy(), create_using=nx.DiGraph)
	for i, (u, v) in enumerate(G.edges()):
		G.edges[u, v]['weight'] = weights[i].item()
	G = nx.relabel_nodes(G, {i: channel_names[i] for i in range(16)})
	G.edges(data=True)
	plt.cla()
	plt.figure(figsize=(5, 5))
	nx.draw_networkx_nodes(G, pos=pos, node_size=600, node_color='k', edgecolors='w', linewidths=2)
	nx.draw_networkx_edges(G, pos=pos, width=weights.pow(2.).cpu().numpy() / 1000,
						   edge_color=["b" if w > 0 else "r" for w in weights],
						   connectionstyle='arc3, rad=0.1', arrowsize=20, arrowstyle='-')
	nx.draw_networkx_labels(G, pos=pos, font_size=12, font_color='w', font_weight='bold')
	plt.axis('off')
	plt.title(f"Network {k+1}")
	plt.tight_layout()
	plt.savefig(f"{dir_figures}network_{k+1}.pdf")
# -----------------------------------------------------------------------------


