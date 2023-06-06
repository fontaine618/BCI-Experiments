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
import networkx as nx

plt.style.use("seaborn-v0_8-whitegrid")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SETUP
Ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# paths
dir = "/home/simon/Documents/BCI/experiments/real_data1/"
dir_chains = dir + "chains/K114_001_BCI_TRN/"
dir_figures = dir + "figures_K/"
# -----------------------------------------------------------------------------


# =============================================================================
results = {K: BFFMResults.from_files(
	[dir_chains + f"seed0_K{K}.chain"],
	warmup=0,
	thin=10
) for K in Ks}

llks = {K: results[K].chains["log_likelihood.observations"].cpu().numpy().flatten()[:2000] for K in Ks}
llk_df = pd.DataFrame(llks)
# -----------------------------------------------------------------------------



# =============================================================================
# Plot LLK
fig, ax = plt.subplots()
sns.lineplot(
	data=llk_df,
	alpha=0.5
)
ax.set_ylim(-1_050_000, -550_000)
ax.set_xticks(np.arange(0, 2001, 500), np.arange(0, 20001, 5000))
ax.set_title("Latent dimension")
ax.set_xlabel("Iteration")
ax.set_ylabel("Log-likelihood")
# ax.set_xscale("log")
fig.savefig(f"{dir_figures}observation_log_likelihood.pdf")
# -----------------------------------------------------------------------------


results = {K: BFFMResults.from_files(
	[dir_chains + f"seed0_K{K}.chain"],
	warmup=10_000,
	thin=10
) for K in Ks}


# =============================================================================
# scaling factor
scaling = {}
for K in Ks:
	results[K].add_transformed_variables()
	chain = results[K].chains["shrinkage_factor"].cpu().numpy()[0, ...]
	mean = np.mean(chain, axis=0)
	std = np.std(chain, axis=0)
	scaling[K] = pd.DataFrame({"mean": mean, "std": std, "K": K, "k": np.arange(1, K+1)})

scaling_df = pd.concat(scaling.values())
scaling_df["K"] = scaling_df["K"].astype(str)

fig, ax = plt.subplots()
sns.lineplot(
	data=scaling_df,
	x="k",
	y="mean",
	hue="K",
	ax=ax
)
ax.set_yscale("log")
ax.set_xlabel("Latent index")
ax.set_ylabel("Scaling factor")
fig.savefig(f"{dir_figures}scaling_factor.pdf")
# -----------------------------------------------------------------------------



# =============================================================================
# Plot loadings
xs = np.arange(0, 16)
fig, axs = plt.subplots(10, 9, figsize=(16, 16), sharex=True, sharey=True)
for K in Ks:
	chain = results[K].chains["loadings"].cpu().numpy()[0, ...]
	mean = np.mean(chain, axis=0)
	std = np.std(chain, axis=0)
	for k in range(K):
		axs[k, K-2].scatter(xs, mean[:, k], label=f"K={K}")
		axs[k, K-2].axhline(0, color="black", linestyle="--")
	axs[0, K-2].set_title(f"K={K}")
for k in range(10):
	axs[k, 0].set_ylabel(f"Latent {k+1}")
plt.tight_layout()
fig.savefig(f"{dir_figures}loadings.pdf")



channel_names = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                       'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

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

fig, axs = plt.subplots(10, 9, figsize=(16, 16), sharex=True, sharey=True)
for K in Ks:
	chain = results[K].chains["loadings"].cpu().numpy()[0, ...]
	mean = np.mean(chain, axis=0)
	for k in range(K):
		component = torch.Tensor(mean[:, [k]])
		network = component @ component.T
		edgelist = torch.tril_indices(16, 16, offset=-1)
		weights = network[edgelist[0], edgelist[1]]
		G = nx.from_edgelist(edgelist.T.cpu().numpy(), create_using=nx.DiGraph)
		for i, (u, v) in enumerate(G.edges()):
			G.edges[u, v]['weight'] = weights[i].item()
		G = nx.relabel_nodes(G, {i: channel_names[i] for i in range(16)})
		G.edges(data=True)
		nx.draw_networkx_nodes(G, pos=pos, node_size=100, node_color='k', edgecolors=None, linewidths=None, ax=axs[k, K-2])
		nx.draw_networkx_edges(G, pos=pos, width=weights.abs().cpu().numpy() / 200,
							   edge_color=["b" if w > 0 else "r" for w in weights],
							   connectionstyle='arc3, rad=0.1', arrowsize=20, arrowstyle='-', ax=axs[k, K-2])
		# nx.draw_networkx_labels(G, pos=pos, font_size=12, font_color='w', font_weight='bold', ax=axs[k, K-2])
		axs[k, K-2].set_axis_off()
	axs[0, K-2].set_title(f"K={K}")
for k in range(10):
	axs[k, 0].set_ylabel(f"Latent {k+1}")
plt.tight_layout()
fig.savefig(f"{dir_figures}networks.pdf")
# -----------------------------------------------------------------------------
