import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import pickle
import itertools as it
from source.bffmbci import importance_statistic
plt.style.use('seaborn-whitegrid')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.bffmbci import BFFMResults
import networkx as nx

# =============================================================================
# SETUP
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
dir_chains = "/home/simon/Documents/BCI/experiments/subject/chains/"
dir_results = "/home/simon/Documents/BCI/experiments/subject/results/"

# experiments
subject = "114"
dir_figures = f"/home/simon/Documents/BCI/experiments/subject/figures/K{subject}/"
os.makedirs(dir_figures, exist_ok=True)
K = 8

# file
file_chain = f"K{subject}/K{subject}.chain"

# channels
channels = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                       'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

channel_positions = {
    'F3': (1.25, 4),
    'Fz': (2, 4),
    'F4': (2.75, 4),
    'T7': (0, 3),
    'C3': (1, 3),
    'Cz': (2, 3),
    'C4': (3, 3),
    'T8': (4, 3),
    'CP3': (1.1, 2.5),
    'CP4': (2.9, 2.5),
    'P3': (1.25, 2),
    'Pz': (2, 2),
    'P4': (2.75, 2),
    'PO7': (0.9, 1.5),
    'Oz': (2, 1),
    'PO8': (3.1, 1.5),
}

xrange = (-1, 5)
yrange = (0, 5)
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD POSTERIOR
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file_chain],
    warmup=0,
    thin=1
)
results.add_transformed_variables()
importance = pd.read_csv(
    dir_results + f"K{subject}/K{subject}_importance.csv",
    index_col=0
)
# the order was reverese when computed
importance["drop_bce"] *= -1
importance["drop_acc"] *= -1
importance["drop_auroc"] *= -1

# channel order
# order = importance.sort_values("posterior", ascending=False)["component"].values
# order = importance.sort_values("drop_auroc", ascending=True)["component"].values
# order = importance.sort_values("drop_acc", ascending=True)["component"].values
order = importance.sort_values("drop_bce", ascending=True)["component"].values

beta_z1 = results.chains["smgp_factors.target_signal.rescaled"]
beta_z0 = results.chains["smgp_factors.nontarget_process.rescaled"]
beta_xi1 = results.chains["smgp_scaling.target_signal"]
beta_xi0 = results.chains["smgp_scaling.nontarget_process"]
diff = beta_z1 * beta_xi1.exp() - beta_z0 * beta_xi0.exp()
diff_xi = results.chains["smgp_scaling.difference_process"]

# fitted mean
L = results.chains["loadings.norm_one"]  # nc x ns x E x K
channel_wise_mean_diff = torch.einsum(
    "cbek, cbkt -> cbket",
    L,
    diff
)

# fitted covariance
cov1 = torch.einsum(
    "cbek, cbfk, cbkt -> cbeft",
    L,
    L,
    beta_xi1.exp().pow(2.)
)
cov0 = torch.einsum(
    "cbek, cbfk, cbkt -> cbeft",
    L,
    L,
    beta_xi0.exp().pow(2.)
)

channel_wise_cov_diff = (cov1 - cov0)

# fitted correlation
sd1 = cov1.diagonal(dim1=-2, dim2=-3).pow(0.5).permute(0, 1, 3, 2)
sd0 = cov0.diagonal(dim1=-2, dim2=-3).pow(0.5).permute(0, 1, 3, 2)
cor1 = cov1 / sd1.unsqueeze(-2) / sd1.unsqueeze(-3)
cor0 = cov0 / sd0.unsqueeze(-2) / sd0.unsqueeze(-3)
channel_wise_cor_diff = cor1 - cor0
# -----------------------------------------------------------------------------




# =============================================================================
# PLOT COMPONENTS
file = f"K{subject}_components"
fig, axes = plt.subplots(
    nrows=6+1,
    ncols=4,
    figsize=(12, 10),
    gridspec_kw={
        "width_ratios": [1, 1, 1, 1],
        "height_ratios": [1.5, 1, 1, 1.5, 1, 1, 0.5]
    },
    sharex="row",
    sharey="row"
)
plt.tight_layout()
for j in range(K):
    k = order[j]
    col = j % 4
    row = 3 * (j // 4)

    # network plot
    ax = axes[row, col]
    component = L[0, :, :, k].mean(0).reshape(-1, 1)
    component_sd = L[0, :, :, k].std(0).reshape(-1, 1)
    excludes = component.abs() / component_sd < 1.96
    colors = ["b" if c > 0 else "r" for c in component]
    # colors = [c if not e else "w" for c, e in zip(colors, excludes)]
    sizes = component.abs().pow(1.).cpu().numpy() * 250
    x = [channel_positions[c][0] for c in channels]
    y = [channel_positions[c][1] for c in channels]
    ax.scatter(x, y, c=colors, s=sizes)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    # remove box around
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # remove grid
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # difference plot
    row += 1
    ax = axes[row, col]
    t = np.arange(25)
    diffkmean = diff[:, :, k, :].mean((0, 1)).cpu().numpy()
    diffkstd = diff[:, :, k, :].std((0, 1)).cpu().numpy()
    ax.axhline(0, color="k", linestyle="--")
    ax.set_title(f"Component {k+1}\n"
                 # f"Effect size: {importance['posterior'][k]:.1f}\n"
                 f"BCE Change: {importance['drop_bce'][k]:.2f}")
                 # f"Accuracy Change: {importance['drop_acc'][k]*100:.1f}%")
    ax.fill_between(
        t,
        diffkmean - diffkstd,
        diffkmean + diffkstd,
        alpha=0.5
    )
    ax.plot(t, diffkmean)
    ax.set_xlim(0, 24)
    # ax.set_ylim(-20, 20)
    # ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels([])

    # difference plot
    row += 1
    ax = axes[row, col]
    t = np.arange(25)
    diffkmean = diff_xi[:, :, k, :].exp().mean((0, 1)).cpu().numpy()
    diffkstd = diff_xi[:, :, k, :].exp().std((0, 1)).cpu().numpy()
    ax.axhline(1, color="k", linestyle="--")
    ax.fill_between(
        t,
        diffkmean - diffkstd,
        diffkmean + diffkstd,
        alpha=0.5
    )
    ax.plot(t, diffkmean)
    ax.set_xlim(0, 24)
    # ax.set_ylim(0.8, 1.2)
    # ax.set_ylim(0.5, 1.5)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels([0, 200, 400, 600, 800])
    ax.set_xlabel("Time (ms)")
axes[1, 0].set_ylabel("Difference in mean")
axes[4, 0].set_ylabel("Difference in mean")
axes[2, 0].set_ylabel("Scaling difference")
axes[5, 0].set_ylabel("Scaling difference")

# legend
gs = axes[6, 0].get_gridspec()
for ax in axes[6, :]:
    ax.remove()
axlegend = fig.add_subplot(gs[6, :], frameon=False)
axlegend.axis("off")

values = torch.Tensor([-1., -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.]).cpu()
xs = torch.arange(-5, 6, 1).cpu()
colors = ["b" if c > 0 else "r" for c in values]
# colors = [c if not e else "w" for c, e in zip(colors, excludes)]
sizes = values.abs().pow(1.).cpu().numpy() * 250
y = [0 for c in values]
axlegend.scatter(xs, y, c=colors, s=sizes)
for i, v in enumerate(values):
    axlegend.text(xs[i], -0.8, f"{v:.1f}", ha="center", va="center")
# remove box around
axlegend.spines['top'].set_visible(False)
axlegend.spines['right'].set_visible(False)
axlegend.spines['left'].set_visible(False)
axlegend.spines['bottom'].set_visible(False)
# remove grid
axlegend.grid(False)
axlegend.set_xticks([])
axlegend.set_yticks([])
axlegend.set_xlim(-10, 10)
axlegend.set_ylim(-1, 1)



plt.savefig(dir_figures + file + ".pdf")
# -----------------------------------------------------------------------------





# =============================================================================
# PLOT MEAN DIFF OVER TIME
file = f"K{subject}_mean_over_time"
fig, axes = plt.subplots(
    nrows=6,
    ncols=4,
    figsize=(12, 18),
    sharex="all",
    sharey="all"
)
plt.tight_layout()
for t in range(24):
    col = t % 4
    row = t // 4
    i = t * 1
    time = round(t * 33.33)

    # network plot
    ax = axes[row, col]
    component = channel_wise_mean_diff[0, :, :, :, i].sum(1).mean(0).reshape(-1, 1)
    component_sd = channel_wise_mean_diff[0, :, :, :, i].sum(1).std(0).reshape(-1, 1)
    excludes = component.abs() / component_sd < 1.96
    colors = ["b" if c > 0 else "r" for c in component]
    # colors = [c if not e else "w" for c, e in zip(colors, excludes)]
    sizes = component.abs().pow(1.).cpu().numpy() * 25
    x = [channel_positions[c][0] for c in channels]
    y = [channel_positions[c][1] for c in channels]
    ax.scatter(x, y, c=colors, s=sizes)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    # remove box around
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # remove grid
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f"{time} ms")
plt.savefig(dir_figures + file + ".pdf")
# -----------------------------------------------------------------------------





# =============================================================================
# PLOT MEAN DIFF OVER TIME
file = f"K{subject}_cov_over_time"
fig, axes = plt.subplots(
    nrows=6,
    ncols=4,
    figsize=(12, 18),
    sharex="all",
    sharey="all"
)
plt.tight_layout()
for t in range(24):
    col = t % 4
    row = t // 4
    i = t * 1
    time = round(t * 33.33)

    # network plot
    ax = axes[row, col]
    network = channel_wise_cor_diff[0, :, :, :, i].mean(0)
    network_sd = channel_wise_cor_diff[0, :, :, :, i].std(0)
    exclude = network.abs() / network_sd < 1.96
    # set small entries to 0
    network[exclude] = 0

    edgelist = torch.tril_indices(16, 16, offset=-1)
    weights = network[edgelist[0], edgelist[1]]
    alphas = weights.abs().pow(1.).cpu().numpy().clip(0., 0.2)*5.
    G = nx.from_edgelist(edgelist.T.cpu().numpy(), create_using=nx.DiGraph)
    for i, (u, v) in enumerate(G.edges()):
        G.edges[u, v]['weight'] = weights[i].item()
    G = nx.relabel_nodes(G, {i: channels[i] for i in range(16)})
    G.edges(data=True)
    nx.draw_networkx_nodes(G, pos=channel_positions, node_size=100,
                           node_color='w', edgecolors='k', linewidths=1,
                           ax=ax)
    nx.draw_networkx_edges(G, pos=channel_positions, width=weights.abs().pow(1.).cpu().numpy()*50.,
                           edge_color=["b" if w > 0 else "r" for w in weights],
                           connectionstyle='arc3, rad=0.1', arrowsize=20, arrowstyle='-',
                           alpha=alphas,
                           ax=ax)
    # nx.draw_networkx_labels(G, pos=channel_positions, font_size=12, font_color='w', font_weight='bold',
    #                         ax=ax)


    ax.set_aspect('equal', 'box')
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    # remove box around
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # remove grid
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f"{time} ms")
plt.savefig(dir_figures + file + ".pdf")
# -----------------------------------------------------------------------------








# =============================================================================
# PLOT MEAN DIFF OVER TIME
file = f"K{subject}_mean_and_cov_over_time"
ts = [5, 6, 7, 8, 9, 10, 11, 12]

fig, axes = plt.subplots(
    nrows=len(ts) // 4,
    ncols=4,
    figsize=(12, 6),
    sharex="all",
    sharey="all"
)
plt.tight_layout()
for i, t in enumerate(ts):
    col = i % 4
    row = i // 4
    i = t * 1
    time = round(t * 33.33)

    # network plot
    ax = axes[row, col]
    network = channel_wise_cor_diff[0, :, :, :, t].mean(0)
    network_sd = channel_wise_cor_diff[0, :, :, :, i].std(0)
    exclude = network.abs() / network_sd < 1.96
    # set small entries to 0
    network[exclude] = 0


    component = channel_wise_mean_diff[0, :, :, :, i].sum(1).mean(0).reshape(-1, 1)
    colors = ["b" if c > 0 else "r" for c in component]
    sizes = component.abs().pow(1.).cpu().numpy() * 25

    edgelist = torch.tril_indices(16, 16, offset=-1)
    weights = network[edgelist[0], edgelist[1]]
    alphas = weights.abs().pow(1.).cpu().numpy().clip(0., 0.2)*5.
    G = nx.from_edgelist(edgelist.T.cpu().numpy(), create_using=nx.DiGraph)
    for i, (u, v) in enumerate(G.edges()):
        G.edges[u, v]['weight'] = weights[i].item()
    G = nx.relabel_nodes(G, {i: channels[i] for i in range(16)})
    G.edges(data=True)
    nx.draw_networkx_nodes(G, pos=channel_positions, node_size=sizes,
                           node_color=colors, edgecolors='k', linewidths=0,
                           ax=ax)
    nx.draw_networkx_edges(G, pos=channel_positions, width=weights.abs().pow(1.).cpu().numpy()*50.,
                           edge_color=["b" if w > 0 else "r" for w in weights],
                           connectionstyle='arc3, rad=0.1', arrowsize=20, arrowstyle='-',
                           alpha=alphas,
                           ax=ax)
    # nx.draw_networkx_labels(G, pos=channel_positions, font_size=12, font_color='w', font_weight='bold',
    #                         ax=ax)


    ax.set_aspect('equal', 'box')
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    # remove box around
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # remove grid
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f"{time} ms")
plt.savefig(dir_figures + file + ".pdf")
# -----------------------------------------------------------------------------




