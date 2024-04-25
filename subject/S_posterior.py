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
    'F3': (1.25, 4.05),
    'Fz': (2, 4),
    'F4': (2.75, 4.05),
    'T7': (0.1, 3),
    'C3': (1.05, 3),
    'Cz': (2, 3),
    'C4': (2.95, 3),
    'T8': (3.9, 3),
    'CP3': (1.1, 2.45),
    'CP4': (2.9, 2.45),
    'P3': (1.25, 1.95),
    'Pz': (2, 2),
    'P4': (2.75, 1.95),
    'PO7': (1., 1.3),
    'Oz': (2, 0.95),
    'PO8': (3., 1.3),
}

extra_channels = {
    "F1": (1.625, 4),
    "F2": (2.375, 4),
    "F5": (0.875, 4.125),
    "F6": (3.125, 4.125),
    "F7": (0.5, 4.25),
    "F8": (3.5, 4.25),
    "F9": (0.125, 4.5),
    "F10": (3.875, 4.5),

    "FT9": (-0.25, 3.875),
    "FT7": (0.2, 3.75),
    "FC5": (0.65, 3.625),
    "FC3": (1.1, 3.55),
    "FC1": (1.5, 3.5),
    "FCz": (2, 3.5),
    "FC2": (2.5, 3.5),
    "FC4": (2.9, 3.55),
    "FC6": (3.35, 3.625),
    "FT8": (3.8, 3.75),
    "FT10": (4.25, 3.875),

    "C5": (0.6, 3.),
    "C1": (1.5, 3.),
    "C2": (2.5, 3.),
    "C6": (3.4, 3.),

    "TP9": (-0.25, 2.125),
    "TP7": (0.2, 2.25),
    "CP5": (0.65, 2.375),
    "CP1": (1.5, 2.5),
    "CPz": (2, 2.5),
    "CP2": (2.5, 2.5),
    "CP6": (3.35, 2.375),
    "TP8": (3.8, 2.25),
    "TP10": (4.25, 2.125),

    "P9": (0.125, 1.5),
    "P7": (0.5, 1.75),
    "P5": (0.875, 1.875),
    "P1": (1.625, 2.),
    "P2": (2.375, 2.),
    "P6": (3.125, 1.875),
    "P8": (3.5, 1.75),
    "P10": (3.875, 1.5),

    "PO9": (0.625, 1.),
    "PO3": (1.5, 1.42),
    "POz": (2., 1.45),
    "PO4": (2.5, 1.42),
    "PO10": (3.375, 1.),

    "O1": (1.5, 1),
    "O2": (2.5, 1),

    "O9": (1.25, 0.7),
    "O10": (2.75, 0.7),
    "Iz": (2., 0.55),

    "AF3": (1.5, 4.45),
    'AFz': (2, 4.5),
    "AF4": (2.5, 4.45),

    "AF7": (0.9, 4.6),
    "Fp1": (1.4, 4.9),
    "Fpz": (2., 5.),
    "Fp2": (2.6, 4.9),
    "AF8": (3.1, 4.6),
}

xrange = (-0.5, 4.5)
yrange = (0.5, 4.5)
# -----------------------------------------------------------------------------


# =============================================================================
# PLOT NODES
file = f"K{subject}_nodes"
fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(5, 4.8),
    sharex="all",
    sharey="all"
)



# draw ears
ear1 = plt.Circle((-0.2, 3.), 0.5, edgecolor='black', fill=True, facecolor="lightgrey")
ear2 = plt.Circle((4.2, 3.), 0.5, edgecolor='black', fill=True, facecolor="lightgrey")
ax.add_artist(ear1)
ax.add_artist(ear2)

# draw nose as triangle
nose = plt.Polygon([[1.5, 5.], [2.5, 5.], [2., 5.75]], edgecolor='black', fill=True, facecolor="lightgrey")
ax.add_artist(nose)

# draw scalp
circle = plt.Circle((2, 3), 2.5, edgecolor='black', fill=True, facecolor="lightgrey")
ax.add_artist(circle)

for cname, (x, y) in channel_positions.items():
    ax.plot(x, y, "o", markersize=20, color="black", fillstyle='full', markerfacecolor="black")
    ax.text(x, y, cname, ha="center", va="center", color="white", size=8, fontweight="bold")

for cname, (x, y) in extra_channels.items():
    ax.plot(x, y, "o", markersize=20, color="black", fillstyle='full', markerfacecolor="white")
    ax.text(x, y, cname, ha="center", va="center", size=8, color="grey")

ax.set_aspect('equal', 'box')
ax.set_xlim(-0.75, 4.75)
ax.set_ylim(0., 6.)
# remove box around
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# remove grid
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(dir_figures + file + ".pdf")
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
diff_xi_scaled = (beta_xi1.exp().pow(2.) - beta_xi0.exp().pow(2.)) * results.chains["loadings.norm"].movedim(2, 3).pow(2.)

# fitted mean
L = results.chains["loadings.norm_one"]  # nc x ns x E x K
Lnorm = results.chains["loadings.norm"] # nc x ns x K
channel_wise_mean_diff = torch.einsum(
    "cbek, cbkt -> cbket",
    L,
    diff
)

# fitted covariance
L = results.chains["loadings"]
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

# fitted autocovariance
cor, var, power = results.prior["kernel_gp_factor"]

channel_wise_autocorr_diff = torch.zeros(1, 1000, 16, 16, 25, 3)
for lag in range(0, 3):
    kvalue = var * cor ** (lag * lag)
    beta_xi1_shifted = beta_xi1.roll(shifts=lag, dims=-1)
    beta_xi0_shifted = beta_xi0.roll(shifts=lag, dims=-1)
    cov1_shifted = torch.einsum(
        "cbek, cbfk, cbkt -> cbeft",
        L,
        L,
        beta_xi1.exp() * beta_xi1_shifted.exp()
    ) * kvalue
    cov0_shifted = torch.einsum(
        "cbek, cbfk, cbkt -> cbeft",
        L,
        L,
        beta_xi0.exp() * beta_xi0_shifted.exp()
    ) * kvalue
    sd1_shifted = sd1.roll(shifts=lag, dims=-1)
    sd0_shifted = sd0.roll(shifts=lag, dims=-1)
    cor1_shifted = cov1_shifted / sd1.unsqueeze(-2) / sd1_shifted.unsqueeze(-3)
    cor0_shifted = cov0_shifted / sd0.unsqueeze(-2) / sd0_shifted.unsqueeze(-3)
    channel_wise_autocorr_diff[0, :, :, :, :, lag] = cor1_shifted - cor0_shifted

# -----------------------------------------------------------------------------







# =============================================================================
# PLOT COMPONENTS
L = results.chains["loadings.norm_one"]
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
    # ax.text(0, 4, f"k={k+1}", ha="center", va="center", fontsize=14)
    # remove box around
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # remove grid
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # mean difference plot
    row += 1
    ax = axes[row, col]
    t = np.arange(25)
    diffkmean = diff[:, :, k, :].mean((0, 1)).cpu().numpy()
    diffkstd = diff[:, :, k, :].std((0, 1)).cpu().numpy()
    ax.axhline(0, color="k", linestyle="--")
    Lnormk = Lnorm[:, :, 0, k].mean((0, 1)).item()
    ax.set_title(f"Component {k+1}\n"
                 f"Loading norm: {Lnormk:.2f}\n"
                 f"BCE Change: {importance['drop_bce'][k]:.2f}")
    ax.fill_between(
        t,
        diffkmean - diffkstd,
        diffkmean + diffkstd,
        alpha=0.5
    )
    ax.plot(t, diffkmean)
    ax.set_xlim(0, 24)
    ax.set_ylim(-7, 7)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels([])

    # scaling difference plot
    row += 1
    ax = axes[row, col]
    t = np.arange(25)
    # diffkmean = diff_xi[:, :, k, :].exp().mean((0, 1)).cpu().numpy()
    # diffkstd = diff_xi[:, :, k, :].exp().std((0, 1)).cpu().numpy()
    diffkmean = diff_xi_scaled[:, :, k, :].mean((0, 1)).cpu().numpy()
    diffkstd = diff_xi_scaled[:, :, k, :].std((0, 1)).cpu().numpy()
    ax.axhline(1, color="k", linestyle="--")
    ax.fill_between(
        t,
        diffkmean - diffkstd,
        diffkmean + diffkstd,
        alpha=0.5
    )
    ax.plot(t, diffkmean)
    ax.set_xlim(0, 24)
    ax.set_ylim(-90, 90)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels([0, 200, 400, 600, 800])
    ax.set_xlabel("Time (ms)")
axes[1, 0].set_ylabel("Diff. in mean")
axes[4, 0].set_ylabel("Diff. in mean")
axes[2, 0].set_ylabel("Diff. in covariance")
axes[5, 0].set_ylabel("Diff. in covariance")

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


axlegend.text(-6., 0., "Std. loading", ha="right", va="center")
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



plt.tight_layout()
plt.savefig(dir_figures + file + ".pdf")
# -----------------------------------------------------------------------------






# =============================================================================
# PLOT AUTOCORR DIFF
file = f"K{subject}_autocorr_diff"
ts = range(2, 15)
lags = [0, 1, 2]
ncols = len(lags)
nrows = len(ts)
fig, axes = plt.subplots(
    nrows=nrows + 1,
    ncols=ncols,
    figsize=(ncols * 2.5, nrows * 2.5),
    gridspec_kw={
        "width_ratios": [1] * ncols,
        "height_ratios": [1] * nrows + [0.5]
    },
    sharex="all",
    sharey="all"
)
for row, t in enumerate(ts):
    for col, lag in enumerate(lags):
        i = t * 1
        time = round(t * 33.33)
        time_other = round((t - lag) * 33.33)

        # network plot
        ax = axes[row, col]
        network = channel_wise_autocorr_diff[0, :, :, :, t, lag].mean(0)
        network_sd = channel_wise_autocorr_diff[0, :, :, :, t, lag].std(0)
        exclude = network.abs() / network_sd < 1.96
        # set small entries to 0
        network[exclude] = 0

        component = channel_wise_autocorr_diff[0, :, :, :, t, lag].sum(1).mean(0).reshape(-1, 1)
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

        ax.set_title(f"{time_other} ms")
        if col == 0:
            ax.set_ylabel(f"{time} ms")

# legend
gs = axes[nrows, 0].get_gridspec()
for ax in axes[nrows, :]:
    ax.remove()
axlegend = fig.add_subplot(gs[nrows, :], frameon=False)
axlegend.axis("off")

# values = torch.Tensor([-10., -5., -2., -1., -0.5, 0., 0.5, 1., 2., 5., 10.]).cpu()
# xs = torch.arange(-5, 6, 1).cpu()
# colors = ["b" if c > 0 else "r" for c in values]
# # colors = [c if not e else "w" for c, e in zip(colors, excludes)]
# sizes = values.abs().pow(1.).cpu().numpy() * 25
# y = [0 for c in values]
# axlegend.scatter(xs, y, c=colors, s=sizes)
# for i, v in enumerate(values):
#     axlegend.text(xs[i], -0.5, f"{v:.1f}", ha="center", va="center")

values = torch.Tensor([-0.2, -0.1, -0.05, -0.02, -0.01, 0., 0.01, 0.02, 0.05, 0.1, 0.2]).cpu()
xs = torch.arange(-5, 6, 1).cpu()
y = [1. for c in values]
colors = ["b" if c > 0 else "r" for c in values]
width = values.abs().pow(1.).cpu().numpy() * 50.
alphas = values.abs().pow(1.).cpu().numpy().clip(0., 0.2) * 5.
for i, (x, y, c, w, a, v) in enumerate(zip(xs, y, colors, width, alphas, values)):
    axlegend.plot([x - 0.2, x + 0.2], [y, y], c=c, lw=w, alpha=a)
    axlegend.text(x, 0.5, f"{v:.2f}", ha="center", va="center")

# axlegend.text(-6., 0., "Difference in mean", ha="right", va="center")
axlegend.text(-6., 1., "Difference in correlation", ha="right", va="center")
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
axlegend.set_ylim(-1, 1.5)

plt.tight_layout()
plt.savefig(dir_figures + file + ".pdf")
# -----------------------------------------------------------------------------



# =============================================================================
# PLOT MEAN AND CORRELATION DIFF
file = f"K{subject}_mean_and_cov_over_time"
ts = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
ncols = 5
nrows = len(ts) // ncols
fig, axes = plt.subplots(
    nrows=nrows + 1,
    ncols=ncols,
    figsize=(12, nrows* 2.5 + 1),
    gridspec_kw={
        "width_ratios": [1] * ncols,
        "height_ratios": [1] * nrows + [0.5]
    },
    sharex="all",
    sharey="all"
)
for i, t in enumerate(ts):
    col = i % ncols
    row = i // ncols
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


# legend
gs = axes[nrows, 0].get_gridspec()
for ax in axes[nrows, :]:
    ax.remove()
axlegend = fig.add_subplot(gs[nrows, :], frameon=False)
axlegend.axis("off")

values = torch.Tensor([-10., -5., -2., -1., -0.5, 0., 0.5, 1., 2., 5., 10.]).cpu()
xs = torch.arange(-5, 6, 1).cpu()
colors = ["b" if c > 0 else "r" for c in values]
# colors = [c if not e else "w" for c, e in zip(colors, excludes)]
sizes = values.abs().pow(1.).cpu().numpy() * 25
y = [0 for c in values]
axlegend.scatter(xs, y, c=colors, s=sizes)
for i, v in enumerate(values):
    axlegend.text(xs[i], -0.5, f"{v:.1f}", ha="center", va="center")

values = torch.Tensor([-0.2, -0.1, -0.05, -0.02, -0.01, 0., 0.01, 0.02, 0.05, 0.1, 0.2]).cpu()
xs = torch.arange(-5, 6, 1).cpu()
y = [1. for c in values]
colors = ["b" if c > 0 else "r" for c in values]
width = values.abs().pow(1.).cpu().numpy()*50.
alphas = values.abs().pow(1.).cpu().numpy().clip(0., 0.2) * 5.
for i, (x, y, c, w, a, v) in enumerate(zip(xs, y, colors, width, alphas, values)):
    axlegend.plot([x-0.2, x+0.2], [y, y], c=c, lw=w, alpha=a)
    axlegend.text(x, 0.5, f"{v:.2f}", ha="center", va="center")

axlegend.text(-6., 0., "Difference in mean", ha="right", va="center")
axlegend.text(-6., 1., "Difference in correlation", ha="right", va="center")
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
axlegend.set_ylim(-1, 1.5)

plt.tight_layout()
plt.savefig(dir_figures + file + ".pdf")
# -----------------------------------------------------------------------------




