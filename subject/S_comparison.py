import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
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
dir_figures = f"/home/simon/Documents/BCI/experiments/subject/figures/K_Comparison/"
os.makedirs(dir_figures, exist_ok=True)
K = 8

# colors
TARGET = "#FFCB05"
NONTARGET = "#00274C"
POSITIVE = "#00A398"
NEGATIVE = "#EF4135"


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

xrange = (-0.5, 4.5)
yrange = (0.5, 4.5)
# -----------------------------------------------------------------------------




# =============================================================================
# PLOTS
PLOTS = {
    "N2": (
        {
            114: ("Control", [6, 7], 0, [0, 1]),
            117: ("Control", [2, 6], 0, [2, 3]),
            171: ("Control", [3, ], 2, [0, ]),
            183: ("Control", [2, ], 2, [2]),
            146: ("ALS", [8, 6], 4, [0, 1]),
            151: ("ALS", [8, 6], 4, [2, 3]),
        },
        (-15, 10), # yrange
        [(2, 1), (2, 3)] # missing
    ),
    "P3": (
        {
            114: ("Control", [1, ], 0, [0, ]),
            117: ("Control", [1, 2], 0, [2, 3]),
            171: ("Control", [1, 2], 2, [0, 1]),
            183: ("Control", [1, ], 2, [2]),
            146: ("ALS", [1, ], 4, [0, ]),
            151: ("ALS", [1, ], 4, [2, ]),
        },
        (-15, 25), # yrange
        [(2, 3), (0, 1), (4, 1), (4, 3)] # missing
    ),
    "LR": (
        {
            114: ("Control", [3, ], 0, [0, ]),
            117: ("Control", [4, ], 0, [2, ]),
            171: ("Control", [7, 4], 2, [0, 1]),
            183: ("Control", [3, ], 2, [2]),
            146: ("ALS", [5, ], 4, [0, ]),
            151: ("ALS", [4, ], 4, [2, ]),
        },
        (-3, 7), # yrange
        [(0, 1), (0, 3), (2, 3), (4, 1), (4, 3)] # missing
    ),
    "P4": (
        {
            114: ("Control", [2, ], 0, [0, ]),
            117: ("Control", [2, 5], 0, [2, 3]),
            171: ("Control", [2, ], 2, [0, ]),
            183: ("Control", [2, ], 2, [2]),
            146: ("ALS", [2, 3], 4, [0, 1]),
            151: ("ALS", [3, ], 4, [2, ]),
        },
        (-5, 10), # yrange
        [(0, 1), (2, 1), (2, 3), (4, 3)] # missing
    )
}
# -----------------------------------------------------------------------------




# =============================================================================
def confidance_band(
        samples: torch.Tensor,
        level: float = 0.95,
):
    X = samples.reshape(-1, samples.shape[-1])
    m = X.mean(0)
    alpha = 1. - level
    dim = X.shape[-1]
    qs = reversed(torch.linspace(alpha/(2*dim), alpha/2, 500))
    for q in qs:
        Xmin = X.quantile(q, 0)
        Xmax = X.quantile(1. - q, 0)
        prop = ((Xmin.reshape(1, -1) <= X) & (X <= Xmax.reshape(1, -1))).all(1).float().mean()
        if prop >= level:
            break
    return m, Xmin, Xmax


def draw_head(ax, facecolor="lightgrey"):
    # draw ears
    ear1 = plt.Circle((-0.2, 3.), 0.5, edgecolor='black', fill=True, facecolor=facecolor)
    ear2 = plt.Circle((4.2, 3.), 0.5, edgecolor='black', fill=True, facecolor=facecolor)
    ax.add_artist(ear1)
    ax.add_artist(ear2)
    # draw nose as triangle
    nose = plt.Polygon([[1.5, 5.], [2.5, 5.], [2., 5.75]], edgecolor='black', fill=True, facecolor=facecolor)
    ax.add_artist(nose)
    # draw scalp
    circle = plt.Circle((2, 3), 2.5, edgecolor='black', fill=True, facecolor=facecolor)
    ax.add_artist(circle)


def blank_canvas(ax):
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
# -----------------------------------------------------------------------------


# =============================================================================
# MAKE PLOTS

for plot_name, (subjects, yyrange, missing) in PLOTS.items():
    fig, axes = plt.subplots(
        nrows=6 + 1,
        ncols=4,
        figsize=(10, 12),
        gridspec_kw={
            "width_ratios": [1, 1, 1, 1],
            "height_ratios": [1.5, 1, 1.5, 1, 1.5, 1, 0.5]
        },
        sharex="row",
        sharey="row"
    )
    for subject, (group, components, row, columns) in subjects.items():
        # LOAD POSTERIOR
        subject = str(subject)
        file_chain = f"K{subject}/K{subject}.chain"
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
        importance["drop_bce"] *= -1
        importance["drop_acc"] *= -1
        importance["drop_auroc"] *= -1
        beta_z1 = results.chains["smgp_factors.target_signal.rescaled"]
        beta_z0 = results.chains["smgp_factors.nontarget_process.rescaled"]
        beta_xi1 = results.chains["smgp_scaling.target_signal"]
        beta_xi0 = results.chains["smgp_scaling.nontarget_process"]
        mean0 = beta_z0 * beta_xi0.exp()
        mean1 = beta_z1 * beta_xi1.exp()
        mean_diff = mean1 - mean0
        L = results.chains["loadings.norm_one"]
        Lnorm = results.chains["loadings.norm"]
        # PLOT ALL COMPONENTS
        for i, (k, col) in enumerate(zip(components, columns)):
            k = k - 1 # to 0-index
            Lnormk = Lnorm[:, :, 0, k].mean((0, 1)).item()
            # NETWORK PLOT
            ax = axes[row, col]
            draw_head(ax, "white")
            component = L[0, :, :, k].mean(0).reshape(-1, 1)
            component_sd = L[0, :, :, k].std(0).reshape(-1, 1)
            excludes = component.abs() / component_sd < 1.96
            colors = [POSITIVE if c > 0 else NEGATIVE for c in component]
            sizes = component.abs().pow(1.).cpu().numpy() * 250
            x = [channel_positions[c][0] for c in channels]
            y = [channel_positions[c][1] for c in channels]
            ax.scatter(x, y, c=colors, s=sizes)
            ax.set_aspect('equal', 'box')
            blank_canvas(ax)
            if i == 0:
                ax.set_title(f"Subject {subject} ({group})", x=1.2, fontweight="bold")

            # PLOT MEAN PROCESSES
            ax = axes[row + 1, col]
            ax.set_title(f"Component {k+1}\n"
                         f"Loading norm: {Lnormk:.2f}\n"
                         f"BCE Change: {importance['drop_bce'][k]:.2f}",
                         size=10.)
            t = np.arange(25) * 31.25
            nontarget_mean, nontarget_min, nontarget_max = confidance_band(mean0[:, :, k, :])
            target_mean, target_min, target_max = confidance_band(mean1[:, :, k, :])

            # highlet where differential
            _, diff_min, diff_max = confidance_band(mean_diff[:, :, k, :], 0.95)
            differential = diff_min.gt(0.) | diff_max.lt(0.)
            for x, d in zip(t, differential):
                if d:
                    ax.axvline(x, color="black", linestyle="-", alpha=0.1, linewidth=6.25)

            # curves
            ax.axhline(0, color="k", linestyle="--")
            ax.fill_between(
                t,
                nontarget_min.cpu().numpy(),
                nontarget_max.cpu().numpy(),
                alpha=0.5,
                color=NONTARGET
            )
            ax.plot(t, nontarget_mean.cpu().numpy(), color=NONTARGET)
            ax.fill_between(
                t,
                target_min.cpu().numpy(),
                target_max.cpu().numpy(),
                alpha=0.5,
                color=TARGET
            )
            ax.plot(t, target_mean.cpu().numpy(), color=TARGET)
            ax.set_xlim(0, 24*31.25)
            ax.set_ylim(*yyrange)
            ax.set_xticks([0, 150, 300, 450, 600, 750])

            ax.set_xlabel("Time (ms)")
    for r, c in missing:
        axes[r, c].axis("off")
        axes[r + 1, c].axis("off")
    axes[1, 0].set_ylabel("Mean process")
    axes[3, 0].set_ylabel("Mean process")
    axes[5, 0].set_ylabel("Mean process")

    # ADD LEGEND
    gs = axes[6, 0].get_gridspec()
    for ax in axes[6, :]:
        ax.remove()
    axlegend = fig.add_subplot(gs[6, :], frameon=False)
    axlegend.axis("off")

    # loadings legend
    values = torch.Tensor([-1., -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.]).cpu()
    xs = torch.arange(-5, 6, 1).cpu()
    colors = [POSITIVE if c > 0 else NEGATIVE for c in values]
    sizes = values.abs().pow(1.).cpu().numpy() * 250
    y = [0 for c in values]
    axlegend.scatter(xs, y, c=colors, s=sizes)
    for i, v in enumerate(values):
        axlegend.text(xs[i], -0.8, f"{v:.2f}", ha="center", va="center")

    axlegend.text(-6., 0., "Std. loading", ha="right", va="center")

    # curve legends
    xs = torch.Tensor([7, 8]).cpu()
    for y, which, color, draw_line, alpha in zip(
            [0.8, -0.8, 0.],
            ["Nontarget", "Differential", "Target"],
            [NONTARGET, "black", TARGET],
            [True, False, True],
            [0.5, 0.1, 0.5]
    ):
        if draw_line:
            axlegend.plot(xs, [y, y], color=color)
        axlegend.fill_between(xs, [y - 0.2, y - 0.2], [y + 0.2, y + 0.2], color=color, alpha=alpha)
        axlegend.text(8.5, y, which, va="center")



    # remove box around
    axlegend.spines['top'].set_visible(False)
    axlegend.spines['right'].set_visible(False)
    axlegend.spines['left'].set_visible(False)
    axlegend.spines['bottom'].set_visible(False)
    # remove grid
    axlegend.grid(False)
    axlegend.set_xticks([])
    axlegend.set_yticks([])
    axlegend.set_xlim(-10, 13)
    axlegend.set_ylim(-1, 1)

    # SAVE
    plt.tight_layout()
    plt.savefig(dir_figures + plot_name + ".pdf")
# -----------------------------------------------------------------------------
