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
samplers = {
    "mala": "MALA",
    "asm": "ASM",
    "ess": "ESS (post)"
}

# paths
dir = "/home/simon/Documents/BCI/experiments/real_data1/"
dir_chains = dir + "chains/K114_001_BCI_TRN/"
dir_figures = dir + "figures_samplers/"
# -----------------------------------------------------------------------------


# =============================================================================
results = BFFMResults.from_files(
	[dir_chains + f"seed0_{sampler}.chain" for sampler in samplers],
	warmup=0,
	thin=10
)
# -----------------------------------------------------------------------------



# =============================================================================
# Plot LLK
fig, ax = plt.subplots()
df = pd.DataFrame(results.chains["log_likelihood.observations"].cpu().T)
df.columns = [samplers[k] for k in samplers]
sns.lineplot(
	data=df,
	alpha=0.5
)
ax.set_ylim(-800_000, -735_000)
ax.set_xticks(np.arange(0, 5001, 500), np.arange(0, 50001, 5000))
ax.set_title("Exponential link samplers")
ax.set_xlabel("Iteration")
ax.set_ylabel("Log-likelihood")
# ax.set_xscale("log")
fig.savefig(f"{dir_figures}observation_log_likelihood.pdf")
# -----------------------------------------------------------------------------




# =============================================================================
results = BFFMResults.from_files(
	[dir_chains + f"seed0_{sampler}.chain" for sampler in samplers],
	warmup=48_000,
	thin=1
)
# -----------------------------------------------------------------------------



# =============================================================================
# Plot LLK
fig, ax = plt.subplots()
df = pd.DataFrame(results.chains["log_likelihood.observations"].cpu().T)
df.columns = [samplers[k] for k in samplers]
sns.lineplot(
	data=df,
	alpha=0.5
)
ax.set_ylim(-745_000, -735_000)
ax.set_xticks(np.arange(0, 2001, 500), np.arange(48000, 50001, 500))
ax.set_title("Exponential link samplers")
ax.set_xlabel("Iteration")
ax.set_ylabel("Log-likelihood")
# ax.set_xscale("log")
fig.savefig(f"{dir_figures}observation_log_likelihood_last2K.pdf")
# -----------------------------------------------------------------------------
