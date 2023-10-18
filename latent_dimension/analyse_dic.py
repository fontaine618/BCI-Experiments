import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from source.bffmbci import BFFMResults
from source.data.k_protocol import KProtocol
from source.bffmbci.bffm import BFFModel

plt.style.use('seaborn-whitegrid')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_figures = "/home/simon/Documents/BCI/experiments/latent_dimension/figures/"
dir_results = "/home/simon/Documents/BCI/experiments/latent_dimension/results/"
dir_chains = "/home/simon/Documents/BCI/experiments/latent_dimension/chains/"
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
os.makedirs(dir_figures, exist_ok=True)
os.makedirs(dir_results, exist_ok=True)

Ks = list(range(2, 13))
subject = "114"

out = pd.read_csv(dir_results + f"K{subject}_llk.csv")
out.rename(columns={"Unnamed: 0": "K"}, inplace=True)
# -----------------------------------------------------------------------------



# =============================================================================
# COMPUTE DICs
out["p_mllk"] = -2 * (out["mean_mllk"] - out["mllk_postmean"])
out["p_llk"] = -2 * (out["mean_llk"] - out["llk_postmean"])
out["dic_mllk"] = -2 * out["mean_mllk"] + out["p_mllk"]
out["dic_llk"] = -2 * out["mean_llk"] + out["p_llk"]
