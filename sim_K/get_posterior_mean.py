import sys
import pickle
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.bffmbci import BFFMResults

# =============================================================================
# SETUP
dir_data = "/home/simon/Documents/BCI/experiments/sim_K/data/"
file = f"K114_dim10.chain"
file_out = f"K114_dim10.postmean"
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD CHAIN
results = BFFMResults.from_files([dir_data + file], warmup=0, thin=1)
# -----------------------------------------------------------------------------


# =============================================================================
# GET POSTERIOR MEAN of all variables
variables = {
    k: v.mean(dim=(0, 1)).cpu().numpy()
    for k, v in results.chains.items()
}
# -----------------------------------------------------------------------------


# =============================================================================
# SAVE
pickle.dump(variables, open(dir_data + file_out, "wb"))
# -----------------------------------------------------------------------------