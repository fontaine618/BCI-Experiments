import sys
sys.path.insert(1, '/home/simfont/Documents/BCI/source')

import time
import torch
import pickle
from source.bffmbci.bffm import BFFModel
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SETUP

# 27h, 2GB

# parameters
seed = 2 #sys.argv[1]
# 0: ASRWMH
# 1: MALA
# 2: ESS Posterior
n_iter = 20_000

# paths
dir = "/home/simon/Documents/BCI/experiments/synthetic/"
dir_data = dir + "data/"
dir_chains = dir + "chains/"

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
# -----------------------------------------------------------------------------


# =============================================================================
# MODEL
model = BFFModel(
	sequences=sequence,
	stimulus_order=order,
	target_stimulus=target,
	**settings,
	**prior_parameters
)
# -----------------------------------------------------------------------------


# =============================================================================
# INITIALIZE CHAIN
torch.manual_seed(seed)
status = False
while not status:
	try:
		model.initialize_chain()
		status = True
	except Exception as e:
		print(e)
# -----------------------------------------------------------------------------



# =============================================================================
# RUN CHAIN
t0 = time.time()
t00 = t0
for i in range(n_iter):
	model.sample()
	if i % 1 == 0:
		print(f"{i:>10} "
			  f"{model.variables['observations'].log_density_history[-1]:>20.4f}"
			  f"  dt={time.time() - t00:>20.4f}   elapsed={time.time() - t0:20.4f}")
		t00 = time.time()
# -----------------------------------------------------------------------------



# =============================================================================
# SAVE CHAIN
# NB: use .chain to avoid VC-ing large objects
out = model.results()
with open(dir_chains + f"seed{seed}.chain", "wb") as f:
	pickle.dump(out, f)
# -----------------------------------------------------------------------------