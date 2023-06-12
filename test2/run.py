import torch
import pickle
from source.models.bffmbci.bffm import BFFModel
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SETUP

# parameters
seed = 0
n_iter = 200_000

# paths
dir = "/home/simon/Documents/BCI/experiments/test2/"
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
for i in range(n_iter):
	model.sample()
	print(seed, i, model.variables["observations"].log_density_history[-1])
# -----------------------------------------------------------------------------



# =============================================================================
# SAVE CHAIN
# NB: use .chain to avoid VC-ing large objects
model.results().save(dir_chains + f"seed{seed}.chain")
# -----------------------------------------------------------------------------