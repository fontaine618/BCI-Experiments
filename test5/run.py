import sys
sys.path.insert(1, '/home/simfont/Documents/BCI/src')

import time
import torch
import pickle
from src.bffmbci.bffm import BFFModel
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SETUP

# parameters
seed = 0
training_repetitions = int(sys.argv[1])
n_iter = 50_000

# paths
dir = "/home/simfont/Documents/BCI/experiments/test5/"
dir_data = dir + "data/train/"
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

# FILTER FIRST REPS
n_characters = settings["n_characters"]
n_repetitions = settings["n_repetitions"]
sequence_ids = torch.hstack([
	torch.arange(i, sequence.shape[0], n_repetitions)
	for i in range(training_repetitions)
])
model.filter(sequence_ids)
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
	if i % 1000 == 0:
		print(f"{i:>10} "
			  f"{model.variables['observations'].log_density_history[-1]:>20.4f}"
			  f"  dt={time.time() - t00:>20.4f}   elapsed={time.time() - t0:20.4f}")
		t00 = time.time()
# -----------------------------------------------------------------------------



# =============================================================================
# SAVE CHAIN
# NB: use .chain to avoid VC-ing large objects
out = model.results()
with open(dir_chains + f"nrep{training_repetitions:02}.chain", "wb") as f:
	pickle.dump(out, f)
# -----------------------------------------------------------------------------