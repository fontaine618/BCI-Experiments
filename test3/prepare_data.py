import torch
import pickle
from src.bffmbci.bffm import BFFModel
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# PARAMETERS

# data generation
latent_dim = 2
n_channels = 7
stimulus_to_stimulus_interval = 10
stimulus_window = 40
n_stimulus = (3, 3)
n_sequences = 200
nonnegative_smgp = True
heterogeneities = 3.
shrinkage_factor = (2., 10.)
seed = 0

# paths
dir = "/home/simon/Documents/BCI/experiments/test3/"
dir_data = dir + "data/"
# -----------------------------------------------------------------------------


# =============================================================================
# DATA GENERATION
torch.manual_seed(seed)
model = BFFModel.generate_from_dimensions(
	latent_dim=latent_dim,
	n_channels=n_channels,
	stimulus_to_stimulus_interval=stimulus_to_stimulus_interval,
	stimulus_window=stimulus_window,
	n_stimulus=n_stimulus,
	n_sequences=n_sequences,
	nonnegative_smgp=nonnegative_smgp,
	heterogeneities=heterogeneities,
	shrinkage_factor=shrinkage_factor
)
true_values = model.current_values()
true_llk = model.variables["observations"].log_density
true_values["observation_log_likelihood"] = true_llk

# save generating values
with open(dir_data + "true_values.pkl", "wb") as f:
	pickle.dump(true_values, f)

# save data
with open(dir_data + "order.pkl", "wb") as f:
	pickle.dump(model.variables["sequence_data"].order.data, f)
with open(dir_data + "target.pkl", "wb") as f:
	pickle.dump(model.variables["sequence_data"].target.data, f)
with open(dir_data + "sequence.pkl", "wb") as f:
	pickle.dump(model.variables["observations"].data, f)

# save settings
settings = {
	"latent_dim": latent_dim,
	"n_channels": n_channels,
	"stimulus_to_stimulus_interval": stimulus_to_stimulus_interval,
	"stimulus_window": stimulus_window,
	"n_stimulus": n_stimulus,
	"n_sequences": n_sequences,
	"nonnegative_smgp": nonnegative_smgp,
	"seed": seed
}
with open(dir_data + "settings.pkl", "wb") as f:
	pickle.dump(settings, f)

prior_parameters = model.prior_parameters
with open(dir_data + "prior_parameters.pkl", "wb") as f:
	pickle.dump(prior_parameters, f)
# -----------------------------------------------------------------------------