import torch
import pickle
from src.bffmbci.bffm import BFFModel
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import matplotlib.pyplot as plt
import copy

# =============================================================================
# PARAMETERS

# data generation
latent_dim = 3
n_channels = 15
stimulus_to_stimulus_interval = 5
stimulus_window = 20
n_stimulus = (6, 6)
n_characters_training = 20
n_characters_testing = 100
n_characters = n_characters_training + n_characters_testing
n_repetitions = 15
n_sequences = n_repetitions * n_characters
nonnegative_smgp = True
heterogeneities = 3.
shrinkage_factor = (2., 10.)
seed = 0

# paths
dir = "/home/simon/Documents/BCI/experiments/test5/"
dir_data = dir + "data/"
# -----------------------------------------------------------------------------


# =============================================================================
# DATA GENERATION

# start from a dummy model
torch.manual_seed(seed)
model = BFFModel.generate_from_dimensions(
	latent_dim=latent_dim,
	n_channels=n_channels,
	stimulus_to_stimulus_interval=stimulus_to_stimulus_interval,
	stimulus_window=stimulus_window,
	n_stimulus=n_stimulus,
	n_characters=n_characters,
	n_repetitions=n_repetitions,
	nonnegative_smgp=nonnegative_smgp,
	heterogeneities=heterogeneities,
	shrinkage_factor=shrinkage_factor
)

# replace values

# smgp_scaling
t = torch.arange(0, stimulus_window)
smgp_scaling_target = torch.vstack([
	torch.sin(2 * torch.pi * (t - 2*i) / 12)
	for i in range(latent_dim)
])
smgp_scaling_target = smgp_scaling_target * \
						 (torch.sigmoid((t-1)*2)*torch.sigmoid((stimulus_window-t-10)/3)).reshape(1, -1)
smgp_scaling_target = 1 + smgp_scaling_target * 0.5

# plt.plot(smgp_scaling_target.T.cpu().numpy())
# plt.show()

smgp_scaling_nontarget = torch.vstack([
	-torch.sin(2 * torch.pi * (t - 2*i) / 12)
	for i in range(latent_dim)
])
smgp_scaling_nontarget = 1 + smgp_scaling_nontarget * 0.2

# plt.plot(smgp_scaling_nontarget.T.cpu().numpy())
# plt.show()

smgp_scaling_mixing = torch.vstack([
	(torch.sigmoid((t-1-2*i))*torch.sigmoid((stimulus_window-t-10+2*i))).reshape(1, -1)
	for i in range(latent_dim)
])

# plt.plot(smgp_scaling_mixing.T.cpu().numpy())
# plt.show()


# smgp_factor
t = torch.arange(0, stimulus_window)
smgp_factor_target = torch.vstack([
	torch.sin(2 * torch.pi * (t - 2*i) / 12)
	for i in range(latent_dim)
])
smgp_factor_target = smgp_factor_target * \
						 (torch.sigmoid((t-1)*2)*torch.sigmoid((stimulus_window-t-10)/3)).reshape(1, -1)
smgp_factor_target = smgp_factor_target * 0.5

# plt.plot(smgp_factor_target.T.cpu().numpy())
# plt.show()

smgp_factor_nontarget = torch.vstack([
	-torch.sin(2 * torch.pi * (t - 2*i) / 12)
	for i in range(latent_dim)
])
smgp_factor_nontarget = smgp_factor_nontarget * 0.2

# plt.plot(smgp_factor_nontarget.T.cpu().numpy())
# plt.show()

smgp_factor_mixing = torch.vstack([
	(torch.sigmoid((t-1-2*i))*torch.sigmoid((stimulus_window-t-10+2*i))).reshape(1, -1)
	for i in range(latent_dim)
])

# plt.plot(smgp_factor_mixing.T.cpu().numpy())
# plt.show()


# loadings
shrinkage_factor = torch.logspace(0, 1, latent_dim)
which = torch.randint(0, 2, (n_channels, latent_dim, ))
heterogeneities = which*2+0.5
z = torch.randn_like(heterogeneities)
loadings = which * z / z.abs()
loadings /= shrinkage_factor.reshape(1, -1)

# noise observation_variance
observation_variance = torch.ones(n_channels)



# put them into the model
model.variables["smgp_scaling"].target_process.data = smgp_scaling_target
model.variables["smgp_scaling"].nontarget_process.data = smgp_scaling_nontarget
model.variables["smgp_scaling"].mixing_process.data = smgp_scaling_mixing

model.variables["smgp_factors"].target_process.data = smgp_factor_target
model.variables["smgp_factors"].nontarget_process.data = smgp_factor_nontarget
model.variables["smgp_factors"].mixing_process.data = smgp_factor_mixing

model.variables["heterogeneities"].data = heterogeneities.pow(-2)
model.variables["loadings"].data = loadings
model.variables["observation_variance"].data = observation_variance
model.variables["shrinkage_factor"].data = shrinkage_factor.pow(2.)

# generate the rest
model.variables["loading_processes"].generate()
model.variables["mean_factor_processes"].generate()
model.variables["factor_processes"].generate()
model.variables["observations"].generate()

torch.cov(model.variables["observations"].data.reshape(-1, n_channels).T)
# -----------------------------------------------------------------------------




# =============================================================================
# training / testing
train = copy.deepcopy(model)
test = copy.deepcopy(model)
# these two should be the same
train.variables["sequence_data"].target.data[0:120, :]
train.variables["sequence_data"].target.data[120:240, :]

train_ids = torch.hstack([
	torch.arange(i, n_sequences, n_characters)
	for i in range(0, n_characters_training)
])
test_ids = torch.hstack([
	torch.arange(i, n_sequences, n_characters)
	for i in range(n_characters_training, n_characters)
])

train.filter(train_ids)
test.filter(test_ids)
# -----------------------------------------------------------------------------




# =============================================================================
# save generating values
true_values = train.data
true_llk = train.variables["observations"].log_density
true_values["observation_log_likelihood"] = true_llk
with open(dir_data + "train/values.pkl", "wb") as f:
	pickle.dump(true_values, f)

# save training data
with open(dir_data + "train/order.pkl", "wb") as f:
	pickle.dump(train.variables["sequence_data"].order.data, f)
with open(dir_data + "train/target.pkl", "wb") as f:
	pickle.dump(train.variables["sequence_data"].target.data, f)
with open(dir_data + "train/sequence.pkl", "wb") as f:
	pickle.dump(train.variables["observations"].data, f)

# save testing values
true_values = test.data
true_llk = test.variables["observations"].log_density
true_values["observation_log_likelihood"] = true_llk
with open(dir_data + "test/values.pkl", "wb") as f:
	pickle.dump(true_values, f)

# save testing data
with open(dir_data + "test/order.pkl", "wb") as f:
	pickle.dump(test.variables["sequence_data"].order.data, f)
with open(dir_data + "test/target.pkl", "wb") as f:
	pickle.dump(test.variables["sequence_data"].target.data, f)
with open(dir_data + "test/sequence.pkl", "wb") as f:
	pickle.dump(test.variables["observations"].data, f)

# save settings
settings = {
	"latent_dim": latent_dim,
	"n_channels": n_channels,
	"stimulus_to_stimulus_interval": stimulus_to_stimulus_interval,
	"stimulus_window": stimulus_window,
	"n_stimulus": n_stimulus,
	"n_characters": n_characters_training,
	"n_repetitions": n_repetitions,
	"n_sequences": n_characters * n_characters_training,
	"nonnegative_smgp": nonnegative_smgp,
	"seed": seed
}
with open(dir_data + "train/settings.pkl", "wb") as f:
	pickle.dump(settings, f)

prior_parameters = model.prior_parameters
with open(dir_data + "train/prior_parameters.pkl", "wb") as f:
	pickle.dump(prior_parameters, f)
# -----------------------------------------------------------------------------