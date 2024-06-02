import torch
from torch import nn
from torch.nn import functional as F

class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, layers_dims, latent_dim, var_transformation=lambda x: torch.exp(x) ** 0.5):
        super(GaussianEncoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, layers_dims[0])
        self.fc2 = nn.Linear(layers_dims[0], layers_dims[1])
        self.fc3 = nn.Linear(layers_dims[1], layers_dims[2])

        self.mean_layer = nn.Linear(layers_dims[2], latent_dim)
        self.var_layer = nn.Linear(layers_dims[2], latent_dim)

        self.var_transformation = var_transformation

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        z_means = self.mean_layer(x)
        z_log_vars = self.var_layer(x)
        z_vars = self.var_transformation(z_log_vars)

        return z_means, z_vars


class GaussianCountsEncoder(nn.Module):
    def __init__(self, input_dim, layers_dims, latent_dim, var_transformation=lambda x: torch.exp(x) ** 0.5):
        super(GaussianCountsEncoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, layers_dims[0])
        self.fc2 = nn.Linear(layers_dims[0], layers_dims[1])
        self.fc3 = nn.Linear(layers_dims[1], layers_dims[2])

        self.mean_layer = nn.Linear(layers_dims[2], latent_dim)
        self.var_layer = nn.Linear(layers_dims[2], latent_dim)

        self.var_transformation = var_transformation

    def forward(self, input):
        x = torch.log1p(input)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        z_means = self.mean_layer(x)
        z_log_vars = self.var_layer(x)
        z_vars = self.var_transformation(z_log_vars)

        return z_means, z_vars