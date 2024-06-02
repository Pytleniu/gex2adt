import torch
import torch.distributions as td
from torch import nn
from torch.nn import functional as F
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
import scvi


class GaussianDecoder(nn.Module):
    def __init__(self, latent_dim, layers_dims, output_dim):
        super(GaussianDecoder,
              self).__init__()
        scvi.settings.seed = 1
        self.fc1 = nn.Linear(latent_dim, layers_dims[0])
        self.fc2 = nn.Linear(layers_dims[0], layers_dims[1])
        self.fc3 = nn.Linear(layers_dims[1], layers_dims[2])
        self.fc4 = nn.Linear(layers_dims[2], output_dim)

    def forward(self, z_sample):
        x = self.fc1(z_sample)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        covs = torch.ones_like(x)

        return td.Independent(
            td.Normal(x, covs.sqrt()),
            1
        )

    def decode(self, z_sample):
        x = self.fc1(z_sample)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return F.relu(x)


class PoissonDecoder(nn.Module):
    def __init__(self, latent_dim, layers_dims, output_dim):
        super(PoissonDecoder,
              self).__init__()
        scvi.settings.seed = 1
        self.fc1 = nn.Linear(latent_dim, layers_dims[0])
        self.fc2 = nn.Linear(layers_dims[0], layers_dims[1])
        self.fc3 = nn.Linear(layers_dims[1], layers_dims[2])
        self.fc4_out = nn.Linear(layers_dims[2], output_dim)

    def forward(self, z_sample):
        x = self.fc1(z_sample)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        rate = self.fc4_out(x)
        rate = torch.exp(rate)

        return td.Poisson(
            rate=rate,
            validate_args=False
        )

    def decode(self, z_sample):
        x = self.fc1(z_sample)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        rate = self.fc4_out(x)
        rate = torch.exp(rate)

        return rate


class NegativeBinomialDecoder(nn.Module):
    def __init__(self, latent_dim, layers_dims, output_dim):
        super(NegativeBinomialDecoder,
              self).__init__()
        scvi.settings.seed = 1
        self.fc1 = nn.Linear(latent_dim, layers_dims[0])
        self.fc2 = nn.Linear(layers_dims[0], layers_dims[1])
        self.fc3 = nn.Linear(layers_dims[1], layers_dims[2])
        self.fc4_out1 = nn.Linear(layers_dims[2], output_dim)
        self.fc4_out2 = nn.Linear(layers_dims[2], output_dim)

    def forward(self, z_sample):
        x = self.fc1(z_sample)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        mu = self.fc4_out1(x)
        mu = F.softplus(mu)

        theta = self.fc4_out2(x)
        theta = F.softplus(theta)

        return NegativeBinomial(
            mu=mu,
            theta=theta,
            validate_args=True
        )

    def decode(self, z_sample):
        x = self.fc1(z_sample)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        mu = self.fc4_out1(x)
        mu = F.softplus(mu)

        return mu


class ZeroInflatedNegativeBinomialDecoder(nn.Module):
    def __init__(self, latent_dim, layers_dims, output_dim, train_theta=True):
        super(ZeroInflatedNegativeBinomialDecoder,
              self).__init__()
        scvi.settings.seed = 1
        self.fc1 = nn.Linear(latent_dim, layers_dims[0])
        self.fc2 = nn.Linear(layers_dims[0], layers_dims[1])
        self.fc3 = nn.Linear(layers_dims[1], layers_dims[2])

        self.theta_fc1 = nn.Linear(layers_dims[2], layers_dims[3])
        self.theta_fc2 = nn.Linear(layers_dims[3], output_dim)

        self.mu_fc1 = nn.Linear(layers_dims[2], layers_dims[3])
        self.mu_fc2 = nn.Linear(layers_dims[3], output_dim)

        self.zi_logits_fc1 = nn.Linear(layers_dims[2], layers_dims[3])
        self.zi_logits_fc2 = nn.Linear(layers_dims[3], output_dim)
        self.train_theta = train_theta

    def forward(self, z_sample):
        x = self.fc1(z_sample)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        mu = self.mu_fc1(x)
        mu = self.mu_fc2(mu)
        mu = F.softplus(mu)

        if self.train_theta:
            theta = self.theta_fc1(x)
            theta = self.theta_fc2(theta)
            theta = F.softplus(theta)
        else:
            theta = torch.ones_like(mu) * 5

        zi_logits = self.zi_logits_fc1(x)
        zi_logits = self.zi_logits_fc2(zi_logits) * 10.0 + 4
        # zi_logits = torch.exp(zi_logits)

        return ZeroInflatedNegativeBinomial(
            mu=mu,
            theta=theta,
            zi_logits=zi_logits,
            validate_args=True,
        )

    def decode(self, z_sample, threshold=None):
        mu, theta, zi_logits = self.full_decode(z_sample)
        preds = self(z_sample).mean
        if threshold is not None:
            preds[zi_logits > threshold] = 0
        return preds

    def full_decode(self, z_sample):
        x = self.fc1(z_sample)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

        mu = self.mu_fc1(x)
        mu = self.mu_fc2(mu)
        mu = F.softplus(mu)

        if self.train_theta:
            theta = self.theta_fc1(x)
            theta = self.theta_fc2(theta)
            theta = F.softplus(theta)
        else:
            theta = torch.ones_like(mu) * 5

        zi_logits = self.zi_logits_fc1(x)
        zi_logits = self.zi_logits_fc2(zi_logits) * 10.0 + 4
        # zi_logits = torch.exp(zi_logits)

        return mu, theta, zi_logits
