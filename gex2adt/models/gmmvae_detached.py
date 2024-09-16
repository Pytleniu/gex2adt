import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from typing import Dict
import torch.distributions as td
from torch import nn


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class GMMVAE(pl.LightningModule):
    def __init__(self, encoder, gex_decoder, adt_decoder, latent_dim, linear_out_size, no_components,
                 components_std=1., var_transformation=lambda x: torch.exp(x) ** 0.5, learning_rate=0.001,
                 loss_function_weights=(1., 1., 1., 1., 1., 1.), batch_norm=False, optimizer="adam", clip_neg_preds=False):
        super().__init__()
        self.validation_step_outputs = []
        self.encoder = encoder
        self.gex_decoder = gex_decoder
        self.adt_decoder = adt_decoder
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        self.component_logits = torch.nn.Parameter(data=torch.zeros(size=(no_components,)), requires_grad=True)
        self.means = torch.nn.Parameter(torch.randn(no_components, self.latent_dim), requires_grad=True)

        # STDs of GMM
        self.register_buffer("stds", components_std * torch.ones(no_components, self.latent_dim))

        # Loss function weights
        self.gex_likelihood_weight = loss_function_weights[0]
        self.adt_likelihood_weight = loss_function_weights[1]
        self.entropy_weight = loss_function_weights[2]
        self.gmm_likelihood_weight = loss_function_weights[3]
        self.l2_reg_weight = loss_function_weights[4]

        self.optimizer = optimizer

        self.linear = Linear(latent_dim, linear_out_size)
        self.linear_optimizer = torch.optim.Adam(
            self.linear.parameters(),
            lr=0.0001,
            weight_decay=0.0001
        )

    def configure_optimizers(self):
        filtered_params = [param for name, param in self.named_parameters() if "linear" not in name]
        if self.optimizer == "adam":
            optim = torch.optim.Adam(
                filtered_params,
                lr=self.learning_rate,
                weight_decay=self.l2_reg_weight
            )
        if self.optimizer == "sgd":
            optim = torch.optim.SGD(
                filtered_params,
                lr=self.learning_rate,
                weight_decay=self.l2_reg_weight
            )
        return optim

    def forward(self, inputs):
        z_means, z_stds = self.encoder(inputs)

        normal_rv = self.make_normal_rv(z_means, z_stds)
        z_sample = normal_rv.rsample()

        gmm = self.make_gmm()

        gex_dist = self.gex_decoder(z_sample)
        adt_dist = self.adt_decoder(z_sample)

        return gex_dist, adt_dist, z_sample, normal_rv, gmm, z_means

    def loss_function(self, gex_true, adt_true, gex_dist, adt_dist, z_means, z_sample, normal_rv, guiding_classes,
                      gmm, linear_labels, linear_pred):
        entropy = torch.mean(normal_rv.entropy())
        gex_mse, adt_mse = self.compute_mse_losses(gex_true, adt_true, z_means)
        gmm_likelihood = self.compute_gmm_likelihood(z_sample, guiding_classes, gmm)

        gex_likelihood = - gex_dist.log_prob(gex_true).mean() / gex_true.size()[1]
        adt_likelihood = - adt_dist.log_prob(adt_true).mean() / adt_true.size()[1]

        loss = self.gex_likelihood_weight * gex_likelihood \
            + self.adt_likelihood_weight * adt_likelihood \
            - self.entropy_weight * entropy \
            - self.gmm_likelihood_weight * gmm_likelihood

        linear_loss, linear_accuracy = self.linear_loss_function(linear_labels, linear_pred)

        metrics = {
            "gex_MSE": gex_mse,
            "adt_MSE": adt_mse,
            "gex_pred_likelihood": gex_likelihood,
            "adt_pred_likelihood": adt_likelihood,
            "entropy": entropy,
            "GMM_likelihood": gmm_likelihood,
            "loss": loss,
            "linear_loss": linear_loss,
            "linear_accuracy": linear_accuracy
        }

        return metrics

    def linear_loss_function(self, true_labels, linear_pred):
        true_labels = torch.argmax(true_labels, dim=1)
        linear_loss = nn.CrossEntropyLoss()(linear_pred, true_labels)

        # Calculate accuracy
        _, predicted_labels = torch.max(linear_pred.data, 1)
        matches = (predicted_labels == true_labels).float()
        accuracy = matches.mean()

        return linear_loss, accuracy

    def manual_train(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def training_step(self, train_batch, batch_idx):
        gex_true, adt_true, guiding_classes, linear_labels = train_batch
        gex_true, adt_true = gex_true.float(), adt_true.float()

        gex_dist, adt_dist, z_sample, normal_rv, gmm, z_means = self.forward(gex_true)

        linear_pred = self.linear(z_means.detach())

        metrics = self.loss_function(gex_true, adt_true, gex_dist, adt_dist, z_means, z_sample,
                                     normal_rv, guiding_classes, gmm, linear_labels, linear_pred)

        self.manual_train(self.linear_optimizer, metrics["linear_loss"])

        self.log_metrics(metrics, prefix='training_step/')

        return metrics['loss']

    def validation_step(self, val_batch, batch_idx):
        gex_true, adt_true, guiding_classes, linear_labels = val_batch
        gex_true, adt_true = gex_true.float(), adt_true.float()

        gex_dist, adt_dist, z_sample, normal_rv, gmm, z_means = self.forward(gex_true)

        linear_pred = self.linear(z_means.detach())

        metrics = self.loss_function(gex_true, adt_true, gex_dist, adt_dist, z_means, z_sample,
                                     normal_rv, guiding_classes, gmm, linear_labels, linear_pred)

        self.validation_step_outputs.append({
            "input_likelihood": metrics["gex_pred_likelihood"],
            "guiding_likelihood": metrics["adt_pred_likelihood"],
            "gex_mse": metrics["gex_MSE"],
            "adt_mse": metrics["adt_MSE"],
            "linear_accuracy": metrics["linear_accuracy"]
        })
        self.log_metrics(metrics, prefix="validation_step/")

    def on_validation_epoch_end(self):
        avg_adt_mse = torch.stack([x["gex_mse"] for x in self.validation_step_outputs]).mean()
        avg_adt_rmse = avg_adt_mse ** 0.5

        avg_gex_mse = torch.stack([x["adt_mse"] for x in self.validation_step_outputs]).mean()
        avg_gex_rmse = avg_gex_mse ** 0.5

        avg_linear_accuracy = torch.stack([x["linear_accuracy"] for x in self.validation_step_outputs]).mean()

        validation_epoch_metrics = {
            "avg_adt_mse": avg_adt_mse,
            "avg_gex_mse": avg_gex_mse,
            "avg_adt_rmse": avg_adt_rmse,
            "avg_gex_rmse": avg_gex_rmse,
            "avg_linear_accuracy": avg_linear_accuracy
        }

        self.log_metrics(validation_epoch_metrics, prefix="validation_epoch_metrics/")

        self.validation_step_outputs.clear()

    def compute_mse_losses(self, gex_true, adt_true, z_means):
        gex_pred = self.gex_decoder.decode(z_means)
        adt_pred = self.adt_decoder.decode(z_means)
        return F.mse_loss(gex_true, gex_pred), F.mse_loss(adt_true, adt_pred)

    def compute_gmm_likelihood(self, z_sample, guiding_classes, gmm):
        mask_unknown = torch.isnan(guiding_classes)
        mask_known = torch.logical_not(mask_unknown)

        z_sample_known = z_sample[mask_known]
        z_sample_unknown = z_sample[mask_unknown]

        if z_sample_unknown.shape[0] == 0:  # All classes known
            per_component_log_probs = torch.stack([gmm.component_distribution.log_prob(z_sample[i]) for i in range(z_sample.shape[0])])
            gmm_likelihood = per_component_log_probs[torch.arange(z_sample.shape[0]), guiding_classes.to(torch.int64)]
            gmm_likelihood = torch.mean(gmm_likelihood)

            return gmm_likelihood

        if z_sample_known.shape[0] == 0:  # All classes unknown
            gmm_likelihood = gmm.log_prob(z_sample)
            gmm_likelihood = torch.mean(gmm_likelihood)

            return gmm_likelihood

        # Mix of known and unknown classes in a batch
        classes_known = guiding_classes[mask_known]
        classes_known = classes_known.to(torch.int64)

        per_component_log_probs = torch.stack([gmm.component_distribution.log_prob(z_sample_known[i]) for i in range(z_sample_known.shape[0])])
        gmm_likelihood_known = per_component_log_probs[torch.arange(z_sample_known.shape[0]), classes_known]

        gmm_likelihood_unknown = gmm.log_prob(z_sample_unknown)

        gmm_likelihood = torch.mean(torch.cat([gmm_likelihood_known, gmm_likelihood_unknown]))

        return gmm_likelihood

    def log_metrics(self, metrics: Dict[str, torch.Tensor], prefix: str = "", on_step: bool = False, on_epoch: bool = True, prog_bar: bool = True):
        for key, value in metrics.items():
            self.log(f"{prefix}{key}", value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)

    def make_gmm(self):
        categorical = td.Categorical(logits=self.component_logits)
        comp = td.Independent(
            td.Normal(self.means, self.stds),
            reinterpreted_batch_ndims=1
        )
        return td.MixtureSameFamily(categorical, comp)

    def make_normal_rv(self, means, vars):
        # return td.Independent(td.Normal(means, covs.sqrt()), 1)
        return td.MultivariateNormal(
            means,
            torch.stack([torch.diag(vars[i, :])
                        for i in range(vars.shape[0])], axis=0)
        )

    @staticmethod
    def mse_loss_with_nans(rec, target):
        # When missing data are nan's
        mask = torch.isnan(target)
        neg_likelihood = F.mse_loss(
            rec[~mask],
            target[~mask]
        )
        if torch.isnan(neg_likelihood):
            return 0
        else:
            return neg_likelihood
