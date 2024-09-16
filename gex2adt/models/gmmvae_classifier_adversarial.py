import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from typing import Dict
import torch.distributions as td
import copy


class Classifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_size),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class VAE(pl.LightningModule):
    def __init__(self, encoder, gex_decoder, latent_dim, classifier_out_size, no_components,
                 components_std=1., var_transformation=lambda x: torch.exp(x) ** 0.5, learning_rate=0.001,
                 loss_function_weights=(1., 1., 1., 1., 1., 1.), batch_norm=False, optimizer="adam", clip_neg_preds=False):
        super().__init__()
        self.validation_step_outputs = []
        self.encoder = encoder
        self.gex_decoder = gex_decoder
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
        self.classifier_weight = loss_function_weights[5]

        self.optimizer = optimizer

        self.classifier = Classifier(latent_dim, classifier_out_size)

        self.automatic_optimization = False

    def configure_optimizers(self):
        gmmvae_params = list(self.encoder.parameters()) + list(self.gex_decoder.parameters())
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                gmmvae_params,
                lr=self.learning_rate,
                weight_decay=self.l2_reg_weight
            )
            optimizer_classifier = torch.optim.Adam(
                self.classifier.parameters(),
                lr=self.learning_rate,
                weight_decay=self.l2_reg_weight
            )
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                gmmvae_params,
                lr=self.learning_rate,
                weight_decay=self.l2_reg_weight
            )
            optimizer_classifier = torch.optim.Adam(
                self.classifier.parameters(),
                lr=self.learning_rate,
                weight_decay=self.l2_reg_weight
            )
        return optimizer, optimizer_classifier

    def forward(self, inputs):
        z_means = self.encoder(inputs)

        cell_pred = self.gex_decoder(z_means)

        return cell_pred, z_means

    def loss_function(self, cell_pred, z_means, cell_labels, classifier_labels, classifier_pred):
        classifier_loss, classifier_accuracy = self.classifier_loss_function(classifier_labels, classifier_pred)

        cell_labels = torch.argmax(cell_labels, dim=1)
        loss = torch.nn.CrossEntropyLoss()(cell_pred, cell_labels) - self.classifier_weight*classifier_loss

        # Calculate accuracy
        _, predicted_labels = torch.max(cell_pred.data, 1)
        matches = (predicted_labels == cell_labels).float()
        accuracy = matches.mean()

        metrics = {"loss": loss,
                   "accuracy": accuracy}
        return metrics

    def classifier_loss_function(self, true_labels, classifier_pred):
        true_labels = torch.argmax(true_labels, dim=1)
        classifier_loss = torch.nn.CrossEntropyLoss()(classifier_pred, true_labels)

        # Calculate accuracy
        _, predicted_labels = torch.max(classifier_pred.data, 1)
        matches = (predicted_labels == true_labels).float()
        accuracy = matches.mean()

        return classifier_loss, accuracy

    def manual_train(self, optimizer, loss):
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

    def training_step(self, train_batch, batch_idx):
        optimizer, optimizer_classifier = self.optimizers()

        gex_true, adt_true, guiding_classes, classifier_labels, cell_labels = train_batch
        gex_true, adt_true = gex_true.float(), adt_true.float()

        self.classifier.train()
        self.encoder.eval()
        self.gex_decoder.eval()
        with torch.no_grad():
            cell_pred, z_means = self.forward(gex_true)
        z_means_d = copy.deepcopy(z_means.detach())
        classifier_pred = self.classifier(z_means_d)
        classifier_loss, classifier_accuracy = self.classifier_loss_function(classifier_labels, classifier_pred)
        self.manual_train(optimizer_classifier, classifier_loss)

        self.classifier.eval()
        self.encoder.train()
        self.gex_decoder.train()
        cell_pred, z_means = self.forward(gex_true)
        classifier_pred = self.classifier(z_means)
        metrics = self.loss_function(cell_pred, z_means, cell_labels,
                                     classifier_labels, classifier_pred)
        self.manual_train(optimizer, metrics["loss"])
        self.log_metrics(metrics, prefix='training_step/')

    def validation_step(self, val_batch, batch_idx):
        gex_true, adt_true, guiding_classes, classifier_labels, cell_labels = val_batch
        gex_true, adt_true = gex_true.float(), adt_true.float()
        cell_pred, z_means = self.forward(gex_true)
        classifier_pred = self.classifier(z_means)

        metrics = self.loss_function(cell_pred, z_means, cell_labels, classifier_labels, classifier_pred)

        self.validation_step_outputs.append({
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"]
        })

        self.log_metrics(metrics, prefix="validation_step/")

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        avg_accuracy = torch.stack([x["accuracy"] for x in self.validation_step_outputs]).mean()

        validation_epoch_metrics = {
            "avg_loss": avg_loss,
            "avg_accuracy": avg_accuracy
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
