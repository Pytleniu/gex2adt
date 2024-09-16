import argparse
import importlib
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch import seed_everything
from gex2adt.decoders.decoder import ClassifierDecoder
from gex2adt.encoders.encoder import ClassifierEncoder
from lightning.pytorch.loggers import CSVLogger
# from lightning.pytorch.loggers import WandbLogger
# import wandb
from copy import deepcopy as copy
import scib
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import entropy


class LinearRegression(torch.nn.Module):

    def __init__(self, input_dim, num_classes):
        super(LinearRegression, self).__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def save_dict_as_csv(dict, path):
    df = pd.DataFrame([dict])
    df.to_csv(path, index=False)


def evaluate_regression(train_input, val_input, train_labels, val_labels, num_epochs=500):
    regression_model = LinearRegression(train_input.shape[1], train_labels.shape[1]).to('cuda')
    optimizer_regression = torch.optim.SGD(regression_model.parameters(), lr=0.1)

    true_labels_train = torch.argmax(train_labels, dim=1).to('cuda')
    true_labels_val = torch.argmax(val_labels, dim=1).to('cuda')

    for epoch in range(num_epochs):
        output_train = regression_model(train_input)

        loss_regression = torch.nn.CrossEntropyLoss()(output_train, true_labels_train)
        optimizer_regression.zero_grad()
        loss_regression.backward()
        optimizer_regression.step()

        _, predicted = torch.max(output_train, 1)
        matches = (predicted == true_labels_train).float()
        accuracy = matches.mean()

        # print(f'Epoch [{epoch+1}/{num_epochs}] -  Loss batch: {loss_regression.item():.4f}, Accuracy batch: {accuracy:.4f}')

    with torch.no_grad():
        output_val = regression_model(val_input)
        loss_regression_val = torch.nn.CrossEntropyLoss()(output_val, true_labels_val)
        _, predicted_val = torch.max(output_val, 1)
        matches_val = (predicted_val == true_labels_val).float()
        accuracy_val = matches_val.mean()

    results = {
        'loss_train': loss_regression.item(),
        'accuracy_train': accuracy.item(),
        'loss_val': loss_regression_val.item(),
        'accuracy_val': accuracy_val.item()
    }

    return results


def create_subsample(labels, sample_size=100):
    subsample_list = []
    for label in labels.unique():
        label_idx = np.where(labels == label)
        subsample = np.random.choice(label_idx[0], sample_size)
        subsample_list.append(subsample)
    return np.hstack(subsample_list)


def create_color_palette(unique_values, palette="tab20"):
    return dict(zip(unique_values, sns.color_palette(palette=palette, n_colors=len(unique_values))))


def plot_labels(ax, x, y, labels_indices, title, color_palette=None, alpha=0.6, legend=True):
    for label, indices in labels_indices:
        ax.scatter(
            x[indices],
            y[indices],
            color=color_palette[label],
            label=label,
            edgecolors='white',
            s=50,
            alpha=alpha,
        )
    ax.set_title(title)
    if legend:
        ax.legend()


def get_log_prob(model, z_mean, gmm=None, detach=True):
    if not torch.is_tensor(z_mean):
        z_mean = torch.Tensor(z_mean)

    if gmm is None:
        with torch.no_grad():
            gmm = model.make_gmm()

    gmm_log_prob = gmm.log_prob(z_mean)

    if detach:
        gmm_log_prob = gmm_log_prob.detach().cpu().numpy()

    return gmm_log_prob


def get_class_probs(model, z_mean, detach=True):
    if not torch.is_tensor(z_mean):
        z_mean = torch.Tensor(z_mean)

    with torch.no_grad():
        gmm = model.make_gmm()
    per_component_log_probs = gmm.component_distribution.log_prob(z_mean[:, None])
    gmm_log_prob = get_log_prob(model, z_mean, gmm=gmm, detach=False)
    log_prob = per_component_log_probs - gmm_log_prob[:, None]
    class_probs = F.softmax(log_prob, dim=1)

    if detach:
        class_probs = class_probs.detach().cpu().numpy()

    return class_probs


def get_mse(model, z_mean, gex_array, adt_array):
    if not torch.is_tensor(z_mean):
        z_mean = torch.Tensor(z_mean)

    if not torch.is_tensor(gex_array):
        gex_array = torch.Tensor(gex_array)

    if not torch.is_tensor(adt_array):
        adt_array = torch.Tensor(adt_array)

    with torch.no_grad():
        gex_decoded = model.gex_decoder.decode(z_mean).cpu()
        adt_decoded = model.adt_decoder.decode(z_mean).cpu()

    gex_mse = F.mse_loss(gex_array, gex_decoded, reduction="none").detach().cpu().numpy()
    adt_mse = F.mse_loss(adt_array, adt_decoded, reduction="none").detach().cpu().numpy()

    return gex_mse, adt_mse


def plot_contour(fig, ax, x, y, z, title):
    contour_plot = ax.tricontourf(x, y, z)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(contour_plot, cax=cax, orientation='vertical')
    ax.set_title(title)


def main(model_name, output_file_name, linear_type):
    # environ["WANDB__SERVICE_WAIT"] = "300"

    ########################
    # Setting random seeds #
    ########################
    seed_everything(42, workers=True)

    ###############################
    # Importing the desired model #
    ###############################
    module = importlib.import_module(f"gex2adt.models.{model_name}")
    VAE = getattr(module, 'VAE')

    #################
    # Setting paths #
    #################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'neurips2021',
                             'GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad')

    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    MODEL_DIR = os.path.join(BASE_DIR, '..', 'models', model_name)
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, f'{output_file_name}.pt')

    METRICS_DIR = os.path.join(RESULTS_DIR, model_name, 'metrics')
    os.makedirs(METRICS_DIR, exist_ok=True)

    SCIB_METRICS_DIR = os.path.join(METRICS_DIR, 'scib')
    os.makedirs(SCIB_METRICS_DIR, exist_ok=True)

    SCIB_BIO_METRICS_DIR = os.path.join(SCIB_METRICS_DIR, 'bio')
    os.makedirs(SCIB_BIO_METRICS_DIR, exist_ok=True)
    SCIB_BIO_METRICS_PATH = os.path.join(SCIB_BIO_METRICS_DIR, f'{output_file_name}.csv')

    SCIB_INTEGRATION_METRICS_DIR = os.path.join(SCIB_METRICS_DIR, 'integration')
    os.makedirs(SCIB_INTEGRATION_METRICS_DIR, exist_ok=True)
    SCIB_INTEGRATION_METRICS_PATH = os.path.join(SCIB_INTEGRATION_METRICS_DIR, f'{output_file_name}.csv')

    REGRESSION_METRICS_DIR = os.path.join(METRICS_DIR, 'regression')
    os.makedirs(REGRESSION_METRICS_DIR, exist_ok=True)

    BATCH_REGRESSION_METRIC_DIR = os.path.join(REGRESSION_METRICS_DIR, 'batch')
    os.makedirs(BATCH_REGRESSION_METRIC_DIR, exist_ok=True)
    BATCH_REGRESSION_METRIC_PATH = os.path.join(BATCH_REGRESSION_METRIC_DIR, f'{output_file_name}.csv')

    SITE_REGRESSION_METRIC_DIR = os.path.join(REGRESSION_METRICS_DIR, 'site')
    os.makedirs(SITE_REGRESSION_METRIC_DIR, exist_ok=True)
    SITE_REGRESSION_METRIC_PATH = os.path.join(SITE_REGRESSION_METRIC_DIR, f'{output_file_name}.csv')

    DONOR_REGRESSION_METRIC_DIR = os.path.join(REGRESSION_METRICS_DIR, 'donor')
    os.makedirs(DONOR_REGRESSION_METRIC_DIR, exist_ok=True)
    DONOR_REGRESSION_METRIC_PATH = os.path.join(DONOR_REGRESSION_METRIC_DIR, f'{output_file_name}.csv')

    CELL_REGRESSION_METRIC_DIR = os.path.join(REGRESSION_METRICS_DIR, 'cell')
    os.makedirs(CELL_REGRESSION_METRIC_DIR, exist_ok=True)
    CELL_REGRESSION_METRIC_PATH = os.path.join(CELL_REGRESSION_METRIC_DIR, f'{output_file_name}.csv')

    GRAPHS_DIR = os.path.join(RESULTS_DIR, model_name, 'graphs')
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    PCA_DIR = os.path.join(GRAPHS_DIR, 'pca')
    os.makedirs(PCA_DIR, exist_ok=True)
    PCA_PATH = os.path.join(PCA_DIR, f'{output_file_name}.png')

    # GRAPH_3D_DIR = os.path.join(GRAPHS_DIR, 'pca_3d')
    # os.makedirs(GRAPH_3D_DIR, exist_ok=True)
    # GRAPH_3D_PATH = os.path.join(GRAPH_3D_DIR, f'{output_file_name}.png')

    BSD_DIR = os.path.join(GRAPHS_DIR, 'batch_site_donor', f'{output_file_name}')
    os.makedirs(BSD_DIR, exist_ok=True)

    NEW_PATIENT_DIR = os.path.join(GRAPHS_DIR, 'new_patient', f'{output_file_name}')
    os.makedirs(NEW_PATIENT_DIR, exist_ok=True)

    LOG_PROB_ENTR = os.path.join(GRAPHS_DIR, 'log_prob_entropy', f'{output_file_name}')
    os.makedirs(LOG_PROB_ENTR, exist_ok=True)

    GEX_ADT_MSE = os.path.join(GRAPHS_DIR, 'gex_adt_mse', f'{output_file_name}')
    os.makedirs(GEX_ADT_MSE, exist_ok=True)

    MSE_PLOTS = os.path.join(GRAPHS_DIR, 'mse_plots', f'{output_file_name}')
    os.makedirs(MSE_PLOTS, exist_ok=True)

    PATH_CELLS = os.path.join(BASE_DIR, '..', 'data', 'neurips2021', 'neurips2021_celltypes.csv')

    ####################
    # Data preparation #
    ####################

    # Load the dataset
    adata = sc.read_h5ad(DATA_PATH)
    # sc.pp.subsample(adata, fraction=0.1, copy=False)

    # Extract GEX and ADT
    adata_gex = adata[:, adata.var.feature_types == "GEX"]
    adata_adt = adata[:, adata.var.feature_types == "ADT"]
    X_gex = adata_gex.X.toarray()
    X_adt = adata_adt.X.toarray()

    # Clean up to free memory
    del adata

    # Log1p X_gex, X_adt is already log1p transformed
    X_gex = np.log1p(X_gex)

    # Map cell types from Level 4 to Level 3 and encode into numerical
    cell_types = adata_gex.obs['cell_type']
    levels_df = pd.read_csv(PATH_CELLS)
    level3_mapping = dict(zip(levels_df['Level 4'], levels_df['Level 3']))
    cell_types_mapped = cell_types.map(level3_mapping)
    label_encoder = preprocessing.LabelEncoder()
    Y = label_encoder.fit_transform(cell_types_mapped)

    batches = adata_gex.obs['batch']
    sites = adata_gex.obs['Site']
    donors = adata_gex.obs['DonorNumber']
    # Auxiliary to data analysis
    # cell_types.value_counts().plot(kind='bar')
    # adata_gex.obs['batch'].value_counts().plot(kind='bar')
    # plt.show()

    # Let's try to predict ADT for new patient at a site we have seen before
    TEST_BATCH = 's4d8'

    test_idx = np.where(adata_gex.obs.batch == TEST_BATCH)
    train_val_idx = np.where(adata_gex.obs.batch != TEST_BATCH)

    out_encoding = pd.get_dummies(cell_types_mapped, prefix="cell_type").astype(int)
    out_labels = torch.tensor(out_encoding.values.tolist())
    out_labels_train_val = out_labels[train_val_idx]
    out_labels_test = out_labels[test_idx]

    X_gex_train_val = X_gex[train_val_idx]
    X_gex_test = X_gex[test_idx]

    X_adt_train_val = X_adt[train_val_idx]
    X_adt_test = X_adt[test_idx]

    Y_train_val = Y[train_val_idx]
    Y_test = Y[test_idx]

    print(
        X_gex_train_val.shape,
        X_adt_train_val.shape,
        Y_train_val.shape,
        X_gex_test.shape,
        X_adt_test.shape,
        Y_test.shape
    )

    batch_encoding = pd.get_dummies(adata_gex.obs["batch"], prefix="batch").astype(int)
    batch_labels = torch.tensor(batch_encoding.values.tolist())
    batch_labels_train_val = batch_labels[train_val_idx]

    site_encoding = pd.get_dummies(adata_gex.obs["Site"], prefix="site").astype(int)
    site_labels = torch.tensor(site_encoding.values.tolist())
    site_labels_train_val = site_labels[train_val_idx]

    donor_encoding = pd.get_dummies(adata_gex.obs["DonorNumber"], prefix="donor").astype(int)
    donor_labels = torch.tensor(donor_encoding.values.tolist())
    donor_labels_train_val = donor_labels[train_val_idx]

    cells_encoding = pd.get_dummies(adata_gex.obs["cell_type"], prefix="cell_type").astype(int)
    cell_labels = torch.tensor(cells_encoding.values.tolist())
    cell_labels_train_val = cell_labels[train_val_idx]

    if linear_type == "batch":
        linear_labels = copy(batch_labels)
    elif linear_type == "site":
        linear_labels = copy(site_labels)
    elif linear_type == "donor":
        linear_labels = copy(donor_labels)
    elif linear_type == "cell":
        linear_labels = copy(cell_labels)

    linear_train_val = linear_labels[train_val_idx]

    out = train_test_split(
        X_gex_train_val,
        X_adt_train_val,
        Y_train_val,
        linear_train_val,
        batch_labels_train_val,
        site_labels_train_val,
        donor_labels_train_val,
        cell_labels_train_val,
        out_labels_train_val,
        test_size=0.25,
        stratify=Y_train_val,
        random_state=42
    )

    X_gex_train = out[0]
    X_gex_val = out[1]
    X_adt_train = out[2]
    X_adt_val = out[3]
    Y_train = out[4]
    Y_val = out[5]
    linear_train = out[6]
    linear_val = out[7]
    batch_labels_train = out[8]
    batch_labels_val = out[9]
    site_labels_train = out[10]
    site_labels_val = out[11]
    donor_labels_train = out[12]
    donor_labels_val = out[13]
    cell_labels_train = out[14]
    cell_labels_val = out[15]
    out_labels_train = out[16]
    out_labels_val = out[17]

    #################
    # Configuration #
    #################

    config = {
        "data": {
            "gex_dim": X_gex.shape[1],
            "adt_dim": X_adt.shape[1],
            "no_components": cell_types.nunique(),
        },
        "model": {
            "latent_dim": 50,
            "linear_out_size": linear_labels.shape[1],
            "components_std": 1.,
            "var_transformation": lambda x: torch.exp(x) ** 0.5,
            "batch_norm": False,
            "optimizer": "adam",
            "out_softplus": False,
            "learning_rate": 0.0005,
            "loss_function_weights": (0.449, 0.449, 0.001, 1.0, 0.0, 0.5),
            "encoder_layers_dims": (512, 256, 128),
            "gex_decoder_layers_dims": (128, 256, 512),
            "adt_decoder_layers_dims": (128, 256, 512),
        },
        "training": {
            "batch_size": 2048,
            "num_workers": 16,
            "num_epochs": 25,
            "num_gpus": 1,
            "train": True,
            "load": False,
            "accelerator": "gpu",
            "check_val_every_n_epoch": 1,
            "num_nodes": 1,
            "strategy": "auto",
            "deterministic": True

        }
    }

    encoder = ClassifierEncoder(
        input_dim=config["data"]["gex_dim"],
        layers_dims=config["model"]["encoder_layers_dims"],
        latent_dim=config["model"]["latent_dim"]
    )
    gex_decoder = ClassifierDecoder(
        latent_dim=config["model"]["latent_dim"],
        layers_dims=config["model"]["gex_decoder_layers_dims"],
        output_dim=config["data"]["gex_dim"]
    )

    model = VAE(
        encoder=encoder,
        gex_decoder=gex_decoder,
        latent_dim=config["model"]["latent_dim"],
        classifier_out_size=config["model"]["linear_out_size"],
        no_components=config["data"]["no_components"],
        components_std=config["model"]["components_std"],
        learning_rate=config["model"]["learning_rate"],
        loss_function_weights=config["model"]["loss_function_weights"],
        var_transformation=config["model"]["var_transformation"],
        batch_norm=config["model"]["batch_norm"],
        optimizer=config["model"]["optimizer"],
    )

    ######################
    # Create dataloaders #
    ######################

    train_dataset = TensorDataset(torch.Tensor(X_gex_train), torch.Tensor(X_adt_train), torch.Tensor(Y_train), torch.Tensor(linear_train), torch.Tensor(out_labels_train))
    val_dataset = TensorDataset(torch.Tensor(X_gex_val), torch.Tensor(X_adt_val), torch.Tensor(Y_val), torch.Tensor(linear_val), torch.Tensor(out_labels_val))
    train_data_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"],
                                   num_workers=config["training"]["num_workers"], shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"],
                                 num_workers=config["training"]["num_workers"], shuffle=False, pin_memory=True)

    ####################
    #     Training     #
    ####################

    if config["training"]["load"]:
        model.load_state_dict(torch.load(MODEL_PATH))
        # model = model.to('cpu')
    if config["training"]["train"]:
        # wandb_logger = WandbLogger(name=f'{model_name}_{linear_type}', project='Inz', log_model='all')
        logger = CSVLogger(f"{BASE_DIR}/logs", name=model_name)
        trainer = pl.Trainer(max_epochs=config["training"]["num_epochs"],
                             accelerator=config["training"]["accelerator"],
                             devices=config["training"]["num_gpus"],
                             check_val_every_n_epoch=config["training"]["check_val_every_n_epoch"],
                             num_nodes=config["training"]["num_nodes"],
                             strategy=config["training"]["strategy"],
                             deterministic=config["training"]["deterministic"],
                             logger=logger)
        # wandb_logger.watch(model)
        trainer.fit(model, train_data_loader, val_data_loader)
        # wandb_logger.experiment.unwatch(model)
        torch.save(model.state_dict(), MODEL_PATH)
    # wandb.finish()

    ####################
    #    Evaluation    #
    ####################
    model.eval()
    model = model.to("cuda")
    X_gex_all_tensor = torch.Tensor(X_gex).to("cuda")
    X_gex_train_tensor = torch.Tensor(X_gex_train).to("cuda")
    X_gex_val_tensor = torch.Tensor(X_gex_val).to("cuda")
    X_gex_test_tensor = torch.Tensor(X_gex_test).to("cuda")
    with torch.no_grad():
        _, latent_train = model(torch.Tensor(X_gex_train_tensor))
        _, latent_val = model(torch.Tensor(X_gex_val_tensor))
        _, latent_all = model(torch.Tensor(X_gex_all_tensor))
        _, latent_test = model(torch.Tensor(X_gex_test_tensor))
        latent_train = latent_train.detach()
        latent_val = latent_val.detach()
        latent_all = latent_all.detach()
        latent_test = latent_test.detach()

    # REGRESSION METRICS
    batch_regression_metrics = evaluate_regression(latent_train, latent_val, batch_labels_train, batch_labels_val)
    sites_regression_metrics = evaluate_regression(latent_train, latent_val, site_labels_train, site_labels_val)
    donor_regression_metrics = evaluate_regression(latent_train, latent_val, donor_labels_train, donor_labels_val)
    cells_regression_metrics = evaluate_regression(latent_train, latent_val, cell_labels_train, cell_labels_val)

    save_dict_as_csv(batch_regression_metrics, BATCH_REGRESSION_METRIC_PATH)
    save_dict_as_csv(sites_regression_metrics, SITE_REGRESSION_METRIC_PATH)
    save_dict_as_csv(donor_regression_metrics, DONOR_REGRESSION_METRIC_PATH)
    save_dict_as_csv(cells_regression_metrics, CELL_REGRESSION_METRIC_PATH)

    # SCIB METRICS
    adata_gex.obsm['X_emb'] = latent_all.cpu().numpy()

    # Biological conservation
    asw_bio = scib.me.silhouette(adata_gex, label_key="cell_type", embed="X_emb")
    asw_bio_iso_batch = scib.me.isolated_labels_asw(adata_gex, batch_key="batch", label_key="cell_type", embed="X_emb")
    asw_bio_iso_site = scib.me.isolated_labels_asw(adata_gex, batch_key="Site", label_key="cell_type", embed="X_emb")
    asw_bio_iso_donor = scib.me.isolated_labels_asw(adata_gex, batch_key="DonorNumber", label_key="cell_type", embed="X_emb")
    clisi = scib.me.clisi_graph(adata_gex, label_key="cell_type", type_="embed", use_rep="X_emb")

    # Batch correction
    asw_batch = scib.me.silhouette_batch(adata_gex, batch_key="batch", label_key="cell_type", embed="X_emb")
    asw_site = scib.me.silhouette_batch(adata_gex, batch_key="Site", label_key="cell_type", embed="X_emb")
    asw_donor = scib.me.silhouette_batch(adata_gex, batch_key="DonorNumber", label_key="cell_type", embed="X_emb")
    ilisi_batch = scib.me.ilisi_graph(adata_gex, batch_key="batch", type_="embed", use_rep="X_emb")
    ilisi_site = scib.me.ilisi_graph(adata_gex, batch_key="Site", type_="embed", use_rep="X_emb")
    ilisi_donor = scib.me.ilisi_graph(adata_gex, batch_key="DonorNumber", type_="embed", use_rep="X_emb")
    # kbet = scib.me.kBET(adata_gex, batch_key="batch", label_key="cell_type", type_="embed", embed="X_emb")

    bio_metrics = {
        'asw_bio': asw_bio,
        'asw_bio_iso_batch': asw_bio_iso_batch,
        'asw_bio_iso_site': asw_bio_iso_site,
        'asw_bio_iso_donor': asw_bio_iso_donor,
        'clisi': clisi
    }
    integration_metrics = {
        'asw_batch': asw_batch,
        'asw_site': asw_site,
        'asw_donor': asw_donor,
        'ilisi_batch': ilisi_batch,
        'ilisi_site': ilisi_site,
        'ilisi_donor': ilisi_donor
    }
    save_dict_as_csv(bio_metrics, SCIB_BIO_METRICS_PATH)
    save_dict_as_csv(integration_metrics, SCIB_INTEGRATION_METRICS_PATH)

    # Visualisations
    no_pcs = 15
    pca = PCA(n_components=no_pcs)
    standard_scaler = preprocessing.StandardScaler()

    latent_scaled = standard_scaler.fit_transform(latent_all.cpu())
    latent_projected = pca.fit_transform(latent_scaled)

    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.savefig(PCA_PATH)
    plt.close()

    # 3D

    # DIMS_PCA = (0, 1, 2)
    # fig = plt.figure(figsize=(9, 6))
    # ax = plt.axes(projection='3d')

    # train_embedded = latent_projected[np.random.choice(train_val_idx[0], 1000)]
    # test_embedded = latent_projected[np.random.choice(test_idx[0], 1000)]

    # ax.scatter3D(
    #     train_embedded[:, DIMS_PCA[0]],
    #     train_embedded[:, DIMS_PCA[1]],
    #     train_embedded[:, DIMS_PCA[2]],
    #     color='tab:red',
    #     label='train_val',
    # )

    # ax.scatter3D(
    #     test_embedded[:, DIMS_PCA[0]],
    #     test_embedded[:, DIMS_PCA[1]],
    #     test_embedded[:, DIMS_PCA[2]],
    #     color='tab:green',
    #     label='test',
    # )

    # ax.set_xlabel(f"PC {DIMS_PCA[0]}")
    # ax.set_ylabel(f"PC {DIMS_PCA[1]}")
    # ax.set_zlabel(f"PC {DIMS_PCA[2]}")
    # fig.legend()
    # plt.savefig(GRAPH_3D_PATH)

    # Other plots

    class_subsample_idx = create_subsample(cell_types)
    # batch_subsample_idx = create_subsample(batches)
    latent_proj_sub = latent_projected[class_subsample_idx, :]

    class_palette = create_color_palette(cell_types.unique())
    batches_palette = create_color_palette(batches.unique())
    sites_palette = create_color_palette(sites.unique())
    donors_palette = create_color_palette(donors.unique())
    new_patient_palette = {'train': 'tab:red', 'new patient': 'tab:green'}

    train_idx = np.where(batches[class_subsample_idx] != TEST_BATCH)
    test_idx = np.where(batches[class_subsample_idx] == TEST_BATCH)

    special_labels_indices = [('train', train_idx), ('new patient', test_idx)]
    cell_labels_indices = [(label, np.where(cell_types[class_subsample_idx] == label)) for label in cell_types.unique()]
    batch_labels_indices = [(label, np.where(batches[class_subsample_idx] == label)) for label in batches.unique()]
    site_labels_indices = [(label, np.where(sites[class_subsample_idx] == label)) for label in sites.unique()]
    donor_labels_indices = [(label, np.where(donors[class_subsample_idx] == label)) for label in donors.unique()]

    for dim1, dim2 in zip(range(0, no_pcs - 1), range(1, no_pcs)):
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(30, 10), ncols=3, sharex=True, sharey=True)
        x = latent_proj_sub[:, dim1]
        y = latent_proj_sub[:, dim2]

        plot_labels(ax1, x, y, special_labels_indices, "New patient", color_palette=new_patient_palette)
        plot_labels(ax2, x, y, cell_labels_indices, "Cell types",  color_palette=class_palette, legend=False)
        plot_labels(ax3, x, y, batch_labels_indices, "Batches", color_palette=batches_palette)

        plt.tight_layout()

        data_file = os.path.join(NEW_PATIENT_DIR, f'{dim1}_{dim2}_pcs.png')
        plt.savefig(data_file)
        plt.close()

    for dim1, dim2 in zip(range(0, no_pcs - 1), range(1, no_pcs)):
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(30, 10), ncols=3, sharex=True, sharey=True)
        x = latent_proj_sub[:, dim1]
        y = latent_proj_sub[:, dim2]

        plot_labels(ax1, x, y, batch_labels_indices, "Batches", color_palette=batches_palette)
        plot_labels(ax2, x, y, site_labels_indices, "Sites", color_palette=sites_palette)
        plot_labels(ax3, x, y, donor_labels_indices, "Donors", color_palette=donors_palette)

        plt.tight_layout()

        data_file = os.path.join(BSD_DIR, f'{dim1}_{dim2}_pcs.png')
        plt.savefig(data_file)
        plt.close()

    exit()
    gmm_log_prob = get_log_prob(model, latent_all)
    class_probs = get_class_probs(model, latent_all)
    # found_components = np.argmax(class_probs, axis=1)
    # found_classes = label_encoder.inverse_transform(found_components)
    entropies = entropy([class_prob[label_encoder.transform(label_encoder.classes_)] for class_prob in class_probs], axis=1)

    gex_mse, adt_mse = get_mse(model, latent_all, X_gex, X_adt)
    gex_mse_mean = np.mean(gex_mse, axis=1)
    adt_mse_mean = np.mean(adt_mse, axis=1)

    for dim1, dim2 in zip(range(0, no_pcs-1), range(1, no_pcs)):

        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(30, 10), ncols=3, sharex=True, sharey=True)

        x = latent_proj_sub[:, dim1]
        y = latent_proj_sub[:, dim2]
        plot_labels(ax1, x, y, cell_labels_indices, "Cell types",  color_palette=class_palette, legend=False)

        x = latent_projected[:, dim1]
        y = latent_projected[:, dim2]
        plot_contour(fig, ax2, x, y, gmm_log_prob, 'GMM log prob')
        plot_contour(fig, ax3, x, y, entropies, 'Entropy')

        plt.tight_layout()
        data_file = os.path.join(LOG_PROB_ENTR, f'{dim1}_{dim2}_pcs.png')
        plt.savefig(data_file)
        plt.close()

    for dim1, dim2 in zip(range(0, no_pcs-1), range(1, no_pcs)):
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(30, 10), ncols=3, sharex=True, sharey=True)

        x = latent_proj_sub[:, dim1]
        y = latent_proj_sub[:, dim2]
        plot_labels(ax1, x, y, cell_labels_indices, "Cell types",  color_palette=class_palette, legend=False)

        x = latent_projected[:, dim1]
        y = latent_projected[:, dim2]
        plot_contour(fig, ax2, x, y, gex_mse_mean, 'GEX MSE')
        plot_contour(fig, ax3, x, y, adt_mse_mean, 'ADT MSE')

        plt.tight_layout()
        data_file = os.path.join(GEX_ADT_MSE, f'{dim1}_{dim2}_pcs.png')
        plt.savefig(data_file)
        plt.close()

    df = pd.DataFrame.from_dict({
        'batch': np.array(batches),
        'donor': np.array(donors),
        'site': np.array(sites),
        'gex_mse': gex_mse_mean,
        'adt_mse': adt_mse_mean
    })

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15, 8), ncols=3)
    sns.violinplot(y=df.gex_mse, x=df.site, ax=ax1)
    sns.violinplot(y=df.gex_mse, x=df.donor, ax=ax2)
    sns.violinplot(y=df.gex_mse, x=df.batch, ax=ax3)
    ax1.tick_params(axis='x', rotation=90)
    ax2.tick_params(axis='x', rotation=90)
    ax3.tick_params(axis='x', rotation=90)
    data_file = os.path.join(MSE_PLOTS, 'gex_site_donor_batch.png')
    plt.savefig(data_file)
    plt.close()

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15, 8), ncols=3)
    sns.violinplot(y=df.adt_mse, x=df.site, ax=ax1)
    sns.violinplot(y=df.adt_mse, x=df.donor, ax=ax2)
    sns.violinplot(y=df.adt_mse, x=df.batch, ax=ax3)
    ax1.tick_params(axis='x', rotation=90)
    ax2.tick_params(axis='x', rotation=90)
    ax3.tick_params(axis='x', rotation=90)
    data_file = os.path.join(MSE_PLOTS, 'adt_site_donor_batch.png')
    plt.savefig(data_file)
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.violinplot(y=df.gex_mse, x=df.donor, hue=df.site, linewidth=0.5)
    data_file = os.path.join(MSE_PLOTS, 'gex_donor_sites.png')
    plt.savefig(data_file)
    plt.close()

    sns.jointplot(y=df.gex_mse, x=df.adt_mse, hue=df.donor, height=10, ratio=5)
    data_file = os.path.join(MSE_PLOTS, 'gex_adt_donor.png')
    plt.savefig(data_file)
    plt.close()

    sns.jointplot(y=np.sqrt(df.gex_mse), x=np.sqrt(df.adt_mse), hue=df.site, height=10, ratio=5, kind="kde")
    data_file = os.path.join(MSE_PLOTS, 'gex_adt_site.png')
    plt.savefig(data_file)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GMMVAE in scDNAseq')
    parser.add_argument('--model', type=str, default="gmmvae_better_classifier", help='Model name')
    parser.add_argument('--output', type=str, default="final", help='Name of the output file')
    parser.add_argument('--linear', type=str, default="batch", help='Linear type')
    args = parser.parse_args()

    if args.linear not in ("batch, site, donor, cell"):
        raise ValueError("Invalid linear type specified.")

    main(args.model, args.output, args.linear)
