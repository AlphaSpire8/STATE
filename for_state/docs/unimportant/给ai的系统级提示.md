任务：训练st模型

我现在希望训练st模型。接下来，我会将关键代码文件发给你.

1./data3/fanpeishan/state/src/state/_cli/_tx/_train.py

```python
import argparse as ap

from omegaconf import DictConfig, OmegaConf


def add_arguments_train(parser: ap.ArgumentParser):
    # Allow remaining args to be passed through to Hydra
    parser.add_argument("hydra_overrides", nargs="*", help="Hydra configuration overrides (e.g., data.batch_size=32)")
    # Add custom help handler
    parser.add_argument("--help", "-h", action="store_true", help="Show configuration help with all parameters")


def run_tx_train(cfg: DictConfig):
    import json
    import logging
    import os
    import pickle
    import shutil
    from os.path import exists, join
    from pathlib import Path

    import lightning.pytorch as pl
    import torch
    from cell_load.data_modules import PerturbationDataModule
    from cell_load.utils.modules import get_datamodule
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.plugins.precision import MixedPrecision

    from ...tx.callbacks import (
        BatchSpeedMonitorCallback,
        CumulativeFLOPSCallback,
        GradNormCallback,
        ModelFLOPSUtilizationCallback,
    )
    from ...tx.utils import get_checkpoint_callbacks, get_lightning_module, get_loggers

    logger = logging.getLogger(__name__)
    torch.set_float32_matmul_precision("medium")

    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Setup output directory
    run_output_dir = join(cfg["output_dir"], cfg["name"])
    if os.path.exists(run_output_dir) and cfg["overwrite"]:
        print(f"Output dir {run_output_dir} already exists, overwriting")
        shutil.rmtree(run_output_dir)
    os.makedirs(run_output_dir, exist_ok=True)

    # Set up wandb directory if needed
    if cfg["use_wandb"]:
        os.makedirs(cfg["wandb"]["local_wandb_dir"], exist_ok=True)

    with open(join(run_output_dir, "config.yaml"), "w") as f:
        f.write(cfg_yaml)

    # Set random seeds
    pl.seed_everything(cfg["training"]["train_seed"])

    # if the provided pert_col is drugname_drugconc, hard code the value of control pert
    # this is because it's surprisingly hard to specify a list of tuples in the config as a string
    if cfg["data"]["kwargs"]["pert_col"] == "drugname_drugconc":
        cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"

    # Initialize data module. this is backwards compatible with previous configs
    try:
        sentence_len = cfg["model"]["cell_set_len"]
    except KeyError:
        if cfg["model"]["name"].lower() in ["cpa", "scvi"] or cfg["model"]["name"].lower().startswith("scgpt"):
            if "cell_sentence_len" in cfg["model"]["kwargs"] and cfg["model"]["kwargs"]["cell_sentence_len"] > 1:
                sentence_len = cfg["model"]["kwargs"]["cell_sentence_len"]
                cfg["training"]["batch_size"] = 1
            else:
                sentence_len = 1
        else:
            try:
                sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["n_positions"]
            except:
                sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["max_position_embeddings"]

    if cfg["model"]["name"].lower().startswith("scgpt"):  # scGPT uses log-normalized expression
        cfg["data"]["kwargs"]["transform"] = "log-normalize"
        cfg["data"]["kwargs"]["hvg_names_uns_key"] = (
            "hvg_names" if cfg["data"]["kwargs"]["train_task"] != "replogle" else None
        )  # TODO: better to not hardcode this

        cfg["data"]["kwargs"]["dataset_cls"] = "scGPTPerturbationDataset"

        model_dir = Path(cfg["model"]["kwargs"]["pretrained_path"])

        vocab_file = model_dir / "vocab.json"

        vocab = json.load(open(vocab_file, "r"))
        cfg["model"]["kwargs"]["pad_token_id"] = vocab["<pad>"]
        for s in cfg["model"]["kwargs"]["special_tokens"]:
            if s not in vocab:
                vocab[s] = len(vocab)

        cfg["data"]["kwargs"]["vocab"] = vocab
        cfg["data"]["kwargs"]["perturbation_type"] = cfg["model"]["kwargs"]["perturbation_type"]
        cfg["model"]["kwargs"]["ntoken"] = len(vocab)
        cfg["model"]["kwargs"]["d_model"] = cfg["model"]["kwargs"]["embsize"]

        logger.info("Added vocab and hvg_names_uns_key to data kwargs for scGPT")

    elif cfg["model"]["name"].lower() == "cpa" and cfg["model"]["kwargs"]["recon_loss"] == "gauss":
        cfg["data"]["kwargs"]["transform"] = "log-normalize"
    elif cfg["model"]["name"].lower() == "scvi":
        cfg["data"]["kwargs"]["transform"] = None
    
    print(f"DEBUG: Batch Size passed to DataModule is: {cfg['training']['batch_size']}")

    data_module: PerturbationDataModule = get_datamodule(
        cfg["data"]["name"],
        cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=sentence_len,
    )

    with open(join(run_output_dir, "data_module.torch"), "wb") as f:
        # TODO-Abhi: only save necessary data
        data_module.save_state(f)

    data_module.setup(stage="fit")
    dl = data_module.train_dataloader()
    print("num_workers:", dl.num_workers)
    print("batch size:", dl.batch_size)

    var_dims = data_module.get_var_dims()  # {"gene_dim": …, "hvg_dim": …}
    if cfg["data"]["kwargs"]["output_space"] == "gene":
        gene_dim = var_dims.get("hvg_dim", 2000)  # fallback if key missing
    else:
        gene_dim = var_dims.get("gene_dim", 2000)  # fallback if key missing
    latent_dim = var_dims["output_dim"]  # same as model.output_dim
    hidden_dims = cfg["model"]["kwargs"].get("decoder_hidden_dims", [1024, 1024, 512])

    decoder_cfg = dict(
        latent_dim=latent_dim,
        gene_dim=gene_dim,
        hidden_dims=hidden_dims,
        dropout=cfg["model"]["kwargs"].get("decoder_dropout", 0.1),
        residual_decoder=cfg["model"]["kwargs"].get("residual_decoder", False),
    )

    # tuck it into the kwargs that will reach the LightningModule
    cfg["model"]["kwargs"]["decoder_cfg"] = decoder_cfg

    # Save the onehot maps as pickle files instead of storing in config
    cell_type_onehot_map_path = join(run_output_dir, "cell_type_onehot_map.pkl")
    pert_onehot_map_path = join(run_output_dir, "pert_onehot_map.pt")
    batch_onehot_map_path = join(run_output_dir, "batch_onehot_map.pkl")
    var_dims_path = join(run_output_dir, "var_dims.pkl")

    with open(cell_type_onehot_map_path, "wb") as f:
        pickle.dump(data_module.cell_type_onehot_map, f)
    torch.save(data_module.pert_onehot_map, pert_onehot_map_path)
    with open(batch_onehot_map_path, "wb") as f:
        pickle.dump(data_module.batch_onehot_map, f)
    with open(var_dims_path, "wb") as f:
        pickle.dump(var_dims, f)

    if cfg["model"]["name"].lower() in ["cpa", "scvi"] or cfg["model"]["name"].lower().startswith("scgpt"):
        cfg["model"]["kwargs"]["n_cell_types"] = len(data_module.celltype_onehot_map)
        cfg["model"]["kwargs"]["n_perts"] = len(data_module.pert_onehot_map)
        cfg["model"]["kwargs"]["n_batches"] = len(data_module.batch_onehot_map)

    # Create model
    model = get_lightning_module(
        cfg["model"]["name"],
        cfg["data"]["kwargs"],
        cfg["model"]["kwargs"],
        cfg["training"],
        data_module.get_var_dims(),
    )

    print(
        f"Model created. Estimated params size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3:.2f} GB"
    )
    loggers = get_loggers(
        output_dir=cfg["output_dir"],
        name=cfg["name"],
        wandb_project=cfg["wandb"]["project"],
        wandb_entity=cfg["wandb"]["entity"],
        local_wandb_dir=cfg["wandb"]["local_wandb_dir"],
        use_wandb=cfg["use_wandb"],
        cfg=cfg,
    )

    # If using wandb, store the run path in a text file for eval
    # that matches the old train_lightning.py logic
    for lg in loggers:
        if isinstance(lg, WandbLogger):
            wandb_info_path = os.path.join(run_output_dir, "wandb_path.txt")
            with open(wandb_info_path, "w") as f:
                f.write(lg.experiment.path)
            break

    # Set up callbacks
    ckpt_callbacks = get_checkpoint_callbacks(
        cfg["output_dir"],
        cfg["name"],
        cfg["training"]["val_freq"],
        cfg["training"].get("ckpt_every_n_steps", 4000),
    )
    # Add BatchSpeedMonitorCallback to log batches per second to wandb
    batch_speed_monitor = BatchSpeedMonitorCallback()

    callbacks = ckpt_callbacks + [batch_speed_monitor]

    # Track gradient norm only for state transition model
    if cfg["model"]["name"] == "state":
        callbacks.append(GradNormCallback())

    # Add ModelFLOPSUtilizationCallback to track and log MFU. currently only works for state transition model
    if cfg["training"]["use_mfu"] and cfg["model"]["name"] == "state":
        mfu_available_flops = cfg["training"]["mfu_kwargs"]["available_flops"]
        mfu_use_backward = cfg["training"]["mfu_kwargs"]["use_backward"]
        mfu_logging_interval = cfg["training"]["mfu_kwargs"]["logging_interval"]
        mfu_window_size = cfg["training"]["mfu_kwargs"]["window_size"]
        mfu_cb = ModelFLOPSUtilizationCallback(
            available_flops=mfu_available_flops,
            use_backward=mfu_use_backward,
            logging_interval=mfu_logging_interval,
            cell_set_len=cfg["model"]["kwargs"]["cell_set_len"],
            window_size=mfu_window_size,
        )

        callbacks.append(mfu_cb)

        # Add CumulativeFLOPSCallback to track cumulative FLOPs
        cumulative_flops_use_backward = cfg["training"]["cumulative_flops_use_backward"]
        cumulative_flops_cb = CumulativeFLOPSCallback(use_backward=cumulative_flops_use_backward)
        callbacks.append(cumulative_flops_cb)

    logger.info("Loggers and callbacks set up.")

    if cfg["model"]["name"].lower().startswith("scgpt"):
        plugins = [
            MixedPrecision(
                precision="bf16-mixed",
                device="cuda",
            )
        ]
    else:
        plugins = []

    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Decide on trainer params
    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=cfg["training"].get("devices", 1),
        strategy=cfg["training"].get("strategy", "auto"),
        max_steps=cfg["training"]["max_steps"],  # for normal models
        check_val_every_n_epoch=None,
        val_check_interval=cfg["training"]["val_freq"],
        logger=loggers,
        plugins=plugins,
        callbacks=callbacks,
        gradient_clip_val=cfg["training"]["gradient_clip_val"] if cfg["model"]["name"].lower() != "cpa" else None,
        use_distributed_sampler=False,  # Prevent Lightning from wrapping PerturbationBatchSampler with DistributedSampler
    )

    # Align logging cadence with rolling MFU window (and W&B logging)
    if "log_every_n_steps" in cfg["training"]:
        trainer_kwargs["log_every_n_steps"] = cfg["training"]["log_every_n_steps"]


    # Build trainer
    print(f"Building trainer with kwargs: {trainer_kwargs}")
    trainer = pl.Trainer(**trainer_kwargs)
    print("Trainer built successfully")

    # Load checkpoint if exists
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "last.ckpt")
    if not exists(checkpoint_path):
        checkpoint_path = None
    else:
        logging.info(f"!! Resuming training from {checkpoint_path} !!")

    print(f"Model device: {next(model.parameters()).device}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    logger.info("Starting trainer fit.")

    # if a checkpoint does not exist, start with the provided checkpoint
    # this is mainly used for pretrain -> finetune workflows
    manual_init = cfg["model"]["kwargs"].get("init_from", None)
    if checkpoint_path is None and manual_init is not None:
        print(f"Loading manual checkpoint from {manual_init}")
        checkpoint_path = manual_init
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = model.state_dict()
        checkpoint_state = checkpoint["state_dict"]

        # Check if output_space differs between current config and checkpoint
        checkpoint_output_space = checkpoint.get("hyper_parameters", {}).get("output_space", "gene")
        current_output_space = cfg["data"]["kwargs"]["output_space"]

        if checkpoint_output_space != current_output_space:
            print(
                f"Output space mismatch: checkpoint has '{checkpoint_output_space}', current config has '{current_output_space}'"
            )
            print("Creating new decoder for the specified output space...")

            if cfg["model"]["kwargs"].get("gene_decoder_bool", True) == False:
                model._decoder_externally_configured = False
            else:
                # Override the decoder_cfg to match the new output_space
                if current_output_space == "gene":
                    new_gene_dim = var_dims.get("hvg_dim", 2000)
                else:  # output_space == "all"
                    new_gene_dim = var_dims.get("gene_dim", 2000)

                new_decoder_cfg = dict(
                    latent_dim=var_dims["output_dim"],
                    gene_dim=new_gene_dim,
                    hidden_dims=cfg["model"]["kwargs"].get("decoder_hidden_dims", [1024, 1024, 512]),
                    dropout=cfg["model"]["kwargs"].get("decoder_dropout", 0.1),
                    residual_decoder=cfg["model"]["kwargs"].get("residual_decoder", False),
                )

                # Update the model's decoder_cfg and rebuild decoder
                model.decoder_cfg = new_decoder_cfg
                model._build_decoder()
                model._decoder_externally_configured = True  # Mark that decoder was configured externally
                print(f"Created new decoder for output_space='{current_output_space}' with gene_dim={new_gene_dim}")

        pert_encoder_weight_key = "pert_encoder.0.weight"
        if pert_encoder_weight_key in checkpoint_state:
            checkpoint_pert_dim = checkpoint_state[pert_encoder_weight_key].shape[1]
            if checkpoint_pert_dim != model.pert_dim:
                print(
                    f"pert_encoder input dimension mismatch: model.pert_dim = {model.pert_dim} but checkpoint expects {checkpoint_pert_dim}. Overriding model's pert_dim and rebuilding pert_encoder."
                )
                # Rebuild the pert_encoder with the new pert input dimension
                from ...tx.models.utils import build_mlp

                model.pert_encoder = build_mlp(
                    in_dim=model.pert_dim,
                    out_dim=model.hidden_dim,
                    hidden_dim=model.hidden_dim,
                    n_layers=model.n_encoder_layers,
                    dropout=model.dropout,
                    activation=model.activation_class,
                )
            else:
                print("WARNING: pert_encoder will not be rebuilt since input dimension matches")

        # Filter out mismatched size parameters
        filtered_state = {}
        for name, param in checkpoint_state.items():
            if name in model_state:
                if param.shape == model_state[name].shape:
                    filtered_state[name] = param
                else:
                    print(
                        f"Skipping parameter {name} due to shape mismatch: checkpoint={param.shape}, model={model_state[name].shape}"
                    )
            else:
                print(f"Skipping parameter {name} as it doesn't exist in the current model")

        # Load the filtered state dict
        model.load_state_dict(filtered_state, strict=False)
        print("About to call trainer.fit() with manual checkpoint...")

        # Train - for clarity we pass None
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=None,
        )
        print("trainer.fit() completed with manual checkpoint")
    else:
        print(f"About to call trainer.fit() with checkpoint_path={checkpoint_path}")
        # Train
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )
        print("trainer.fit() completed")

    print("Training completed, saving final checkpoint...")

    # at this point if checkpoint_path does not exist, manually create one
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "final.ckpt")
    if not exists(checkpoint_path):
        trainer.save_checkpoint(checkpoint_path)

```

2./data3/fanpeishan/state/.venv/lib/python3.11/site-packages/cell_load/data_modules/perturbation_dataloader.py

```python
import logging
import glob
import re

from functools import partial
from pathlib import Path
from typing import Literal, Set, Dict

import h5py
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..config import ExperimentConfig
from ..dataset import MetadataConcatDataset, PerturbationDataset
from ..mapping_strategies import BatchMappingStrategy, RandomMappingStrategy
from ..utils.data_utils import (
    GlobalH5MetadataCache,
    generate_onehot_map,
    safe_decode_array,
)
from .samplers import PerturbationBatchSampler

logger = logging.getLogger(__name__)


class PerturbationDataModule(LightningDataModule):
    """
    A unified data module that sets up train/val/test splits for multiple dataset/celltype
    combos. Allows zero-shot, few-shot tasks, and uses a pluggable mapping strategy
    (batch, random, nearest) to match perturbed cells with control cells.
    """

    def __init__(
        self,
        toml_config_path: str,
        batch_size: int = 128,
        num_workers: int = 8,
        random_seed: int = 42,  # this should be removed by seed everything
        pert_col: str = "gene",
        batch_col: str = "gem_group",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        embed_key: Literal["X_hvg", "X_state"] | None = None,
        output_space: Literal["gene", "all", "embedding"] = "gene",
        basal_mapping_strategy: Literal["batch", "random"] = "random",
        n_basal_samples: int = 1,
        should_yield_control_cells: bool = True,
        cell_sentence_len: int = 512,
        cache_perturbation_control_pairs: bool = False,
        drop_last: bool = False,
        **kwargs,  # missing perturbation_features_file  and store_raw_basal for backwards compatibility
    ):
        """
        This class is responsible for serving multiple PerturbationDataset's each of which is specific
        to a dataset/cell type combo. It sets up training, validation, and test splits for each dataset
        and cell type, and uses a pluggable mapping strategy to match perturbed cells with control cells.

        Args:
            toml_config_path: Path to TOML configuration file
            batch_size: Batch size for PyTorch DataLoader
            num_workers: Num workers for PyTorch DataLoader
            few_shot_percent: Fraction of data to use for few-shot tasks
            random_seed: For reproducible splits & sampling
            embed_key: Embedding key or matrix in the H5 file to use for feauturizing cells
            output_space: The output space for model predictions (gene, all genes, or embedding-only)
            basal_mapping_strategy: One of {"batch","random","nearest","ot"}
            n_basal_samples: Number of control cells to sample per perturbed cell
            cache_perturbation_control_pairs: If True cache perturbation-control pairs at the start of training and reuse them.
            drop_last: Whether to drop the last sentence set if it is smaller than cell_sentence_len
        """
        super().__init__()

        # Load and validate configuration
        self.toml_config_path = toml_config_path
        self.config = ExperimentConfig.from_toml(toml_config_path)
        self.config.validate()

        # Experiment level params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.drop_last = drop_last

        # H5 field names
        self.pert_col = pert_col
        self.batch_col = batch_col
        self.cell_type_key = cell_type_key
        self.control_pert = control_pert
        self.embed_key = embed_key
        self.output_space = output_space
        if self.output_space not in {"gene", "all", "embedding"}:
            raise ValueError(
                f"output_space must be one of 'gene', 'all', or 'embedding'; got {self.output_space!r}"
            )

        # Sampling and mapping
        self.n_basal_samples = n_basal_samples
        self.cell_sentence_len = cell_sentence_len
        self.should_yield_control_cells = should_yield_control_cells
        self.cache_perturbation_control_pairs = cache_perturbation_control_pairs

        # Optional behaviors
        self.map_controls = kwargs.get("map_controls", True)
        self.perturbation_features_file = kwargs.get("perturbation_features_file")
        self.int_counts = kwargs.get("int_counts", False)
        self.normalize_counts = kwargs.get("normalize_counts", False)
        self.store_raw_basal = kwargs.get("store_raw_basal", False)
        self.barcode = kwargs.get("barcode", False)

        logger.info(
            f"Initializing DataModule: batch_size={batch_size}, workers={num_workers}, "
            f"random_seed={random_seed}"
        )

        # Mapping strategy
        self.basal_mapping_strategy = basal_mapping_strategy
        self.mapping_strategy_cls = {
            "batch": BatchMappingStrategy,
            "random": RandomMappingStrategy,
        }[basal_mapping_strategy]

        # Determine if raw expression is needed
        self.store_raw_expression = bool(
            self.embed_key
            and (
                (self.embed_key != "X_hvg" and self.output_space == "gene")
                or self.output_space == "all"
            )
        )

        # Prepare dataset lists and maps
        self.train_datasets: list[Dataset] = []
        self.val_datasets: list[Dataset] = []
        self.test_datasets: list[Dataset] = []

        self.all_perts: Set[str] = set()
        self.pert_onehot_map: dict[str, torch.Tensor] | None = None
        self.batch_onehot_map: dict[str, torch.Tensor] | None = None
        self.cell_type_onehot_map: dict[str, torch.Tensor] | None = None

        # Initialize global maps
        self._setup_global_maps()

    def _get_reference_dataset(self) -> PerturbationDataset:
        """Return a dataset to read metadata from, preferring test → val → train."""
        for datasets in (self.test_datasets, self.val_datasets, self.train_datasets):
            if datasets:
                return datasets[0].dataset
        raise ValueError("No datasets available to extract metadata.")

    def get_var_names(self):
        """
        Get the variable names (gene names) from the first available dataset.
        This assumes all datasets have the same gene names.
        """
        underlying_ds = self._get_reference_dataset()
        return underlying_ds.get_gene_names(output_space=self.output_space)

    def setup(self, stage: str | None = None):
        """
        Set up training and test datasets.
        """
        if len(self.train_datasets) == 0:
            self._setup_datasets()
            logger.info(
                "Done! Train / Val / Test splits: %d / %d / %d",
                len(self.train_datasets),
                len(self.val_datasets),
                len(self.test_datasets),
            )

    def save_state(self, filepath: str):
        """
        Save the data module configuration to a torch file.
        This saves only the initialization parameters, not the computed splits for the datasets.

        Args:
            filepath: Path where to save the configuration (should end with .torch)
        """
        save_dict = {
            "toml_config_path": self.toml_config_path,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "random_seed": self.random_seed,
            "pert_col": self.pert_col,
            "batch_col": self.batch_col,
            "cell_type_key": self.cell_type_key,
            "control_pert": self.control_pert,
            "embed_key": self.embed_key,
            "output_space": self.output_space,
            "basal_mapping_strategy": self.basal_mapping_strategy,
            "n_basal_samples": self.n_basal_samples,
            "should_yield_control_cells": self.should_yield_control_cells,
            "cell_sentence_len": self.cell_sentence_len,
            "cache_perturbation_control_pairs": self.cache_perturbation_control_pairs,
            # Include the optional behaviors
            "map_controls": self.map_controls,
            "perturbation_features_file": self.perturbation_features_file,
            "int_counts": self.int_counts,
            "normalize_counts": self.normalize_counts,
            "store_raw_basal": self.store_raw_basal,
            "barcode": self.barcode,
        }

        torch.save(save_dict, filepath)
        logger.info(f"Saved data module configuration to {filepath}")

    @classmethod
    def load_state(cls, filepath: str):
        """
        Load a data module from a saved torch file.
        This reconstructs the data module with the original initialization parameters.
        You will need to call setup() after loading to recreate the datasets.

        Args:
            filepath: Path to the saved configuration file

        Returns:
            PerturbationDataModule: A new instance with the saved configuration
        """
        save_dict = torch.load(filepath, map_location="cpu")
        logger.info(f"Loaded data module configuration from {filepath}")

        # Validate that the toml config file still exists
        toml_path = Path(save_dict["toml_config_path"])
        if not toml_path.exists():
            logger.warning(
                f"TOML config file not found at {toml_path}. "
                "Make sure the file exists or the path is correct."
            )

        # Extract the kwargs that were passed to __init__
        kwargs = {
            "map_controls": save_dict.pop("map_controls", True),
            "cache_perturbation_control_pairs": save_dict.pop(
                "cache_perturbation_control_pairs", False
            ),
            "perturbation_features_file": save_dict.pop(
                "perturbation_features_file", None
            ),
            "int_counts": save_dict.pop("int_counts", False),
            "normalize_counts": save_dict.pop("normalize_counts", False),
            "store_raw_basal": save_dict.pop("store_raw_basal", False),
            "barcode": save_dict.pop("barcode", True),
        }

        # Create new instance with all the saved parameters
        return cls(**save_dict, **kwargs)

    def get_var_dims(self):
        underlying_ds = self._get_reference_dataset()
        if self.embed_key:
            input_dim = underlying_ds.get_dim_for_obsm(self.embed_key)
        else:
            input_dim = underlying_ds.n_genes

        gene_dim = underlying_ds.n_genes
        try:
            hvg_dim = underlying_ds.get_num_hvgs()
        except AttributeError:
            assert self.embed_key is None, "No X_hvg detected, using raw .X"
            hvg_dim = gene_dim

        if self.embed_key is not None:
            output_dim = underlying_ds.get_dim_for_obsm(self.embed_key)
        else:
            output_dim = input_dim  # training on raw gene expression

        gene_names = underlying_ds.get_gene_names(output_space=self.output_space)

        # get the shape of the first value in pert_onehot_map
        pert_dim = next(iter(self.pert_onehot_map.values())).shape[0]
        batch_dim = next(iter(self.batch_onehot_map.values())).shape[0]

        pert_names = list(self.pert_onehot_map.keys())

        return {
            "input_dim": input_dim,
            "gene_dim": gene_dim,
            "hvg_dim": hvg_dim,
            "output_dim": output_dim,
            "pert_dim": pert_dim,
            "gene_names": gene_names,
            "batch_dim": batch_dim,
            "pert_names": pert_names,
        }

    def get_shared_perturbations(self) -> Set[str]:
        """
        Compute shared perturbations between train and test sets by inspecting
        only the actual subset indices in self.train_datasets and self.test_datasets.

        This ensures we don't accidentally include all perturbations from the entire h5 file.
        """

        def _extract_perts_from_subset(subset) -> Set[str]:
            """
            Helper that returns the set of perturbation names for the
            exact subset indices in 'subset'.
            """
            ds = subset.dataset  # The underlying PerturbationDataset
            idxs = subset.indices  # The subset of row indices relevant to this Subset

            # ds.pert_col typically is 'gene' or similar
            pert_codes = ds.metadata_cache.pert_codes[idxs]
            # Convert each code to its corresponding string label
            pert_names = ds.pert_categories[pert_codes]

            return set(pert_names)

        # 1) Gather all perturbations found across the *actual training subsets*
        train_perts = set()
        for subset in self.train_datasets:
            train_perts.update(_extract_perts_from_subset(subset))

        # 2) Gather all perturbations found across the *actual testing subsets*
        test_perts = set()
        for subset in self.test_datasets:
            test_perts.update(_extract_perts_from_subset(subset))

        # 3) Intersection = shared across both train and test
        shared_perts = train_perts & test_perts

        logger.info(f"Found {len(train_perts)} distinct perts in the train subsets.")
        logger.info(f"Found {len(test_perts)} distinct perts in the test subsets.")
        logger.info(f"Found {len(shared_perts)} shared perturbations (train ∩ test).")

        return shared_perts

    def get_control_pert(self):
        # Return the control perturbation name
        return self.train_datasets[0].dataset.control_pert

    def train_dataloader(self, test=False):
        if len(self.train_datasets) == 0:
            raise ValueError(
                "No training datasets available. Please call setup() first."
            )
        return self._create_dataloader(self.train_datasets, test=test)

    def val_dataloader(self):
        if len(self.val_datasets) == 0:
            if len(self.test_datasets) == 0:
                return []
            return self._create_dataloader(self.test_datasets, test=False)
        return self._create_dataloader(self.val_datasets, test=False)

    def test_dataloader(self):
        if len(self.test_datasets) == 0:
            return []
        return self._create_dataloader(self.test_datasets, test=True, batch_size=1)

    def predict_dataloader(self):
        if len(self.test_datasets) == 0:
            return []
        return self._create_dataloader(self.test_datasets, test=True)

    # Helper functions to set up global maps and datasets

    def _create_dataloader(
        self,
        datasets: list[Dataset],
        test: bool = False,
        batch_size: int | None = None,
    ):
        """Create a DataLoader with appropriate configuration."""
        use_int_counts = "int_counts" in self.__dict__ and self.int_counts
        collate_fn = partial(PerturbationDataset.collate_fn, int_counts=use_int_counts)

        ds = MetadataConcatDataset(datasets)
        use_batch = self.basal_mapping_strategy == "batch"

        batch_size = batch_size or (1 if test else self.batch_size)

        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=batch_size,
            drop_last=self.drop_last,
            cell_sentence_len=self.cell_sentence_len,
            test=test,
            use_batch=use_batch,
        )

        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=4 if not test and self.num_workers > 0 else None,
        )

    def _setup_global_maps(self):
        """
        Set up global one-hot maps for perturbations and batches.
        For perturbations, we scan through all files in all train_specs and test_specs.
        """
        all_perts = set()
        all_batches = set()
        all_celltypes = set()

        for dataset_name in self.config.get_all_datasets():
            dataset_path = Path(self.config.datasets[dataset_name])
            files = self._find_dataset_files(dataset_path)

            for _fname, fpath in files.items():
                with h5py.File(fpath, "r") as f:
                    pert_arr = f[f"obs/{self.pert_col}/categories"][:]
                    perts = set(safe_decode_array(pert_arr))
                    all_perts.update(perts)

                    try:
                        batch_arr = f[f"obs/{self.batch_col}/categories"][:]
                    except KeyError:
                        batch_arr = f[f"obs/{self.batch_col}"][:]
                    batches = set(safe_decode_array(batch_arr))
                    all_batches.update(batches)

                    try:
                        celltype_arr = f[f"obs/{self.cell_type_key}/categories"][:]
                    except KeyError:
                        celltype_arr = f[f"obs/{self.cell_type_key}"][:]
                    celltypes = set(safe_decode_array(celltype_arr))
                    all_celltypes.update(celltypes)

        # Create one-hot maps
        if self.perturbation_features_file:
            # Load the custom featurizations from a torch file
            featurization_dict = torch.load(self.perturbation_features_file)
            # Validate that every perturbation in all_perts is in the featurization dict.
            missing = all_perts - set(featurization_dict.keys())
            if len(missing) > 0:
                feature_dim = next(iter(featurization_dict.values())).shape[-1]
                for pert in missing:
                    featurization_dict[pert] = torch.zeros(feature_dim)

                logger.info(
                    "Set %d missing perturbations to zero vectors.", len(missing)
                )

            logger.info(
                "Loaded custom perturbation featurizations for %d perturbations.",
                len(featurization_dict),
            )
            self.pert_onehot_map = featurization_dict  # use the custom featurizations
        else:
            # Fall back to default: generate one-hot mapping
            self.pert_onehot_map = generate_onehot_map(all_perts)

        self.batch_onehot_map = generate_onehot_map(all_batches)
        self.cell_type_onehot_map = generate_onehot_map(all_celltypes)

    def _create_base_dataset(
        self, dataset_name: str, fpath: Path
    ) -> PerturbationDataset:
        """Create a base PerturbationDataset instance."""
        mapping_kwargs = {"map_controls": self.map_controls}

        # Add cache_perturbation_control_pairs to mapping strategy kwargs
        mapping_kwargs["cache_perturbation_control_pairs"] = (
            self.cache_perturbation_control_pairs
        )

        return PerturbationDataset(
            name=dataset_name,
            h5_path=fpath,
            mapping_strategy=self.mapping_strategy_cls(
                random_state=self.random_seed,
                n_basal_samples=self.n_basal_samples,
                **mapping_kwargs,
            ),
            embed_key=self.embed_key,
            pert_onehot_map=self.pert_onehot_map,
            batch_onehot_map=self.batch_onehot_map,
            cell_type_onehot_map=self.cell_type_onehot_map,
            pert_col=self.pert_col,
            cell_type_key=self.cell_type_key,
            batch_col=self.batch_col,
            control_pert=self.control_pert,
            random_state=self.random_seed,
            should_yield_control_cells=self.should_yield_control_cells,
            store_raw_expression=self.store_raw_expression,
            output_space=self.output_space,
            store_raw_basal=self.store_raw_basal,
            barcode=self.barcode,
        )

    def _setup_datasets(self):
        """
        Set up training datasets with proper handling of zeroshot/fewshot splits w/ TOML.
        Uses H5MetadataCache for faster metadata access.
        """

        for dataset_name in self.config.get_all_datasets():
            dataset_path = Path(self.config.datasets[dataset_name])
            files = self._find_dataset_files(dataset_path)

            # Get configuration for this dataset
            zeroshot_celltypes = self.config.get_zeroshot_celltypes(dataset_name)
            fewshot_celltypes = self.config.get_fewshot_celltypes(dataset_name)
            is_training_dataset = self.config.training.get(dataset_name) == "train"

            logger.info(f"Processing dataset {dataset_name}:")
            logger.info(f"  - Training dataset: {is_training_dataset}")
            logger.info(f"  - Zeroshot cell types: {list(zeroshot_celltypes.keys())}")
            logger.info(f"  - Fewshot cell types: {list(fewshot_celltypes.keys())}")

            # Process each file in the dataset
            for fname, fpath in tqdm(
                list(files.items()), desc=f"Processing {dataset_name}"
            ):
                # Create metadata cache
                cache = GlobalH5MetadataCache().get_cache(
                    str(fpath),
                    self.pert_col,
                    self.cell_type_key,
                    self.control_pert,
                    self.batch_col,
                )

                # Create base dataset
                ds = self._create_base_dataset(dataset_name, fpath)
                train_sum = val_sum = test_sum = 0

                # Process each cell type in this file
                for ct_idx, ct in enumerate(cache.cell_type_categories):
                    ct_mask = cache.cell_type_codes == ct_idx
                    n_cells = np.sum(ct_mask)

                    if n_cells == 0:
                        continue

                    ct_indices = np.where(ct_mask)[0]

                    # Split into control and perturbed indices
                    ctrl_mask = cache.pert_codes[ct_indices] == cache.control_pert_code
                    ctrl_indices = ct_indices[ctrl_mask]
                    pert_indices = ct_indices[~ctrl_mask]

                    # Determine how to handle this cell type
                    counts = self._process_celltype(
                        ds,
                        ct,
                        ct_indices,
                        ctrl_indices,
                        pert_indices,
                        cache,
                        dataset_name,
                        zeroshot_celltypes,
                        fewshot_celltypes,
                        is_training_dataset,
                    )

                    train_sum += counts["train"]
                    val_sum += counts["val"]
                    test_sum += counts["test"]

                tqdm.write(
                    f"Processed {fname}: {train_sum} train, {val_sum} val, {test_sum} test"
                )

            logger.info("\n")

    def _split_fewshot_celltype(
        self,
        ds: PerturbationDataset,
        pert_indices: np.ndarray,
        ctrl_indices: np.ndarray,
        cache,
        pert_config: dict[str, list[str]],
    ) -> dict[str, int]:
        """Split a fewshot cell type according to perturbation assignments."""
        counts = {"train": 0, "val": 0, "test": 0}

        # Get perturbation codes for this cell type
        pert_codes = cache.pert_codes[pert_indices]

        # Create sets of perturbation codes for each split
        val_pert_names = set(pert_config.get("val", []))
        test_pert_names = set(pert_config.get("test", []))

        val_pert_codes = set()
        test_pert_codes = set()

        for i, pert_name in enumerate(cache.pert_categories):
            if pert_name in val_pert_names:
                val_pert_codes.add(i)
            if pert_name in test_pert_names:
                test_pert_codes.add(i)

        # Split perturbation indices by their codes
        val_mask = np.isin(pert_codes, list(val_pert_codes))
        test_mask = np.isin(pert_codes, list(test_pert_codes))
        train_mask = ~(val_mask | test_mask)

        val_pert_indices = pert_indices[val_mask]
        test_pert_indices = pert_indices[test_mask]
        train_pert_indices = pert_indices[train_mask]

        # Split controls proportionally
        rng = np.random.default_rng(self.random_seed)
        ctrl_indices_shuffled = rng.permutation(ctrl_indices)

        n_val = len(val_pert_indices)
        n_test = len(test_pert_indices)
        n_train = len(train_pert_indices)
        total_pert = n_val + n_test + n_train

        if total_pert > 0:
            # Create subsets
            if len(val_pert_indices) > 0:
                subset = ds.to_subset_dataset(
                    "val", val_pert_indices, ctrl_indices_shuffled
                )
                self.val_datasets.append(subset)
                counts["val"] = len(subset)

            if len(test_pert_indices) > 0:
                subset = ds.to_subset_dataset(
                    "test", test_pert_indices, ctrl_indices_shuffled
                )
                self.test_datasets.append(subset)
                counts["test"] = len(subset)

            subset = ds.to_subset_dataset(
                "train", train_pert_indices, ctrl_indices_shuffled
            )
            self.train_datasets.append(subset)
            counts["train"] = len(subset)

        return counts

    def _find_dataset_files(self, dataset_path: Path) -> dict[str, Path]:
        files: Dict[str, Path] = {}
        path_str = str(dataset_path)

        # Check if path contains glob patterns
        if any(char in path_str for char in "*?[]{}"):
            # Handle brace expansion manually since Python glob doesn't support it
            expanded_patterns = self._expand_braces(path_str)

            for pattern in expanded_patterns:
                if pattern.startswith("/"):
                    # Absolute path - use glob.glob()
                    if pattern.endswith((".h5", ".h5ad")):
                        # Pattern already specifies file extension
                        for fpath_str in sorted(glob.glob(pattern)):
                            fpath = Path(fpath_str)
                            files[fpath.stem] = fpath
                    else:
                        # Pattern doesn't specify extension, add file patterns
                        for ext in ("*.h5", "*.h5ad"):
                            full_pattern = f"{pattern.rstrip('/')}/{ext}"
                            for fpath_str in sorted(glob.glob(full_pattern)):
                                fpath = Path(fpath_str)
                                files[fpath.stem] = fpath
                else:
                    # Relative path - use Path.glob()
                    if pattern.endswith((".h5", ".h5ad")):
                        for fpath in sorted(Path().glob(pattern)):
                            files[fpath.stem] = fpath
                    else:
                        for ext in ("*.h5", "*.h5ad"):
                            full_pattern = f"{pattern.rstrip('/')}/{ext}"
                            for fpath in sorted(Path().glob(full_pattern)):
                                files[fpath.stem] = fpath
        else:
            # No glob patterns - treat as regular path
            if dataset_path.is_file():
                # Single file
                files[dataset_path.stem] = dataset_path
            else:
                # Directory - search for files
                for ext in ("*.h5", "*.h5ad"):
                    for fpath in sorted(dataset_path.glob(ext)):
                        files[fpath.stem] = fpath

        return files

    def _expand_braces(self, pattern: str) -> list[str]:
        """Expand brace patterns like {a,b,c} into multiple patterns."""

        def expand_single_brace(text: str) -> list[str]:
            # Find the first brace group
            match = re.search(r"\{([^}]+)\}", text)
            if not match:
                return [text]

            # Extract the options and expand them
            before = text[: match.start()]
            after = text[match.end() :]
            options = match.group(1).split(",")

            results = []
            for option in options:
                new_text = before + option.strip() + after
                # Recursively expand any remaining braces
                results.extend(expand_single_brace(new_text))

            return results

        return expand_single_brace(pattern)

    def _process_celltype(
        self,
        ds: PerturbationDataset,
        celltype: str,
        ct_indices: np.ndarray,
        ctrl_indices: np.ndarray,
        pert_indices: np.ndarray,
        cache,
        dataset_name: str,
        zeroshot_celltypes: dict[str, str],
        fewshot_celltypes: dict[str, dict[str, list[str]]],
        is_training_dataset: bool,
    ) -> dict[str, int]:
        """Process a single cell type and return counts for each split."""
        counts = {"train": 0, "val": 0, "test": 0}

        if celltype in zeroshot_celltypes:
            # Zeroshot: all cells go to specified split
            split = zeroshot_celltypes[celltype]
            train_subset = ds.to_subset_dataset(
                "train", np.array([], dtype=np.int64), ctrl_indices
            )  # adding all observational data to train
            test_subset = ds.to_subset_dataset(split, pert_indices, ctrl_indices)
            if split == "train":
                self.train_datasets.append(test_subset)
            elif split == "val":
                self.train_datasets.append(train_subset)
                counts["train"] = len(train_subset)
                self.val_datasets.append(test_subset)
            elif split == "test":
                self.train_datasets.append(train_subset)
                counts["train"] = len(train_subset)
                self.test_datasets.append(test_subset)

            counts[split] = len(test_subset)

        elif celltype in fewshot_celltypes:
            # Fewshot: split perturbations according to config
            pert_config = fewshot_celltypes[celltype]
            split_counts = self._split_fewshot_celltype(
                ds, pert_indices, ctrl_indices, cache, pert_config
            )
            for split, count in split_counts.items():
                counts[split] += count

        elif is_training_dataset:
            # Regular training cell type
            subset = ds.to_subset_dataset("train", pert_indices, ctrl_indices)
            self.train_datasets.append(subset)
            counts["train"] = len(subset)

        return counts

```

3./data3/fanpeishan/state/.venv/lib/python3.11/site-packages/cell_load/utils/modules.py

```python
from ..data_modules import PerturbationDataModule

DATA_MODULE_DICT = dict(
    PerturbationDataModule=PerturbationDataModule,
)


def get_datamodule(name, kwargs, batch_size=None, cell_sentence_len=1):
    """
    Load data/lightning modules using TOML configuration.

    Args:
        name: Name of the data module (e.g., 'PerturbationDataModule')
        kwargs: Dictionary containing 'toml_config_path' and other parameters
        batch_size: Optional batch size override
        cell_sentence_len: Optional cell sentence length override
    """
    if name not in DATA_MODULE_DICT:
        raise ValueError(f"Unknown data module '{name}'")

    if batch_size is not None:
        kwargs["batch_size"] = batch_size
        kwargs["cell_sentence_len"] = cell_sentence_len

    # Ensure toml_config_path is provided
    if "toml_config_path" not in kwargs:
        raise ValueError("toml_config_path must be provided in kwargs")

    return DATA_MODULE_DICT[name](**kwargs)

```

4./data3/fanpeishan/state/src/state/tx/utils/__init__.py

```python
import time
import logging
from contextlib import contextmanager
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger as BaseCSVLogger
import csv
import os
from lightning.pytorch.callbacks import ModelCheckpoint
from os.path import join


class RobustCSVLogger(BaseCSVLogger):
    """
    A CSV logger that handles dynamic metrics by allowing new columns to be added during training.
    This fixes the issue where PyTorch Lightning's default CSV logger fails when new metrics
    are added after the CSV file is created.
    """

    def log_metrics(self, metrics, step):
        """Override to handle dynamic metrics gracefully"""
        try:
            super().log_metrics(metrics, step)
        except ValueError as e:
            if "dict contains fields not in fieldnames" in str(e):
                # Recreate the CSV file with the new fieldnames
                self._recreate_csv_with_new_fields(metrics)
                # Try logging again
                super().log_metrics(metrics, step)
            else:
                raise e

    def _recreate_csv_with_new_fields(self, new_metrics):
        """Recreate the CSV file with additional fields to accommodate new metrics"""
        if not hasattr(self.experiment, "metrics_file_path"):
            return

        # Read existing data
        existing_data = []
        csv_file = self.experiment.metrics_file_path

        if os.path.exists(csv_file):
            with open(csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)

        # Get all unique fieldnames from existing data and new metrics
        all_fieldnames = set()
        for row in existing_data:
            all_fieldnames.update(row.keys())
        all_fieldnames.update(new_metrics.keys())

        # Sort fieldnames for consistent ordering
        sorted_fieldnames = sorted(all_fieldnames)

        # Rewrite the CSV file with new fieldnames
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted_fieldnames)
            writer.writeheader()

            # Write existing data (missing fields will be empty)
            for row in existing_data:
                writer.writerow(row)

        # Update the experiment's fieldnames
        self.experiment.metrics_keys = sorted_fieldnames


@contextmanager
def time_it(timer_name: str):
    logging.debug(f"Starting timer {timer_name}")
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logging.debug(f"Elapsed time {timer_name}: {elapsed_time:.4f} seconds")


def get_loggers(
    output_dir: str,
    name: str,
    wandb_project: str,
    wandb_entity: str,
    local_wandb_dir: str,
    use_wandb: bool = False,
    use_csv: bool = True,  # Enable CSV by default with robust logger
    cfg: dict = None,
):
    """Set up logging to local CSV and optionally WandB."""
    loggers = []

    # Use robust CSV logger that handles dynamic metrics
    if use_csv:
        csv_logger = RobustCSVLogger(save_dir=output_dir, name=name, version=0)
        loggers.append(csv_logger)

    # Add WandB if requested
    if use_wandb:
        try:
            # Check if wandb is available
            import wandb

            wandb_logger = WandbLogger(
                name=name,
                project=wandb_project,
                entity=wandb_entity,
                dir=local_wandb_dir,
                tags=cfg["wandb"].get("tags", []) if cfg else [],
            )
            if cfg is not None:
                wandb_logger.experiment.config.update(cfg)
            loggers.append(wandb_logger)
        except ImportError:
            print("Warning: wandb is not installed. Skipping wandb logging.")
            print("To enable wandb logging, install it with: pip install wandb")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb logger: {e}")
            print("Continuing without wandb logging.")

    # Ensure at least one logger is present
    if not loggers:
        print("Warning: No loggers configured. Adding robust CSV logger as fallback.")
        csv_logger = RobustCSVLogger(save_dir=output_dir, name=name, version=0)
        loggers.append(csv_logger)

    return loggers


def get_checkpoint_callbacks(output_dir: str, name: str, val_freq: int, ckpt_every_n_steps: int):
    """
    Create checkpoint callbacks based on validation frequency.

    Returns a list of callbacks.
    """
    checkpoint_dir = join(output_dir, name, "checkpoints")
    callbacks = []

    # Save best checkpoint based on validation loss
    best_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step={step}-val_loss={val_loss:.4f}",
        save_last="link",  # Will create last.ckpt symlink to best checkpoint
        monitor="val_loss",
        mode="min",
        save_top_k=1,  # Only keep the best checkpoint
        every_n_train_steps=val_freq,
    )
    callbacks.append(best_ckpt)

    # Also save periodic checkpoints (without affecting the "last" symlink)
    periodic_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step}",
        save_last=False,  # Don't create/update symlink
        every_n_train_steps=ckpt_every_n_steps,
        save_top_k=-1,  # Keep all periodic checkpoints
    )
    callbacks.append(periodic_ckpt)

    return callbacks


def get_lightning_module(model_type: str, data_config: dict, model_config: dict, training_config: dict, var_dims: dict):
    """Create model instance based on config."""
    # combine the model config and training config
    module_config = {**model_config, **training_config}
    module_config["embed_key"] = data_config["embed_key"]
    module_config["output_space"] = data_config["output_space"]
    module_config["gene_names"] = var_dims["gene_names"]
    module_config["batch_size"] = training_config["batch_size"]
    module_config["control_pert"] = data_config.get("control_pert", "non-targeting")

    if data_config["output_space"] == "gene":
        gene_dim = var_dims["hvg_dim"]
    else:
        gene_dim = var_dims["gene_dim"]

    if model_type.lower() == "embedsum":
        from ...tx.models.embed_sum import EmbedSumPerturbationModel

        return EmbedSumPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "old_neuralot":
        from ...tx.models.old_neural_ot import OldNeuralOTPerturbationModel

        return OldNeuralOTPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "neuralot" or model_type.lower() == "pertsets" or model_type.lower() == "state":
        from ...tx.models.state_transition import StateTransitionPerturbationModel

        return StateTransitionPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            basal_mapping_strategy=data_config["basal_mapping_strategy"],
            **module_config,
        )
    elif model_type.lower() == "globalsimplesum" or model_type.lower() == "perturb_mean":
        from ...tx.models.perturb_mean import PerturbMeanPerturbationModel

        return PerturbMeanPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "celltypemean" or model_type.lower() == "context_mean":
        from ...tx.models.context_mean import ContextMeanPerturbationModel

        return ContextMeanPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "decoder_only":
        from ...tx.models.decoder_only import DecoderOnlyPerturbationModel

        return DecoderOnlyPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "pseudobulk":
        from ...tx.models.pseudobulk import PseudobulkPerturbationModel

        return PseudobulkPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "cpa":
        from ...tx.models.cpa import CPAPerturbationModel

        return CPAPerturbationModel(
            input_dim=var_dims["input_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            gene_dim=gene_dim,
            **module_config,
        )
    elif model_type.lower() == "scvi":
        from ...tx.models.scvi import SCVIPerturbationModel

        return SCVIPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "scgpt-chemical" or model_type.lower() == "scgpt-genetic":
        from ...tx.models.scgpt import scGPTForPerturbation

        pretrained_path = module_config["pretrained_path"]
        assert pretrained_path is not None, "pretrained_path must be provided for scGPT"

        model_dir = Path(pretrained_path)
        model_file = model_dir / "best_model.pt"

        model = scGPTForPerturbation(
            ntoken=module_config["ntoken"],
            n_drug_tokens=module_config["n_perts"],  # only used for chemical perturbations
            d_model=module_config["d_model"],
            nhead=module_config["nhead"],
            d_hid=module_config["d_hid"],
            nlayers=module_config["nlayers"],
            nlayers_cls=module_config["n_layers_cls"],
            n_cls=1,
            dropout=module_config["dropout"],
            pad_token_id=module_config["pad_token_id"],
            pad_value=module_config["pad_value"],
            pert_pad_id=module_config["pert_pad_id"],
            do_mvc=module_config["do_MVC"],
            cell_emb_style=module_config["cell_emb_style"],
            mvc_decoder_style=module_config["mvc_decoder_style"],
            use_fast_transformer=module_config["use_fast_transformer"],
            lr=module_config["lr"],
            step_size_lr=module_config["step_size_lr"],
            include_zero_gene=module_config["include_zero_gene"],
            embed_key=module_config["embed_key"],
            perturbation_type=module_config["perturbation_type"],
        )

        load_param_prefixes = module_config["load_param_prefixes"]

        if load_param_prefixes is not None:
            model_dict = model.model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if any([k.startswith(prefix) for prefix in module_config["load_param_prefixes"]])
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")

            model_dict.update(pretrained_dict)
            model.model.load_state_dict(model_dict)
        else:
            try:
                model.model.load_state_dict(torch.load(model_file))
                print(f"Loading all model params from {model_file}")
            except:
                # only load params that are in the model and match the size
                model_dict = model.model.state_dict()
                pretrained_dict = torch.load(model_file)
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    print(f"Loading params {k} with shape {v.shape}")

                model_dict.update(pretrained_dict)
                model.model.load_state_dict(model_dict)

        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

```

5./data3/fanpeishan/state/src/state/configs/config.yaml

```yaml
# This is a template used in the application to generating the config file for
# training tasks
defaults:
  - data: perturbation
  - model: state
  - training: default
  - wandb: default
  - _self_
  

# output_dir must be an absolute path (so that launch scripts are fully descriptive)
name: debug
output_dir: /data3/fanpeishan/state/for_state/run_results/run27/configs_dir
use_wandb: false
overwrite: false
return_adatas: false
pred_adata_path: null
true_adata_path: null

# don't save hydra output
hydra:
  output_subdir: null
  run:
    dir: .
  job_logging:
    formatters:
      simple:
        format: "[%(levelname)s] %(message)s"  # Simple format for logging
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        level: INFO
        stream: ext://sys.stdout
    root:
      level: INFO
    loggers:
      __main__:
        level: DEBUG
        handlers: [console]
        propagate: false
```

6./data3/fanpeishan/state/src/state/configs/data/perturbation.yaml

```yaml
name: PerturbationDataModule
kwargs:
  toml_config_path: null
  embed_key: X_hvg # embed_key: null
  output_space: gene # output_space: all
  pert_rep: onehot
  basal_rep: sample
  num_workers: 8
  pin_memory: true
  n_basal_samples: 1
  basal_mapping_strategy: random
  should_yield_control_cells: true
  batch_col: gem_group # batch_col: plate
  pert_col: drugname_drugconc # pert_col: gene
  cell_type_key: cell_line #cell_type_key: cell_type
  control_pert: ('DMSO_TF', 0.0, 'uM') # control_pert: DMSO_TF
  map_controls: true # for a control cell, should we use it as the target (learn identity) or sample a control?
  perturbation_features_file: null
  store_raw_basal: false
  int_counts: false
  barcode: true
output_dir: null
debug: true

```

7./data3/fanpeishan/state/src/state/configs/model/state.yaml

```yaml
name: state
checkpoint: null
device: cuda

kwargs:
  cell_set_len: 512
  blur: 0.05
  hidden_dim: 696      # hidden dimension going into the transformer backbone
  loss: energy
  confidence_head: False
  n_encoder_layers: 1
  n_decoder_layers: 1
  predict_residual: True
  softplus: True
  freeze_pert_backbone: False
  transformer_decoder: False
  finetune_vci_decoder: False
  residual_decoder: False
  batch_encoder: False
  use_batch_token: False
  nb_decoder: False
  mask_attn: False
  use_effect_gating_token: False
  distributional_loss: energy
  init_from: null
  transformer_backbone_key: llama
  transformer_backbone_kwargs:
      bidirectional_attention: false
      max_position_embeddings: ${model.kwargs.cell_set_len}
      hidden_size: ${model.kwargs.hidden_dim}
      intermediate_size: 2784
      num_hidden_layers: 8
      num_attention_heads: 12
      num_key_value_heads: 12
      head_dim: 58
      use_cache: false
      attention_dropout: 0.0
      hidden_dropout: 0.0
      layer_norm_eps: 1e-6
      pad_token_id: 0
      bos_token_id: 1
      eos_token_id: 2
      tie_word_embeddings: false
      rotary_dim: 0
      use_rotary_embeddings: false
  lora:
      enable: false
      r: 16
      alpha: 32
      dropout: 0.05
      bias: none
      target: auto
      adapt_mlp: false
      task_type: FEATURE_EXTRACTION
      merge_on_eval: false

```

8./data3/fanpeishan/state/src/state/configs/training/default.yaml

```yaml
wandb_track: false
weight_decay: 0.0005
batch_size: 16
lr: 1e-4
max_steps: 40000
train_seed: 42
val_freq: 2000
ckpt_every_n_steps: 2000
gradient_clip_val: 10 # 0 means no clipping
loss_fn: mse
devices: 1  # Number of GPUs to use for training
strategy: auto  # DDP strategy for multi-GPU training
use_mfu: true
mfu_kwargs:
    available_flops: 60e12
    use_backward: true
    logging_interval: 10
    window_size: 2
cumulative_flops_use_backward: true
```

9./data3/fanpeishan/state/for_state/run__commands/tomls/run28.toml

```toml
# Dataset paths - maps dataset names to their directories
[datasets]
c37 = "/data1/fanpeishan/State-Tahoe-Filtered/c37.h5ad" 
c38 = "/data1/fanpeishan/State-Tahoe-Filtered/c38.h5ad" 

# Training specifications
# All cell types in a dataset automatically go into training (excluding zeroshot/fewshot overrides)
[training]
c37 = "train"
c38 = "train"

# Zeroshot specifications - entire cell types go to val or test
[zeroshot]

# Fewshot specifications - explicit perturbation lists
[fewshot]

[fewshot."c37.CVCL_1547"]
val = ["[('Almonertinib (mesylate)', 0.5, 'uM')]","[('Clonidine (hydrochloride)', 5.0, 'uM')]"]
test = ["[('Naproxen', 0.5, 'uM')]","[('Berberine (chloride hydrate)', 5.0, 'uM')]"]


#drugname_drugconc
#[('DMSO_TF', 0.0, 'uM')]                         45150
#[('Adagrasib', 0.05, 'uM')]                      23449
#[('Afatinib', 0.5, 'uM')]                         6042
#[('Almonertinib (mesylate)', 0.5, 'uM')]          5277
#[('Clonidine (hydrochloride)', 5.0, 'uM')]        4556
#[('Naproxen', 0.5, 'uM')]                         4551
#[('Berberine (chloride hydrate)', 5.0, 'uM')]     4550
#[('Berbamine (dihydrochloride)', 0.5, 'uM')]      4480
#[('Belumosudil', 0.5, 'uM')]                      4465
#[('Gemfibrozil', 5.0, 'uM')]                      4410
```

10./data3/fanpeishan/state/src/state/tx/models/state_transition.py

```python
import logging
from typing import Dict, Optional

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geomloss import SamplesLoss
from typing import Tuple

from .base import PerturbationModel
from .decoders import FinetuneVCICountsDecoder
from .decoders_nb import NBDecoder, nb_nll
from .utils import build_mlp, get_activation_class, get_transformer_backbone, apply_lora


logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """
    Combined Sinkhorn + Energy loss
    """

    def __init__(self, sinkhorn_weight=0.001, energy_weight=1.0, blur=0.05):
        super().__init__()
        self.sinkhorn_weight = sinkhorn_weight
        self.energy_weight = energy_weight
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)
        self.energy_loss = SamplesLoss(loss="energy", blur=blur)

    def forward(self, pred, target):
        sinkhorn_val = self.sinkhorn_loss(pred, target)
        energy_val = self.energy_loss(pred, target)
        return self.sinkhorn_weight * sinkhorn_val + self.energy_weight * energy_val


class ConfidenceToken(nn.Module):
    """
    Learnable confidence token that gets appended to the input sequence
    and learns to predict the expected loss value.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Learnable confidence token embedding
        self.confidence_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Projection head to map confidence token output to scalar loss prediction
        self.confidence_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU(),  # Ensure positive loss prediction
        )

    def append_confidence_token(self, seq_input: torch.Tensor) -> torch.Tensor:
        """
        Append confidence token to the sequence input.

        Args:
            seq_input: Input tensor of shape [B, S, E]

        Returns:
            Extended tensor of shape [B, S+1, E]
        """
        batch_size = seq_input.size(0)
        # Expand confidence token to batch size
        confidence_tokens = self.confidence_token.expand(batch_size, -1, -1)
        # Concatenate along sequence dimension
        return torch.cat([seq_input, confidence_tokens], dim=1)

    def extract_confidence_prediction(self, transformer_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract main output and confidence prediction from transformer output.

        Args:
            transformer_output: Output tensor of shape [B, S+1, E]

        Returns:
            main_output: Tensor of shape [B, S, E]
            confidence_pred: Tensor of shape [B, 1]
        """
        # Split the output
        main_output = transformer_output[:, :-1, :]  # [B, S, E]
        confidence_output = transformer_output[:, -1:, :]  # [B, 1, E]

        # Project confidence token output to scalar
        confidence_pred = self.confidence_projection(confidence_output).squeeze(-1)  # [B, 1]

        return main_output, confidence_pred


class StateTransitionPerturbationModel(PerturbationModel):
    """
    This model:
      1) Projects basal expression and perturbation encodings into a shared latent space.
      2) Uses an OT-based distributional loss (energy, sinkhorn, etc.) from geomloss.
      3) Enables cells to attend to one another, learning a set-to-set function rather than
      a sample-to-sample single-cell map.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        basal_mapping_strategy: str = "random",
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: dict = None,
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            input_dim: dimension of the input expression (e.g. number of genes or embedding dimension).
            hidden_dim: not necessarily used, but required by PerturbationModel signature.
            output_dim: dimension of the output space (genes or latent).
            pert_dim: dimension of perturbation embedding.
            gpt: e.g. "TranslationTransformerSamplesModel".
            model_kwargs: dictionary passed to that model's constructor.
            loss: choice of distributional metric ("sinkhorn", "energy", etc.).
            **kwargs: anything else to pass up to PerturbationModel or not used.
        """
        # Call the parent PerturbationModel constructor
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gene_dim=gene_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            output_space=output_space,
            **kwargs,
        )

        # Save or store relevant hyperparams
        self.predict_residual = predict_residual
        self.output_space = output_space
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.cell_sentence_len = kwargs.get("cell_set_len", 256)
        self.decoder_loss_weight = kwargs.get("decoder_weight", 1.0)
        self.regularization = kwargs.get("regularization", 0.0)
        self.detach_decoder = kwargs.get("detach_decoder", False)

        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.transformer_backbone_kwargs["n_positions"] = self.cell_sentence_len + kwargs.get("extra_tokens", 0)

        self.distributional_loss = distributional_loss
        self.gene_dim = gene_dim

        # Build the distributional loss from geomloss
        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")
        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "se":
            sinkhorn_weight = kwargs.get("sinkhorn_weight", 0.01)  # 1/100 = 0.01
            energy_weight = kwargs.get("energy_weight", 1.0)
            self.loss_fn = CombinedLoss(sinkhorn_weight=sinkhorn_weight, energy_weight=energy_weight, blur=blur)
        elif loss_name == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", blur=blur)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        self.use_basal_projection = kwargs.get("use_basal_projection", True)

        # Build the underlying neural OT network
        self._build_networks(lora_cfg=kwargs.get("lora", None))

        # Add an optional encoder that introduces a batch variable
        self.batch_encoder = None
        self.batch_dim = None
        self.predict_mean = kwargs.get("predict_mean", False)
        if kwargs.get("batch_encoder", False) and batch_dim is not None:
            self.batch_encoder = nn.Embedding(
                num_embeddings=batch_dim,
                embedding_dim=hidden_dim,
            )
            self.batch_dim = batch_dim

        # if the model is outputting to counts space, apply relu
        # otherwise its in embedding space and we don't want to
        is_gene_space = kwargs["embed_key"] == "X_hvg" or kwargs["embed_key"] is None
        if is_gene_space or self.gene_decoder is None:
            self.relu = torch.nn.ReLU()

        self.use_batch_token = kwargs.get("use_batch_token", False)
        self.basal_mapping_strategy = basal_mapping_strategy
        # Disable batch token only for truly incompatible cases
        disable_reasons = []
        if self.batch_encoder and self.use_batch_token:
            disable_reasons.append("batch encoder is used")
        if basal_mapping_strategy == "random" and self.use_batch_token:
            disable_reasons.append("basal mapping strategy is random")

        if disable_reasons:
            self.use_batch_token = False
            logger.warning(
                f"Batch token is not supported when {' or '.join(disable_reasons)}, setting use_batch_token to False"
            )
            try:
                self.hparams["use_batch_token"] = False
            except Exception:
                pass

        self.batch_token_weight = kwargs.get("batch_token_weight", 0.1)
        self.batch_token_num_classes: Optional[int] = batch_dim if self.use_batch_token else None

        if self.use_batch_token:
            if self.batch_token_num_classes is None:
                raise ValueError("batch_token_num_classes must be set when use_batch_token is True")
            self.batch_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
            self.batch_classifier = build_mlp(
                in_dim=self.hidden_dim,
                out_dim=self.batch_token_num_classes,
                hidden_dim=self.hidden_dim,
                n_layers=1,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.batch_token = None
            self.batch_classifier = None

        # Internal cache for last token features (B, S, H) from transformer for aux loss
        self._batch_token_cache: Optional[torch.Tensor] = None

        # initialize a confidence token
        self.confidence_token = None
        self.confidence_loss_fn = None
        if kwargs.get("confidence_token", False):
            self.confidence_token = ConfidenceToken(hidden_dim=self.hidden_dim, dropout=self.dropout)
            self.confidence_loss_fn = nn.MSELoss()

        # Backward-compat: accept legacy key `freeze_pert`
        self.freeze_pert_backbone = kwargs.get("freeze_pert_backbone", kwargs.get("freeze_pert", False))
        if self.freeze_pert_backbone:
            # Freeze backbone base weights but keep LoRA adapter weights (if present) trainable
            for name, param in self.transformer_backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # Freeze projection head as before
            for param in self.project_out.parameters():
                param.requires_grad = False

        if kwargs.get("nb_decoder", False):
            self.gene_decoder = NBDecoder(
                latent_dim=self.output_dim + (self.batch_dim or 0),
                gene_dim=gene_dim,
                hidden_dims=[512, 512, 512],
                dropout=self.dropout,
            )

        control_pert = kwargs.get("control_pert", "non-targeting")
        if kwargs.get("finetune_vci_decoder", False):  # TODO: This will go very soon
            gene_names = []

            if output_space == "gene":
                # hvg's but for which dataset?
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/jurkat.h5")
                    # gene_names = temp.var.index.values
            else:
                assert output_space == "all"
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    # temp = ad.read_h5ad('/scratch/ctc/ML/vci/paper_replogle/jurkat.h5')
                    # gene_names = temp.var.index.values
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/cross_dataset/replogle/jurkat.h5")
                    gene_names = temp.var.index.values

            self.gene_decoder = FinetuneVCICountsDecoder(
                genes=gene_names,
                # latent_dim=self.output_dim + (self.batch_dim or 0),
            )
        print(self)

    def _build_networks(self, lora_cfg=None):
        """
        Here we instantiate the actual GPT2-based model.
        """
        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        # Simple linear layer that maintains the input dimension
        if self.use_basal_projection:
            self.basal_encoder = build_mlp(
                in_dim=self.input_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.basal_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        # Optionally wrap backbone with LoRA adapters
        if lora_cfg and lora_cfg.get("enable", False):
            self.transformer_backbone = apply_lora(
                self.transformer_backbone,
                self.transformer_backbone_key,
                lora_cfg,
            )

        # Project from input_dim to hidden_dim for transformer input
        # self.project_to_hidden = nn.Linear(self.input_dim, self.hidden_dim)

        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        if self.output_space == "all":
            self.final_down_then_up = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 8),
                nn.GELU(),
                nn.Linear(self.output_dim // 8, self.output_dim),
            )

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """If needed, define how we embed the raw perturbation input."""
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        return self.basal_encoder(expr)

    def forward(self, batch: dict, padded=True) -> torch.Tensor:
        """
        The main forward call. Batch is a flattened sequence of cell sentences,
        which we reshape into sequences of length cell_sentence_len.

        Expects input tensors of shape (B, S, N) where:
        B = batch size
        S = sequence length (cell_sentence_len)
        N = feature dimension

        The `padded` argument here is set to True if the batch is padded. Otherwise, we
        expect a single batch, so that sentences can vary in length across batches.
        """
        if padded:
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
        else:
            # we are inferencing on a single batch, so accept variable length sentences
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)

        # Shape: [B, S, input_dim]
        pert_embedding = self.encode_perturbation(pert)
        control_cells = self.encode_basal_expression(basal)

        # Add encodings in input_dim space, then project to hidden_dim
        combined_input = pert_embedding + control_cells  # Shape: [B, S, hidden_dim]
        seq_input = combined_input  # Shape: [B, S, hidden_dim]

        if self.batch_encoder is not None:
            # Extract batch indices (assume they are integers or convert from one-hot)
            batch_indices = batch["batch"]

            # Handle one-hot encoded batch indices
            if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
                batch_indices = batch_indices.argmax(-1)

            # Reshape batch indices to match sequence structure
            if padded:
                batch_indices = batch_indices.reshape(-1, self.cell_sentence_len)
            else:
                batch_indices = batch_indices.reshape(1, -1)

            # Get batch embeddings and add to sequence input
            batch_embeddings = self.batch_encoder(batch_indices.long())  # Shape: [B, S, hidden_dim]
            seq_input = seq_input + batch_embeddings

        if self.use_batch_token and self.batch_token is not None:
            batch_size, _, _ = seq_input.shape
            # Prepend the batch token to the sequence along the sequence dimension
            # [B, S, H] -> [B, S+1, H], batch token at position 0
            seq_input = torch.cat([self.batch_token.expand(batch_size, -1, -1), seq_input], dim=1)

        confidence_pred = None
        if self.confidence_token is not None:
            # Append confidence token: [B, S, E] -> [B, S+1, E] (might be one more if we have the batch token)
            seq_input = self.confidence_token.append_confidence_token(seq_input)

        # forward pass + extract CLS last hidden state
        if self.hparams.get("mask_attn", False):
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device
            self.transformer_backbone._attn_implementation = "eager"   # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

            # create a [1,1,S,S] mask (now S+1 if confidence token is used)
            base = torch.eye(seq_length, device=device, dtype=torch.bool).view(1, 1, seq_length, seq_length)
            
            # Get number of attention heads from model config
            num_heads = self.transformer_backbone.config.num_attention_heads

            # repeat out to [B,H,S,S]
            attn_mask = base.repeat(batch_size, num_heads, 1, 1)

            outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
            transformer_output = outputs.last_hidden_state
        else:
            outputs = self.transformer_backbone(inputs_embeds=seq_input)
            transformer_output = outputs.last_hidden_state

        # Extract outputs accounting for optional prepended batch token and optional confidence token at the end
        if self.confidence_token is not None and self.use_batch_token and self.batch_token is not None:
            # transformer_output: [B, 1 + S + 1, H] -> batch token at 0, cells 1..S, confidence at -1
            batch_token_pred = transformer_output[:, :1, :]  # [B, 1, H]
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(
                transformer_output[:, 1:, :]
            )
            # res_pred currently excludes the confidence token and starts from former index 1
            self._batch_token_cache = batch_token_pred
        elif self.confidence_token is not None:
            # Only confidence token appended at the end
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(transformer_output)
            self._batch_token_cache = None
        elif self.use_batch_token and self.batch_token is not None:
            # Only batch token prepended at the beginning
            batch_token_pred = transformer_output[:, :1, :]  # [B, 1, H]
            res_pred = transformer_output[:, 1:, :]  # [B, S, H]
            self._batch_token_cache = batch_token_pred
        else:
            # Neither special token used
            res_pred = transformer_output
            self._batch_token_cache = None

        # add to basal if predicting residual
        if self.predict_residual and self.output_space == "all":
            # Project control_cells to hidden_dim space to match res_pred
            # control_cells_hidden = self.project_to_hidden(control_cells)
            # treat the actual prediction as a residual sum to basal
            out_pred = self.project_out(res_pred) + basal
            out_pred = self.final_down_then_up(out_pred)
        elif self.predict_residual:
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)

        # apply relu if specified and we output to HVG space
        is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None
        # logger.info(f"DEBUG: is_gene_space: {is_gene_space}")
        # logger.info(f"DEBUG: self.gene_decoder: {self.gene_decoder}")
        if is_gene_space or self.gene_decoder is None:
            out_pred = self.relu(out_pred)

        output = out_pred.reshape(-1, self.output_dim)

        if confidence_pred is not None:
            return output, confidence_pred
        else:
            return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        confidence_pred = None
        if self.confidence_token is not None:
            pred, confidence_pred = self.forward(batch, padded=padded)
        else:
            pred = self.forward(batch, padded=padded)

        target = batch["pert_cell_emb"]

        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)

        main_loss = self.loss_fn(pred, target).nanmean()
        self.log("train_loss", main_loss)

        # Log individual loss components if using combined loss
        if hasattr(self.loss_fn, "sinkhorn_loss") and hasattr(self.loss_fn, "energy_loss"):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).nanmean()
            energy_component = self.loss_fn.energy_loss(pred, target).nanmean()
            self.log("train/sinkhorn_loss", sinkhorn_component)
            self.log("train/energy_loss", energy_component)

        # Process decoder if available
        decoder_loss = None
        total_loss = main_loss

        if self.use_batch_token and self.batch_classifier is not None and self._batch_token_cache is not None:
            logits = self.batch_classifier(self._batch_token_cache)  # [B, 1, C]
            batch_token_targets = batch["batch"]

            B = logits.shape[0]
            C = logits.size(-1)

            # Prepare one label per sequence (all S cells share the same batch)
            if batch_token_targets.dim() > 1 and batch_token_targets.size(-1) == C:
                # One-hot labels; reshape to [B, S, C]
                if padded:
                    target_oh = batch_token_targets.reshape(-1, self.cell_sentence_len, C)
                else:
                    target_oh = batch_token_targets.reshape(1, -1, C)
                sentence_batch_labels = target_oh.argmax(-1)
            else:
                # Integer labels; reshape to [B, S]
                if padded:
                    sentence_batch_labels = batch_token_targets.reshape(-1, self.cell_sentence_len)
                else:
                    sentence_batch_labels = batch_token_targets.reshape(1, -1)

            if sentence_batch_labels.shape[0] != B:
                sentence_batch_labels = sentence_batch_labels.reshape(B, -1)

            if self.basal_mapping_strategy == "batch":
                uniform_mask = sentence_batch_labels.eq(sentence_batch_labels[:, :1]).all(dim=1)
                if not torch.all(uniform_mask):
                    bad_indices = torch.where(~uniform_mask)[0]
                    label_strings = []
                    for idx in bad_indices:
                        labels = sentence_batch_labels[idx].detach().cpu().tolist()
                        logger.error("Batch labels for sentence %d: %s", idx.item(), labels)
                        label_strings.append(f"sentence {idx.item()}: {labels}")
                    raise ValueError(
                        "Expected all cells in a sentence to share the same batch when "
                        "basal_mapping_strategy is 'batch'. "
                        f"Found mixed batch labels: {', '.join(label_strings)}"
                    )

            target_idx = sentence_batch_labels[:, 0]

            # Safety: ensure exactly one target per sequence
            if target_idx.numel() != B:
                target_idx = target_idx.reshape(-1)[:B]

            ce_loss = F.cross_entropy(logits.reshape(B, -1, C).squeeze(1), target_idx.long())
            self.log("train/batch_token_loss", ce_loss)
            total_loss = total_loss + self.batch_token_weight * ce_loss

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]
            # Train decoder to map latent predictions to gene space

            if self.detach_decoder:
                # with some random change, use the true targets
                if np.random.rand() < 0.1:
                    latent_preds = target.reshape_as(pred).detach()
                else:
                    latent_preds = pred.detach()
            else:
                latent_preds = pred

            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds)
                if padded:
                    gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                else:
                    gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = total_loss.detach().clone().unsqueeze(0) * 10

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("train/confidence_loss", confidence_loss)
            self.log("train/actual_loss", loss_target.mean())

            # Add to total loss with weighting
            confidence_weight = 0.1  # You can make this configurable
            total_loss = total_loss + confidence_weight * confidence_loss

            # Add to total loss
            total_loss = total_loss + confidence_loss

        if self.regularization > 0.0:
            ctrl_cell_emb = batch["ctrl_cell_emb"].reshape_as(pred)
            delta = pred - ctrl_cell_emb

            # compute l1 loss
            l1_loss = torch.abs(delta).mean()

            # Log the regularization loss
            self.log("train/l1_regularization", l1_loss)

            # Add regularization to total loss
            total_loss = total_loss + self.regularization * l1_loss

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch), None
        else:
            pred, confidence_pred = self.forward(batch)

        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["pert_cell_emb"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)

        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss)

        # Log individual loss components if using combined loss
        if hasattr(self.loss_fn, "sinkhorn_loss") and hasattr(self.loss_fn, "energy_loss"):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).mean()
            energy_component = self.loss_fn.energy_loss(pred, target).mean()
            self.log("val/sinkhorn_loss", sinkhorn_component)
            self.log("val/energy_loss", energy_component)

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]

            # Get model predictions from validation step
            latent_preds = pred

            # Train decoder to map latent predictions to gene space
            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                # Get decoder predictions
                pert_cell_counts_preds = self.gene_decoder(latent_preds).reshape(
                    -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
                )
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log the validation metric
            self.log("val/decoder_loss", decoder_loss)
            loss = loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("val/confidence_loss", confidence_loss)
            self.log("val/actual_loss", loss_target.mean())

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch, padded=False), None
        else:
            pred, confidence_pred = self.forward(batch, padded=False)

        target = batch["pert_cell_emb"]
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("test_loss", loss)

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10.0

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("test/confidence_loss", confidence_loss)

    def predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:s
         returning 'preds', 'X', 'pert_name', etc.
        """
        if self.confidence_token is None:
            latent_output = self.forward(batch, padded=padded)  # shape [B, ...]
            confidence_pred = None
        else:
            latent_output, confidence_pred = self.forward(batch, padded=padded)

        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
            "pert_cell_barcode": batch.get("pert_cell_barcode", None),
            "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
        }

        # Add confidence prediction to output if available
        if confidence_pred is not None:
            output_dict["confidence_pred"] = confidence_pred

        if self.gene_decoder is not None:
            if isinstance(self.gene_decoder, NBDecoder):
                mu, _ = self.gene_decoder(latent_output)
                pert_cell_counts_preds = mu
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_output)

            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict

```

11./data3/fanpeishan/state/.venv/lib/python3.11/site-packages/cell_load/data_modules/samplers.py

```python
import logging
import time
from typing import Iterator
import copy

import numpy as np
from torch.utils.data import Sampler, Subset
import torch.distributed as dist

from ..dataset import MetadataConcatDataset, PerturbationDataset
from ..utils.data_utils import H5MetadataCache

logger = logging.getLogger(__name__)


class PerturbationBatchSampler(Sampler):
    """
    Samples batches ensuring that cells in each batch share the same
    (cell_type, perturbation) combination, using only H5 codes.

    Instead of grouping by cell type and perturbation names, this sampler
    groups based on integer codes stored in the H5 file (e.g. `cell_type_codes`
    and `pert_codes` in the H5MetadataCache). This avoids repeated string operations.

    Supports distributed training.
    """

    def __init__(
        self,
        dataset: "MetadataConcatDataset",
        batch_size: int,
        drop_last: bool = False,
        cell_sentence_len: int = 512,
        test: bool = False,
        use_batch: bool = False,
        seed: int = 0,
        epoch: int = 0,
    ):
        logger.info(
            "Creating perturbation batch sampler with metadata caching (using codes)..."
        )
        start_time = time.time()

        # If the provided dataset has a `.data_source` attribute, use that.
        self.dataset = (
            dataset.data_source if hasattr(dataset, "data_source") else dataset
        )
        self.batch_size = batch_size
        self.test = test
        self.use_batch = use_batch
        self.seed = seed
        self.epoch = epoch

        if self.test and self.batch_size != 1:
            logger.warning(
                "Batch size should be 1 for test mode. Setting batch size to 1."
            )
            self.batch_size = 1

        self.cell_sentence_len = cell_sentence_len
        self.drop_last = drop_last

        # Setup distributed settings if distributed mode is enabled.
        self.distributed = False
        self.num_replicas = 1
        self.rank = 0

        if dist.is_available() and dist.is_initialized():
            self.distributed = True
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(
                f"Distributed mode enabled. World size: {self.num_replicas}, rank: {self.rank}."
            )

        # Create caches for all unique H5 files.
        self.metadata_caches = {}
        for subset in self.dataset.datasets:
            base_dataset: PerturbationDataset = subset.dataset
            self.metadata_caches[base_dataset.h5_path] = base_dataset.metadata_cache

        # Create batches using the code-based grouping.
        self.sentences = self._create_sentences()
        sentence_lens = [len(sentence) for sentence in self.sentences]
        avg_num = np.mean(sentence_lens)
        std_num = np.std(sentence_lens)
        tot_num = np.sum(sentence_lens)
        logger.info(
            f"Total # cells {tot_num}. Cell set size mean / std before resampling: {avg_num:.2f} / {std_num:.2f}."
        )

        # combine sentences into batches that are flattened
        logger.info(
            f"Creating meta-batches with cell_sentence_len={cell_sentence_len}..."
        )
        start_time = time.time()
        self.batches = self._create_batches()
        self.tot_num = tot_num
        end_time = time.time()
        logger.info(
            f"Sampler created with {len(self.batches)} batches in {end_time - start_time:.2f} seconds."
        )

    def _create_batches(self) -> list[list[int]]:
        """
        Combines existing batches into meta-batches of size batch_size * cell_sentence_len,
        sampling with replacement if needed to reach cell_sentence_len.

        IF distributed, each rank will process a subset of the sentences.
        """

        if self.distributed:
            rank_sentences = self._get_rank_sentences()

        else:
            rank_sentences = self.sentences

        all_batches = []
        current_batch = []

        num_full = 0
        num_partial = 0
        for sentence in rank_sentences:
            # If batch is smaller than cell_sentence_len, sample with replacement
            if len(sentence) < self.cell_sentence_len and not self.test:
                # during inference, don't sample by replacement
                new_sentence = np.random.choice(
                    sentence, size=self.cell_sentence_len, replace=True
                ).tolist()
                num_partial += 1
            else:
                new_sentence = copy.deepcopy(sentence)
                assert len(new_sentence) == self.cell_sentence_len or self.test
                num_full += 1

            sentence_len = len(new_sentence) if self.test else self.cell_sentence_len

            if len(current_batch) + len(new_sentence) <= self.batch_size * sentence_len:
                current_batch.extend(new_sentence)
            else:
                if current_batch:  # Add the completed meta-batch
                    all_batches.append(current_batch)
                current_batch = new_sentence

        if self.distributed:
            logger.info(
                f"Rank {self.rank}: Of {len(rank_sentences)} sentences, {num_full} were full and {num_partial} were partial."
            )
        else:
            logger.info(
                f"Of all batches, {num_full} were full and {num_partial} were partial."
            )

        # Add the last meta-batch if it exists
        if current_batch and not self.drop_last:
            all_batches.append(current_batch)

        return all_batches

    def _get_rank_sentences(self) -> list[list[int]]:
        """
        Get the subset of sentences that this rank should process.
        Sentences are shuffled using epoch-based seed, then distributed across ranks.
        """
        # Shuffle sentences using epoch-based seed for consistent ordering across ranks
        shuffled_sentences = self.sentences.copy()
        np.random.RandomState(self.seed + self.epoch).shuffle(shuffled_sentences)

        # Calculate sentence distribution across processes
        total_sentences = len(shuffled_sentences)
        base_sentences = total_sentences // self.num_replicas
        remainder = total_sentences % self.num_replicas

        # Calculate number of sentences for this specific rank
        if self.rank < remainder:
            num_sentences_for_rank = base_sentences + 1
        else:
            num_sentences_for_rank = base_sentences

        # Calculate starting sentence index for this rank
        start_sentence_idx = self.rank * base_sentences + min(self.rank, remainder)
        end_sentence_idx = start_sentence_idx + num_sentences_for_rank

        rank_sentences = shuffled_sentences[start_sentence_idx:end_sentence_idx]

        logger.info(
            f"Rank {self.rank}: Processing {len(rank_sentences)} sentences "
            f"(indices {start_sentence_idx} to {end_sentence_idx - 1} of {total_sentences})"
        )

        return rank_sentences

    def _process_subset(self, global_offset: int, subset: Subset) -> list[list[int]]:
        """
        Process a single subset to create batches based on H5 codes.

        Optimized version with integer group encoding:
        - Groups are encoded into a single integer via np.ravel_multi_index.
        - Sorting/grouping is done on simple integers instead of structured dtypes.
        - Much faster for large numbers of groups.
        """
        base_dataset = subset.dataset
        indices = np.array(subset.indices)
        cache: H5MetadataCache = self.metadata_caches[base_dataset.h5_path]

        # Codes
        cell_codes = cache.cell_type_codes[indices]
        pert_codes = cache.pert_codes[indices]

        if getattr(self, "use_batch", False):
            batch_codes = cache.batch_codes[indices]
            # Encode (batch, cell, pert) into one integer
            group_keys = np.ravel_multi_index(
                (batch_codes, cell_codes, pert_codes),
                (batch_codes.max() + 1, cell_codes.max() + 1, pert_codes.max() + 1),
            )
        else:
            # Encode (cell, pert) into one integer
            group_keys = np.ravel_multi_index(
                (cell_codes, pert_codes), (cell_codes.max() + 1, pert_codes.max() + 1)
            )

        # Global indices
        global_indices = np.arange(global_offset, global_offset + len(indices))

        # Sort once by group key
        order = np.argsort(group_keys)
        sorted_keys = group_keys[order]
        sorted_indices = global_indices[order]

        # Find group boundaries
        unique_keys, group_starts = np.unique(sorted_keys, return_index=True)
        group_starts = np.r_[group_starts, len(sorted_keys)]

        subset_batches = []

        # Iterate groups
        for start, end in zip(group_starts[:-1], group_starts[1:]):
            group_indices = sorted_indices[start:end]
            np.random.shuffle(group_indices)

            for i in range(0, len(group_indices), self.cell_sentence_len):
                sentence = group_indices[i : i + self.cell_sentence_len]
                if len(sentence) < self.cell_sentence_len and self.drop_last:
                    continue
                subset_batches.append(sentence.tolist())

        return subset_batches

    def _create_sentences(self) -> list[list[int]]:
        """
        Process each subset sequentially (across all datasets) and combine the batches.
        """
        global_offset = 0
        all_batches = []
        for subset in self.dataset.datasets:
            subset_batches = self._process_subset(global_offset, subset)
            all_batches.extend(subset_batches)
            global_offset += len(subset)
        np.random.shuffle(all_batches)

        return all_batches

    def __iter__(self) -> Iterator[list[int]]:
        # Shuffle the order of batches each time we iterate in non-distributed mode.
        if not self.distributed:
            self.batches = self._create_batches()
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        This ensures all replicas use a different random ordering for each epoch.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch
        # Recreate batches for new epoch (sentences remain the same)
        self.batches = self._create_batches()

```

12./data3/fanpeishan/state/src/state/configs/model/state.yaml

```yaml
name: state
checkpoint: null
device: cuda

kwargs:
  cell_set_len: 512
  blur: 0.05
  hidden_dim: 696      # hidden dimension going into the transformer backbone
  loss: energy
  confidence_head: False
  n_encoder_layers: 1
  n_decoder_layers: 1
  predict_residual: True
  softplus: True
  freeze_pert_backbone: False
  transformer_decoder: False
  finetune_vci_decoder: False
  residual_decoder: False
  batch_encoder: False
  use_batch_token: False
  nb_decoder: False
  mask_attn: False
  use_effect_gating_token: False
  distributional_loss: energy
  init_from: null
  transformer_backbone_key: llama
  transformer_backbone_kwargs:
      bidirectional_attention: false
      max_position_embeddings: ${model.kwargs.cell_set_len}
      hidden_size: ${model.kwargs.hidden_dim}
      intermediate_size: 2784
      num_hidden_layers: 8
      num_attention_heads: 12
      num_key_value_heads: 12
      head_dim: 58
      use_cache: false
      attention_dropout: 0.0
      hidden_dropout: 0.0
      layer_norm_eps: 1e-6
      pad_token_id: 0
      bos_token_id: 1
      eos_token_id: 2
      tie_word_embeddings: false
      rotary_dim: 0
      use_rotary_embeddings: false
  lora:
      enable: false
      r: 16
      alpha: 32
      dropout: 0.05
      bias: none
      target: auto
      adapt_mlp: false
      task_type: FEATURE_EXTRACTION
      merge_on_eval: false

```

13./data3/fanpeishan/state/src/state/tx/models/base.py

```python
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
import typing as tp

from .utils import get_loss_fn

logger = logging.getLogger(__name__)


class LatentToGeneDecoder(nn.Module):
    """
    A decoder module to transform latent embeddings back to gene expression space.

    This takes concat([cell embedding]) as the input, and predicts
    counts over all genes as output.

    This decoder is trained separately from the main perturbation model.

    Args:
        latent_dim: Dimension of latent space
        gene_dim: Dimension of gene space (number of HVGs)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        residual_decoder: If True, adds residual connections between every other layer block
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: List[int] = [512, 1024],
        dropout: float = 0.1,
        residual_decoder=False,
    ):
        super().__init__()

        self.residual_decoder = residual_decoder

        if residual_decoder:
            # Build individual blocks for residual connections
            self.blocks = nn.ModuleList()
            input_dim = latent_dim

            for hidden_dim in hidden_dims:
                block = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
                )
                self.blocks.append(block)
                input_dim = hidden_dim

            # Final output layer
            self.final_layer = nn.Sequential(nn.Linear(input_dim, gene_dim), nn.ReLU())
        else:
            # Original implementation without residual connections
            layers = []
            input_dim = latent_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

            # Final output layer
            layers.append(nn.Linear(input_dim, gene_dim))
            # Make sure outputs are non-negative
            layers.append(nn.ReLU())

            self.decoder = nn.Sequential(*layers)

    def gene_dim(self):
        # return the output dimension of the last layer
        if self.residual_decoder:
            return self.final_layer[0].out_features
        else:
            for module in reversed(self.decoder):
                if isinstance(module, nn.Linear):
                    return module.out_features
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x: Latent embeddings of shape [batch_size, latent_dim]

        Returns:
            Gene expression predictions of shape [batch_size, gene_dim]
        """
        if self.residual_decoder:
            # Apply blocks with residual connections between every other block
            block_outputs = []
            current = x

            for i, block in enumerate(self.blocks):
                output = block(current)

                # Add residual connection from every other previous block
                # Pattern: blocks 1, 3, 5, ... get residual from blocks 0, 2, 4, ...
                if i >= 1 and i % 2 == 1:  # Odd-indexed blocks (1, 3, 5, ...)
                    residual_idx = i - 1  # Previous even-indexed block
                    output = output + block_outputs[residual_idx]

                block_outputs.append(output)
                current = output

            return self.final_layer(current)
        else:
            return self.decoder(x)


class PerturbationModel(ABC, LightningModule):
    """
    Base class for perturbation models that can operate on either raw counts or embeddings.

    Args:
        input_dim: Dimension of input features (genes or embeddings)
        hidden_dim: Hidden dimension for neural network layers
        output_dim: Dimension of output (always gene space)
        pert_dim: Dimension of perturbation embeddings
        dropout: Dropout rate
        lr: Learning rate for optimizer
        loss_fn: Loss function ('mse' or custom nn.Module)
        output_space: 'gene' or 'latent'
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        dropout: float = 0.1,
        lr: float = 3e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        control_pert: str = "non-targeting",
        embed_key: Optional[str] = None,
        output_space: str = "gene",
        gene_names: Optional[List[str]] = None,
        batch_size: int = 64,
        gene_dim: int = 5000,
        hvg_dim: int = 2001,
        decoder_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.decoder_cfg = decoder_cfg
        self.save_hyperparameters()
        self.gene_decoder_bool = kwargs.get("gene_decoder_bool", True)

        # Core architecture settings
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pert_dim = pert_dim
        self.batch_dim = batch_dim
        self.gene_dim = gene_dim
        self.hvg_dim = hvg_dim

        if kwargs.get("batch_encoder", False):
            self.batch_dim = batch_dim
        else:
            self.batch_dim = None

        self.residual_decoder = kwargs.get("residual_decoder", False)

        self.embed_key = embed_key
        self.output_space = output_space
        self.batch_size = batch_size
        self.control_pert = control_pert

        # Training settings
        self.gene_names = gene_names  # store the gene names that this model output for gene expression space
        self.dropout = dropout
        self.lr = lr
        self.loss_fn = get_loss_fn(loss_fn)
        self._build_decoder()

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    @abstractmethod
    def _build_networks(self):
        """Build the core neural network components."""
        pass

    def _build_decoder(self):
        """Create self.gene_decoder from self.decoder_cfg (or leave None)."""
        if self.gene_decoder_bool == False:
            self.gene_decoder = None
            return
        if self.decoder_cfg is None:
            self.gene_decoder = None
            return
        self.gene_decoder = LatentToGeneDecoder(**self.decoder_cfg)

    def on_load_checkpoint(self, checkpoint: dict[str, tp.Any]) -> None:
        """
        Lightning calls this *before* the checkpoint's state_dict is loaded.
        Re-create the decoder using the exact hyper-parameters saved in the ckpt,
        so that parameter shapes match and load_state_dict succeeds.
        """
        # Check if decoder_cfg was already set externally (e.g., by training script for output_space mismatch)
        decoder_already_configured = (
            hasattr(self, "_decoder_externally_configured") and self._decoder_externally_configured
        )

        if self.gene_decoder_bool == False:
            self.gene_decoder = None
            return
        if not decoder_already_configured and "decoder_cfg" in checkpoint["hyper_parameters"]:
            self.decoder_cfg = checkpoint["hyper_parameters"]["decoder_cfg"]
            self.gene_decoder = LatentToGeneDecoder(**self.decoder_cfg)
            logger.info(f"Loaded decoder from checkpoint decoder_cfg: {self.decoder_cfg}")
        elif not decoder_already_configured:
            # Only fall back to old logic if no decoder_cfg was saved and not externally configured
            self.decoder_cfg = None
            self._build_decoder()
            logger.info(f"DEBUG: output_space: {self.output_space}")
            if self.gene_decoder is None:
                gene_dim = self.hvg_dim if self.output_space == "gene" else self.gene_dim
                logger.info(f"DEBUG: gene_dim: {gene_dim}")
                if (self.embed_key and self.embed_key != "X_hvg" and self.output_space == "gene") or (
                    self.embed_key and self.output_space == "all"
                ):  # we should be able to decode from hvg to all
                    logger.info("DEBUG: Creating gene_decoder, checking conditions...")
                    if gene_dim > 10000:
                        hidden_dims = [1024, 512, 256]
                    else:
                        if "DMSO_TF" in self.control_pert:
                            if self.residual_decoder:
                                hidden_dims = [2058, 2058, 2058, 2058, 2058]
                            else:
                                hidden_dims = [4096, 2048, 2048]
                        elif "PBS" in self.control_pert:
                            hidden_dims = [2048, 1024, 1024]
                        else:
                            hidden_dims = [1024, 1024, 512]  # make this config

                    self.gene_decoder = LatentToGeneDecoder(
                        latent_dim=self.output_dim,
                        gene_dim=gene_dim,
                        hidden_dims=hidden_dims,
                        dropout=self.dropout,
                        residual_decoder=self.residual_decoder,
                    )
                    logger.info(f"Initialized gene decoder for embedding {self.embed_key} to gene space")
        else:
            logger.info("Decoder was already configured externally, skipping checkpoint decoder configuration")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        pred = self(batch)

        # Compute main model loss
        main_loss = self.loss_fn(pred, batch["pert_cell_emb"])
        self.log("train_loss", main_loss)

        # Process decoder if available
        decoder_loss = None
        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            # Train decoder to map latent predictions to gene space
            with torch.no_grad():
                latent_preds = pred.detach()  # Detach to prevent gradient flow back to main model

            pert_cell_counts_preds = self.gene_decoder(latent_preds)
            gene_targets = batch["pert_cell_counts"]
            decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets)

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = main_loss + decoder_loss
        else:
            total_loss = main_loss

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        pred = self(batch)
        loss = self.loss_fn(pred, batch["pert_cell_emb"])

        # TODO: remove unused
        # is_control = self.control_pert in batch["pert_name"]
        self.log("val_loss", loss)

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        latent_output = self(batch)
        target = batch[self.embed_key]
        loss = self.loss_fn(latent_output, target)

        output_dict = {
            "preds": latent_output,  # The distribution's sample
            "pert_cell_emb": batch.get("pert_cell_emb", None),  # The target gene expression or embedding
            "pert_cell_counts": batch.get("pert_cell_counts", None),  # the true, raw gene expression
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds
            decoder_loss = self.loss_fn(pert_cell_counts_preds, batch["pert_cell_counts"])
            self.log("test_decoder_loss", decoder_loss, prog_bar=True)

        self.log("test_loss", loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        """
        latent_output = self.forward(batch)
        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict

    def decode_to_gene_space(self, latent_embeds: torch.Tensor, basal_expr: None) -> torch.Tensor:
        """
        Decode latent embeddings to gene expression space.

        Args:
            latent_embeds: Embeddings in latent space

        Returns:
            Gene expression predictions or None if decoder is not available
        """
        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_embeds)
            if basal_expr is not None:
                # Add basal expression if provided
                pert_cell_counts_preds += basal_expr
            return pert_cell_counts_preds
        return None

    def configure_optimizers(self):
        """
        Configure a single optimizer for both the main model and the gene decoder.
        """
        # Use a single optimizer for all parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
```

我将一些官方提供的.toml文件样例发给你。

14./data3/fanpeishan/state/examples/fewshot.toml

```toml
# Dataset paths - maps dataset names to their directories
[datasets]
example = "/home/aadduri/state/examples" # CHANGE THIS TO YOUR DIRECTORY

# Training specifications
# All cell types in a dataset automatically go into training (excluding zeroshot/fewshot overrides)
[training]
example = "train"

# Zeroshot specifications - entire cell types go to val or test
[zeroshot]

# Fewshot specifications - explicit perturbation lists
[fewshot]

[fewshot."example.CT4"]
val = ["TARGET3"]
test = ["TARGET4", "TARGET5"]  # can overlap with val

```

15./data3/fanpeishan/state/examples/zeroshot.toml

```toml
# Dataset paths - maps dataset names to their directories
[datasets]
example = "/home/aadduri/state/examples"

# Training specifications
# All cell types in a dataset automatically go into training (excluding zeroshot/fewshot overrides)
[training]
example = "train"

# Zeroshot specifications - entire cell types go to val or test
[zeroshot]
"example.CT3" = "test"

# Fewshot specifications - explicit perturbation lists
[fewshot]

```

16./data3/fanpeishan/state/examples/mixed.toml

```toml
# Dataset paths - maps dataset names to their directories
[datasets]
example = "/home/aadduri/state/examples"

# Training specifications
# All cell types in a dataset automatically go into training (excluding zeroshot/fewshot overrides)
[training]
example = "train"

# Zeroshot specifications - entire cell types go to val or test
[zeroshot]
"example.CT3" = "test"

# Fewshot specifications - explicit perturbation lists
[fewshot]

[fewshot."example.CT4"]
val = ["TARGET3"]
test = ["TARGET4", "TARGET5"]  # can overlap with val
```

17./data3/fanpeishan/state/.venv/lib/python3.11/site-packages/toml/decoder.py

```python
import datetime
import io
from os import linesep
import re
import sys

from toml.tz import TomlTz

if sys.version_info < (3,):
    _range = xrange  # noqa: F821
else:
    unicode = str
    _range = range
    basestring = str
    unichr = chr


def _detect_pathlib_path(p):
    if (3, 4) <= sys.version_info:
        import pathlib
        if isinstance(p, pathlib.PurePath):
            return True
    return False


def _ispath(p):
    if isinstance(p, (bytes, basestring)):
        return True
    return _detect_pathlib_path(p)


def _getpath(p):
    if (3, 6) <= sys.version_info:
        import os
        return os.fspath(p)
    if _detect_pathlib_path(p):
        return str(p)
    return p


try:
    FNFError = FileNotFoundError
except NameError:
    FNFError = IOError


TIME_RE = re.compile(r"([0-9]{2}):([0-9]{2}):([0-9]{2})(\.([0-9]{3,6}))?")


class TomlDecodeError(ValueError):
    """Base toml Exception / Error."""

    def __init__(self, msg, doc, pos):
        lineno = doc.count('\n', 0, pos) + 1
        colno = pos - doc.rfind('\n', 0, pos)
        emsg = '{} (line {} column {} char {})'.format(msg, lineno, colno, pos)
        ValueError.__init__(self, emsg)
        self.msg = msg
        self.doc = doc
        self.pos = pos
        self.lineno = lineno
        self.colno = colno


# Matches a TOML number, which allows underscores for readability
_number_with_underscores = re.compile('([0-9])(_([0-9]))*')


class CommentValue(object):
    def __init__(self, val, comment, beginline, _dict):
        self.val = val
        separator = "\n" if beginline else " "
        self.comment = separator + comment
        self._dict = _dict

    def __getitem__(self, key):
        return self.val[key]

    def __setitem__(self, key, value):
        self.val[key] = value

    def dump(self, dump_value_func):
        retstr = dump_value_func(self.val)
        if isinstance(self.val, self._dict):
            return self.comment + "\n" + unicode(retstr)
        else:
            return unicode(retstr) + self.comment


def _strictly_valid_num(n):
    n = n.strip()
    if not n:
        return False
    if n[0] == '_':
        return False
    if n[-1] == '_':
        return False
    if "_." in n or "._" in n:
        return False
    if len(n) == 1:
        return True
    if n[0] == '0' and n[1] not in ['.', 'o', 'b', 'x']:
        return False
    if n[0] == '+' or n[0] == '-':
        n = n[1:]
        if len(n) > 1 and n[0] == '0' and n[1] != '.':
            return False
    if '__' in n:
        return False
    return True


def load(f, _dict=dict, decoder=None):
    """Parses named file or files as toml and returns a dictionary

    Args:
        f: Path to the file to open, array of files to read into single dict
           or a file descriptor
        _dict: (optional) Specifies the class of the returned toml dictionary
        decoder: The decoder to use

    Returns:
        Parsed toml file represented as a dictionary

    Raises:
        TypeError -- When f is invalid type
        TomlDecodeError: Error while decoding toml
        IOError / FileNotFoundError -- When an array with no valid (existing)
        (Python 2 / Python 3)          file paths is passed
    """

    if _ispath(f):
        with io.open(_getpath(f), encoding='utf-8') as ffile:
            return loads(ffile.read(), _dict, decoder)
    elif isinstance(f, list):
        from os import path as op
        from warnings import warn
        if not [path for path in f if op.exists(path)]:
            error_msg = "Load expects a list to contain filenames only."
            error_msg += linesep
            error_msg += ("The list needs to contain the path of at least one "
                          "existing file.")
            raise FNFError(error_msg)
        if decoder is None:
            decoder = TomlDecoder(_dict)
        d = decoder.get_empty_table()
        for l in f:  # noqa: E741
            if op.exists(l):
                d.update(load(l, _dict, decoder))
            else:
                warn("Non-existent filename in list with at least one valid "
                     "filename")
        return d
    else:
        try:
            return loads(f.read(), _dict, decoder)
        except AttributeError:
            raise TypeError("You can only load a file descriptor, filename or "
                            "list")


_groupname_re = re.compile(r'^[A-Za-z0-9_-]+$')


def loads(s, _dict=dict, decoder=None):
    """Parses string as toml

    Args:
        s: String to be parsed
        _dict: (optional) Specifies the class of the returned toml dictionary

    Returns:
        Parsed toml file represented as a dictionary

    Raises:
        TypeError: When a non-string is passed
        TomlDecodeError: Error while decoding toml
    """

    implicitgroups = []
    if decoder is None:
        decoder = TomlDecoder(_dict)
    retval = decoder.get_empty_table()
    currentlevel = retval
    if not isinstance(s, basestring):
        raise TypeError("Expecting something like a string")

    if not isinstance(s, unicode):
        s = s.decode('utf8')

    original = s
    sl = list(s)
    openarr = 0
    openstring = False
    openstrchar = ""
    multilinestr = False
    arrayoftables = False
    beginline = True
    keygroup = False
    dottedkey = False
    keyname = 0
    key = ''
    prev_key = ''
    line_no = 1

    for i, item in enumerate(sl):
        if item == '\r' and sl[i + 1] == '\n':
            sl[i] = ' '
            continue
        if keyname:
            key += item
            if item == '\n':
                raise TomlDecodeError("Key name found without value."
                                      " Reached end of line.", original, i)
            if openstring:
                if item == openstrchar:
                    oddbackslash = False
                    k = 1
                    while i >= k and sl[i - k] == '\\':
                        oddbackslash = not oddbackslash
                        k += 1
                    if not oddbackslash:
                        keyname = 2
                        openstring = False
                        openstrchar = ""
                continue
            elif keyname == 1:
                if item.isspace():
                    keyname = 2
                    continue
                elif item == '.':
                    dottedkey = True
                    continue
                elif item.isalnum() or item == '_' or item == '-':
                    continue
                elif (dottedkey and sl[i - 1] == '.' and
                      (item == '"' or item == "'")):
                    openstring = True
                    openstrchar = item
                    continue
            elif keyname == 2:
                if item.isspace():
                    if dottedkey:
                        nextitem = sl[i + 1]
                        if not nextitem.isspace() and nextitem != '.':
                            keyname = 1
                    continue
                if item == '.':
                    dottedkey = True
                    nextitem = sl[i + 1]
                    if not nextitem.isspace() and nextitem != '.':
                        keyname = 1
                    continue
            if item == '=':
                keyname = 0
                prev_key = key[:-1].rstrip()
                key = ''
                dottedkey = False
            else:
                raise TomlDecodeError("Found invalid character in key name: '" +
                                      item + "'. Try quoting the key name.",
                                      original, i)
        if item == "'" and openstrchar != '"':
            k = 1
            try:
                while sl[i - k] == "'":
                    k += 1
                    if k == 3:
                        break
            except IndexError:
                pass
            if k == 3:
                multilinestr = not multilinestr
                openstring = multilinestr
            else:
                openstring = not openstring
            if openstring:
                openstrchar = "'"
            else:
                openstrchar = ""
        if item == '"' and openstrchar != "'":
            oddbackslash = False
            k = 1
            tripquote = False
            try:
                while sl[i - k] == '"':
                    k += 1
                    if k == 3:
                        tripquote = True
                        break
                if k == 1 or (k == 3 and tripquote):
                    while sl[i - k] == '\\':
                        oddbackslash = not oddbackslash
                        k += 1
            except IndexError:
                pass
            if not oddbackslash:
                if tripquote:
                    multilinestr = not multilinestr
                    openstring = multilinestr
                else:
                    openstring = not openstring
            if openstring:
                openstrchar = '"'
            else:
                openstrchar = ""
        if item == '#' and (not openstring and not keygroup and
                            not arrayoftables):
            j = i
            comment = ""
            try:
                while sl[j] != '\n':
                    comment += s[j]
                    sl[j] = ' '
                    j += 1
            except IndexError:
                break
            if not openarr:
                decoder.preserve_comment(line_no, prev_key, comment, beginline)
        if item == '[' and (not openstring and not keygroup and
                            not arrayoftables):
            if beginline:
                if len(sl) > i + 1 and sl[i + 1] == '[':
                    arrayoftables = True
                else:
                    keygroup = True
            else:
                openarr += 1
        if item == ']' and not openstring:
            if keygroup:
                keygroup = False
            elif arrayoftables:
                if sl[i - 1] == ']':
                    arrayoftables = False
            else:
                openarr -= 1
        if item == '\n':
            if openstring or multilinestr:
                if not multilinestr:
                    raise TomlDecodeError("Unbalanced quotes", original, i)
                if ((sl[i - 1] == "'" or sl[i - 1] == '"') and (
                        sl[i - 2] == sl[i - 1])):
                    sl[i] = sl[i - 1]
                    if sl[i - 3] == sl[i - 1]:
                        sl[i - 3] = ' '
            elif openarr:
                sl[i] = ' '
            else:
                beginline = True
            line_no += 1
        elif beginline and sl[i] != ' ' and sl[i] != '\t':
            beginline = False
            if not keygroup and not arrayoftables:
                if sl[i] == '=':
                    raise TomlDecodeError("Found empty keyname. ", original, i)
                keyname = 1
                key += item
    if keyname:
        raise TomlDecodeError("Key name found without value."
                              " Reached end of file.", original, len(s))
    if openstring:  # reached EOF and have an unterminated string
        raise TomlDecodeError("Unterminated string found."
                              " Reached end of file.", original, len(s))
    s = ''.join(sl)
    s = s.split('\n')
    multikey = None
    multilinestr = ""
    multibackslash = False
    pos = 0
    for idx, line in enumerate(s):
        if idx > 0:
            pos += len(s[idx - 1]) + 1

        decoder.embed_comments(idx, currentlevel)

        if not multilinestr or multibackslash or '\n' not in multilinestr:
            line = line.strip()
        if line == "" and (not multikey or multibackslash):
            continue
        if multikey:
            if multibackslash:
                multilinestr += line
            else:
                multilinestr += line
            multibackslash = False
            closed = False
            if multilinestr[0] == '[':
                closed = line[-1] == ']'
            elif len(line) > 2:
                closed = (line[-1] == multilinestr[0] and
                          line[-2] == multilinestr[0] and
                          line[-3] == multilinestr[0])
            if closed:
                try:
                    value, vtype = decoder.load_value(multilinestr)
                except ValueError as err:
                    raise TomlDecodeError(str(err), original, pos)
                currentlevel[multikey] = value
                multikey = None
                multilinestr = ""
            else:
                k = len(multilinestr) - 1
                while k > -1 and multilinestr[k] == '\\':
                    multibackslash = not multibackslash
                    k -= 1
                if multibackslash:
                    multilinestr = multilinestr[:-1]
                else:
                    multilinestr += "\n"
            continue
        if line[0] == '[':
            arrayoftables = False
            if len(line) == 1:
                raise TomlDecodeError("Opening key group bracket on line by "
                                      "itself.", original, pos)
            if line[1] == '[':
                arrayoftables = True
                line = line[2:]
                splitstr = ']]'
            else:
                line = line[1:]
                splitstr = ']'
            i = 1
            quotesplits = decoder._get_split_on_quotes(line)
            quoted = False
            for quotesplit in quotesplits:
                if not quoted and splitstr in quotesplit:
                    break
                i += quotesplit.count(splitstr)
                quoted = not quoted
            line = line.split(splitstr, i)
            if len(line) < i + 1 or line[-1].strip() != "":
                raise TomlDecodeError("Key group not on a line by itself.",
                                      original, pos)
            groups = splitstr.join(line[:-1]).split('.')
            i = 0
            while i < len(groups):
                groups[i] = groups[i].strip()
                if len(groups[i]) > 0 and (groups[i][0] == '"' or
                                           groups[i][0] == "'"):
                    groupstr = groups[i]
                    j = i + 1
                    while ((not groupstr[0] == groupstr[-1]) or
                           len(groupstr) == 1):
                        j += 1
                        if j > len(groups) + 2:
                            raise TomlDecodeError("Invalid group name '" +
                                                  groupstr + "' Something " +
                                                  "went wrong.", original, pos)
                        groupstr = '.'.join(groups[i:j]).strip()
                    groups[i] = groupstr[1:-1]
                    groups[i + 1:j] = []
                else:
                    if not _groupname_re.match(groups[i]):
                        raise TomlDecodeError("Invalid group name '" +
                                              groups[i] + "'. Try quoting it.",
                                              original, pos)
                i += 1
            currentlevel = retval
            for i in _range(len(groups)):
                group = groups[i]
                if group == "":
                    raise TomlDecodeError("Can't have a keygroup with an empty "
                                          "name", original, pos)
                try:
                    currentlevel[group]
                    if i == len(groups) - 1:
                        if group in implicitgroups:
                            implicitgroups.remove(group)
                            if arrayoftables:
                                raise TomlDecodeError("An implicitly defined "
                                                      "table can't be an array",
                                                      original, pos)
                        elif arrayoftables:
                            currentlevel[group].append(decoder.get_empty_table()
                                                       )
                        else:
                            raise TomlDecodeError("What? " + group +
                                                  " already exists?" +
                                                  str(currentlevel),
                                                  original, pos)
                except TypeError:
                    currentlevel = currentlevel[-1]
                    if group not in currentlevel:
                        currentlevel[group] = decoder.get_empty_table()
                        if i == len(groups) - 1 and arrayoftables:
                            currentlevel[group] = [decoder.get_empty_table()]
                except KeyError:
                    if i != len(groups) - 1:
                        implicitgroups.append(group)
                    currentlevel[group] = decoder.get_empty_table()
                    if i == len(groups) - 1 and arrayoftables:
                        currentlevel[group] = [decoder.get_empty_table()]
                currentlevel = currentlevel[group]
                if arrayoftables:
                    try:
                        currentlevel = currentlevel[-1]
                    except KeyError:
                        pass
        elif line[0] == "{":
            if line[-1] != "}":
                raise TomlDecodeError("Line breaks are not allowed in inline"
                                      "objects", original, pos)
            try:
                decoder.load_inline_object(line, currentlevel, multikey,
                                           multibackslash)
            except ValueError as err:
                raise TomlDecodeError(str(err), original, pos)
        elif "=" in line:
            try:
                ret = decoder.load_line(line, currentlevel, multikey,
                                        multibackslash)
            except ValueError as err:
                raise TomlDecodeError(str(err), original, pos)
            if ret is not None:
                multikey, multilinestr, multibackslash = ret
    return retval


def _load_date(val):
    microsecond = 0
    tz = None
    try:
        if len(val) > 19:
            if val[19] == '.':
                if val[-1].upper() == 'Z':
                    subsecondval = val[20:-1]
                    tzval = "Z"
                else:
                    subsecondvalandtz = val[20:]
                    if '+' in subsecondvalandtz:
                        splitpoint = subsecondvalandtz.index('+')
                        subsecondval = subsecondvalandtz[:splitpoint]
                        tzval = subsecondvalandtz[splitpoint:]
                    elif '-' in subsecondvalandtz:
                        splitpoint = subsecondvalandtz.index('-')
                        subsecondval = subsecondvalandtz[:splitpoint]
                        tzval = subsecondvalandtz[splitpoint:]
                    else:
                        tzval = None
                        subsecondval = subsecondvalandtz
                if tzval is not None:
                    tz = TomlTz(tzval)
                microsecond = int(int(subsecondval) *
                                  (10 ** (6 - len(subsecondval))))
            else:
                tz = TomlTz(val[19:])
    except ValueError:
        tz = None
    if "-" not in val[1:]:
        return None
    try:
        if len(val) == 10:
            d = datetime.date(
                int(val[:4]), int(val[5:7]),
                int(val[8:10]))
        else:
            d = datetime.datetime(
                int(val[:4]), int(val[5:7]),
                int(val[8:10]), int(val[11:13]),
                int(val[14:16]), int(val[17:19]), microsecond, tz)
    except ValueError:
        return None
    return d


def _load_unicode_escapes(v, hexbytes, prefix):
    skip = False
    i = len(v) - 1
    while i > -1 and v[i] == '\\':
        skip = not skip
        i -= 1
    for hx in hexbytes:
        if skip:
            skip = False
            i = len(hx) - 1
            while i > -1 and hx[i] == '\\':
                skip = not skip
                i -= 1
            v += prefix
            v += hx
            continue
        hxb = ""
        i = 0
        hxblen = 4
        if prefix == "\\U":
            hxblen = 8
        hxb = ''.join(hx[i:i + hxblen]).lower()
        if hxb.strip('0123456789abcdef'):
            raise ValueError("Invalid escape sequence: " + hxb)
        if hxb[0] == "d" and hxb[1].strip('01234567'):
            raise ValueError("Invalid escape sequence: " + hxb +
                             ". Only scalar unicode points are allowed.")
        v += unichr(int(hxb, 16))
        v += unicode(hx[len(hxb):])
    return v


# Unescape TOML string values.

# content after the \
_escapes = ['0', 'b', 'f', 'n', 'r', 't', '"']
# What it should be replaced by
_escapedchars = ['\0', '\b', '\f', '\n', '\r', '\t', '\"']
# Used for substitution
_escape_to_escapedchars = dict(zip(_escapes, _escapedchars))


def _unescape(v):
    """Unescape characters in a TOML string."""
    i = 0
    backslash = False
    while i < len(v):
        if backslash:
            backslash = False
            if v[i] in _escapes:
                v = v[:i - 1] + _escape_to_escapedchars[v[i]] + v[i + 1:]
            elif v[i] == '\\':
                v = v[:i - 1] + v[i:]
            elif v[i] == 'u' or v[i] == 'U':
                i += 1
            else:
                raise ValueError("Reserved escape sequence used")
            continue
        elif v[i] == '\\':
            backslash = True
        i += 1
    return v


class InlineTableDict(object):
    """Sentinel subclass of dict for inline tables."""


class TomlDecoder(object):

    def __init__(self, _dict=dict):
        self._dict = _dict

    def get_empty_table(self):
        return self._dict()

    def get_empty_inline_table(self):
        class DynamicInlineTableDict(self._dict, InlineTableDict):
            """Concrete sentinel subclass for inline tables.
            It is a subclass of _dict which is passed in dynamically at load
            time

            It is also a subclass of InlineTableDict
            """

        return DynamicInlineTableDict()

    def load_inline_object(self, line, currentlevel, multikey=False,
                           multibackslash=False):
        candidate_groups = line[1:-1].split(",")
        groups = []
        if len(candidate_groups) == 1 and not candidate_groups[0].strip():
            candidate_groups.pop()
        while len(candidate_groups) > 0:
            candidate_group = candidate_groups.pop(0)
            try:
                _, value = candidate_group.split('=', 1)
            except ValueError:
                raise ValueError("Invalid inline table encountered")
            value = value.strip()
            if ((value[0] == value[-1] and value[0] in ('"', "'")) or (
                    value[0] in '-0123456789' or
                    value in ('true', 'false') or
                    (value[0] == "[" and value[-1] == "]") or
                    (value[0] == '{' and value[-1] == '}'))):
                groups.append(candidate_group)
            elif len(candidate_groups) > 0:
                candidate_groups[0] = (candidate_group + "," +
                                       candidate_groups[0])
            else:
                raise ValueError("Invalid inline table value encountered")
        for group in groups:
            status = self.load_line(group, currentlevel, multikey,
                                    multibackslash)
            if status is not None:
                break

    def _get_split_on_quotes(self, line):
        doublequotesplits = line.split('"')
        quoted = False
        quotesplits = []
        if len(doublequotesplits) > 1 and "'" in doublequotesplits[0]:
            singlequotesplits = doublequotesplits[0].split("'")
            doublequotesplits = doublequotesplits[1:]
            while len(singlequotesplits) % 2 == 0 and len(doublequotesplits):
                singlequotesplits[-1] += '"' + doublequotesplits[0]
                doublequotesplits = doublequotesplits[1:]
                if "'" in singlequotesplits[-1]:
                    singlequotesplits = (singlequotesplits[:-1] +
                                         singlequotesplits[-1].split("'"))
            quotesplits += singlequotesplits
        for doublequotesplit in doublequotesplits:
            if quoted:
                quotesplits.append(doublequotesplit)
            else:
                quotesplits += doublequotesplit.split("'")
                quoted = not quoted
        return quotesplits

    def load_line(self, line, currentlevel, multikey, multibackslash):
        i = 1
        quotesplits = self._get_split_on_quotes(line)
        quoted = False
        for quotesplit in quotesplits:
            if not quoted and '=' in quotesplit:
                break
            i += quotesplit.count('=')
            quoted = not quoted
        pair = line.split('=', i)
        strictly_valid = _strictly_valid_num(pair[-1])
        if _number_with_underscores.match(pair[-1]):
            pair[-1] = pair[-1].replace('_', '')
        while len(pair[-1]) and (pair[-1][0] != ' ' and pair[-1][0] != '\t' and
                                 pair[-1][0] != "'" and pair[-1][0] != '"' and
                                 pair[-1][0] != '[' and pair[-1][0] != '{' and
                                 pair[-1].strip() != 'true' and
                                 pair[-1].strip() != 'false'):
            try:
                float(pair[-1])
                break
            except ValueError:
                pass
            if _load_date(pair[-1]) is not None:
                break
            if TIME_RE.match(pair[-1]):
                break
            i += 1
            prev_val = pair[-1]
            pair = line.split('=', i)
            if prev_val == pair[-1]:
                raise ValueError("Invalid date or number")
            if strictly_valid:
                strictly_valid = _strictly_valid_num(pair[-1])
        pair = ['='.join(pair[:-1]).strip(), pair[-1].strip()]
        if '.' in pair[0]:
            if '"' in pair[0] or "'" in pair[0]:
                quotesplits = self._get_split_on_quotes(pair[0])
                quoted = False
                levels = []
                for quotesplit in quotesplits:
                    if quoted:
                        levels.append(quotesplit)
                    else:
                        levels += [level.strip() for level in
                                   quotesplit.split('.')]
                    quoted = not quoted
            else:
                levels = pair[0].split('.')
            while levels[-1] == "":
                levels = levels[:-1]
            for level in levels[:-1]:
                if level == "":
                    continue
                if level not in currentlevel:
                    currentlevel[level] = self.get_empty_table()
                currentlevel = currentlevel[level]
            pair[0] = levels[-1].strip()
        elif (pair[0][0] == '"' or pair[0][0] == "'") and \
                (pair[0][-1] == pair[0][0]):
            pair[0] = _unescape(pair[0][1:-1])
        k, koffset = self._load_line_multiline_str(pair[1])
        if k > -1:
            while k > -1 and pair[1][k + koffset] == '\\':
                multibackslash = not multibackslash
                k -= 1
            if multibackslash:
                multilinestr = pair[1][:-1]
            else:
                multilinestr = pair[1] + "\n"
            multikey = pair[0]
        else:
            value, vtype = self.load_value(pair[1], strictly_valid)
        try:
            currentlevel[pair[0]]
            raise ValueError("Duplicate keys!")
        except TypeError:
            raise ValueError("Duplicate keys!")
        except KeyError:
            if multikey:
                return multikey, multilinestr, multibackslash
            else:
                currentlevel[pair[0]] = value

    def _load_line_multiline_str(self, p):
        poffset = 0
        if len(p) < 3:
            return -1, poffset
        if p[0] == '[' and (p.strip()[-1] != ']' and
                            self._load_array_isstrarray(p)):
            newp = p[1:].strip().split(',')
            while len(newp) > 1 and newp[-1][0] != '"' and newp[-1][0] != "'":
                newp = newp[:-2] + [newp[-2] + ',' + newp[-1]]
            newp = newp[-1]
            poffset = len(p) - len(newp)
            p = newp
        if p[0] != '"' and p[0] != "'":
            return -1, poffset
        if p[1] != p[0] or p[2] != p[0]:
            return -1, poffset
        if len(p) > 5 and p[-1] == p[0] and p[-2] == p[0] and p[-3] == p[0]:
            return -1, poffset
        return len(p) - 1, poffset

    def load_value(self, v, strictly_valid=True):
        if not v:
            raise ValueError("Empty value is invalid")
        if v == 'true':
            return (True, "bool")
        elif v.lower() == 'true':
            raise ValueError("Only all lowercase booleans allowed")
        elif v == 'false':
            return (False, "bool")
        elif v.lower() == 'false':
            raise ValueError("Only all lowercase booleans allowed")
        elif v[0] == '"' or v[0] == "'":
            quotechar = v[0]
            testv = v[1:].split(quotechar)
            triplequote = False
            triplequotecount = 0
            if len(testv) > 1 and testv[0] == '' and testv[1] == '':
                testv = testv[2:]
                triplequote = True
            closed = False
            for tv in testv:
                if tv == '':
                    if triplequote:
                        triplequotecount += 1
                    else:
                        closed = True
                else:
                    oddbackslash = False
                    try:
                        i = -1
                        j = tv[i]
                        while j == '\\':
                            oddbackslash = not oddbackslash
                            i -= 1
                            j = tv[i]
                    except IndexError:
                        pass
                    if not oddbackslash:
                        if closed:
                            raise ValueError("Found tokens after a closed " +
                                             "string. Invalid TOML.")
                        else:
                            if not triplequote or triplequotecount > 1:
                                closed = True
                            else:
                                triplequotecount = 0
            if quotechar == '"':
                escapeseqs = v.split('\\')[1:]
                backslash = False
                for i in escapeseqs:
                    if i == '':
                        backslash = not backslash
                    else:
                        if i[0] not in _escapes and (i[0] != 'u' and
                                                     i[0] != 'U' and
                                                     not backslash):
                            raise ValueError("Reserved escape sequence used")
                        if backslash:
                            backslash = False
                for prefix in ["\\u", "\\U"]:
                    if prefix in v:
                        hexbytes = v.split(prefix)
                        v = _load_unicode_escapes(hexbytes[0], hexbytes[1:],
                                                  prefix)
                v = _unescape(v)
            if len(v) > 1 and v[1] == quotechar and (len(v) < 3 or
                                                     v[1] == v[2]):
                v = v[2:-2]
            return (v[1:-1], "str")
        elif v[0] == '[':
            return (self.load_array(v), "array")
        elif v[0] == '{':
            inline_object = self.get_empty_inline_table()
            self.load_inline_object(v, inline_object)
            return (inline_object, "inline_object")
        elif TIME_RE.match(v):
            h, m, s, _, ms = TIME_RE.match(v).groups()
            time = datetime.time(int(h), int(m), int(s), int(ms) if ms else 0)
            return (time, "time")
        else:
            parsed_date = _load_date(v)
            if parsed_date is not None:
                return (parsed_date, "date")
            if not strictly_valid:
                raise ValueError("Weirdness with leading zeroes or "
                                 "underscores in your number.")
            itype = "int"
            neg = False
            if v[0] == '-':
                neg = True
                v = v[1:]
            elif v[0] == '+':
                v = v[1:]
            v = v.replace('_', '')
            lowerv = v.lower()
            if '.' in v or ('x' not in v and ('e' in v or 'E' in v)):
                if '.' in v and v.split('.', 1)[1] == '':
                    raise ValueError("This float is missing digits after "
                                     "the point")
                if v[0] not in '0123456789':
                    raise ValueError("This float doesn't have a leading "
                                     "digit")
                v = float(v)
                itype = "float"
            elif len(lowerv) == 3 and (lowerv == 'inf' or lowerv == 'nan'):
                v = float(v)
                itype = "float"
            if itype == "int":
                v = int(v, 0)
            if neg:
                return (0 - v, itype)
            return (v, itype)

    def bounded_string(self, s):
        if len(s) == 0:
            return True
        if s[-1] != s[0]:
            return False
        i = -2
        backslash = False
        while len(s) + i > 0:
            if s[i] == "\\":
                backslash = not backslash
                i -= 1
            else:
                break
        return not backslash

    def _load_array_isstrarray(self, a):
        a = a[1:-1].strip()
        if a != '' and (a[0] == '"' or a[0] == "'"):
            return True
        return False

    def load_array(self, a):
        atype = None
        retval = []
        a = a.strip()
        if '[' not in a[1:-1] or "" != a[1:-1].split('[')[0].strip():
            strarray = self._load_array_isstrarray(a)
            if not a[1:-1].strip().startswith('{'):
                a = a[1:-1].split(',')
            else:
                # a is an inline object, we must find the matching parenthesis
                # to define groups
                new_a = []
                start_group_index = 1
                end_group_index = 2
                open_bracket_count = 1 if a[start_group_index] == '{' else 0
                in_str = False
                while end_group_index < len(a[1:]):
                    if a[end_group_index] == '"' or a[end_group_index] == "'":
                        if in_str:
                            backslash_index = end_group_index - 1
                            while (backslash_index > -1 and
                                   a[backslash_index] == '\\'):
                                in_str = not in_str
                                backslash_index -= 1
                        in_str = not in_str
                    if not in_str and a[end_group_index] == '{':
                        open_bracket_count += 1
                    if in_str or a[end_group_index] != '}':
                        end_group_index += 1
                        continue
                    elif a[end_group_index] == '}' and open_bracket_count > 1:
                        open_bracket_count -= 1
                        end_group_index += 1
                        continue

                    # Increase end_group_index by 1 to get the closing bracket
                    end_group_index += 1

                    new_a.append(a[start_group_index:end_group_index])

                    # The next start index is at least after the closing
                    # bracket, a closing bracket can be followed by a comma
                    # since we are in an array.
                    start_group_index = end_group_index + 1
                    while (start_group_index < len(a[1:]) and
                           a[start_group_index] != '{'):
                        start_group_index += 1
                    end_group_index = start_group_index + 1
                a = new_a
            b = 0
            if strarray:
                while b < len(a) - 1:
                    ab = a[b].strip()
                    while (not self.bounded_string(ab) or
                           (len(ab) > 2 and
                            ab[0] == ab[1] == ab[2] and
                            ab[-2] != ab[0] and
                            ab[-3] != ab[0])):
                        a[b] = a[b] + ',' + a[b + 1]
                        ab = a[b].strip()
                        if b < len(a) - 2:
                            a = a[:b + 1] + a[b + 2:]
                        else:
                            a = a[:b + 1]
                    b += 1
        else:
            al = list(a[1:-1])
            a = []
            openarr = 0
            j = 0
            for i in _range(len(al)):
                if al[i] == '[':
                    openarr += 1
                elif al[i] == ']':
                    openarr -= 1
                elif al[i] == ',' and not openarr:
                    a.append(''.join(al[j:i]))
                    j = i + 1
            a.append(''.join(al[j:]))
        for i in _range(len(a)):
            a[i] = a[i].strip()
            if a[i] != '':
                nval, ntype = self.load_value(a[i])
                if atype:
                    if ntype != atype:
                        raise ValueError("Not a homogeneous array")
                else:
                    atype = ntype
                retval.append(nval)
        return retval

    def preserve_comment(self, line_no, key, comment, beginline):
        pass

    def embed_comments(self, idx, currentlevel):
        pass


class TomlPreserveCommentDecoder(TomlDecoder):

    def __init__(self, _dict=dict):
        self.saved_comments = {}
        super(TomlPreserveCommentDecoder, self).__init__(_dict)

    def preserve_comment(self, line_no, key, comment, beginline):
        self.saved_comments[line_no] = (key, comment, beginline)

    def embed_comments(self, idx, currentlevel):
        if idx not in self.saved_comments:
            return

        key, comment, beginline = self.saved_comments[idx]
        currentlevel[key] = CommentValue(currentlevel[key], comment, beginline,
                                         self._dict)

```

18./data3/fanpeishan/state/src/state/configs/config.yaml

```yaml
# This is a template used in the application to generating the config file for
# training tasks
defaults:
  - data: perturbation
  - model: state
  - training: default
  - wandb: default
  - _self_
  

# output_dir must be an absolute path (so that launch scripts are fully descriptive)
name: debug
output_dir: /data3/fanpeishan/state/for_state/run_results/run27/configs_dir
use_wandb: false
overwrite: false
return_adatas: false
pred_adata_path: null
true_adata_path: null

# don't save hydra output
hydra:
  output_subdir: null
  run:
    dir: .
  job_logging:
    formatters:
      simple:
        format: "[%(levelname)s] %(message)s"  # Simple format for logging
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        level: INFO
        stream: ext://sys.stdout
    root:
      level: INFO
    loggers:
      __main__:
        level: DEBUG
        handlers: [console]
        propagate: false

```

19./data3/fanpeishan/state/src/state/configs/data/perturbation.yaml

```yaml
name: PerturbationDataModule
kwargs:
  toml_config_path: null
  embed_key: X_hvg # embed_key: null
  output_space: gene # output_space: all
  pert_rep: onehot
  basal_rep: sample
  num_workers: 8
  pin_memory: true
  n_basal_samples: 8 #1
  basal_mapping_strategy: random
  should_yield_control_cells: true
  batch_col: gem_group # batch_col: plate
  pert_col: drugname_drugconc # pert_col: gene
  cell_type_key: cell_line #cell_type_key: cell_type
  control_pert: ('DMSO_TF', 0.0, 'uM') # control_pert: DMSO_TF
  map_controls: true # for a control cell, should we use it as the target (learn identity) or sample a control?
  perturbation_features_file: null
  store_raw_basal: false
  int_counts: false
  barcode: true
output_dir: null
debug: true

```

20./data3/fanpeishan/state/src/state/configs/model/state.yaml

```yaml
name: state
checkpoint: null
device: cuda

kwargs:
  cell_set_len: 512
  blur: 0.05
  hidden_dim: 696      # hidden dimension going into the transformer backbone
  loss: energy
  confidence_head: False
  n_encoder_layers: 1
  n_decoder_layers: 1
  predict_residual: True
  softplus: True
  freeze_pert_backbone: False
  transformer_decoder: False
  finetune_vci_decoder: False
  residual_decoder: False
  batch_encoder: False
  use_batch_token: False
  nb_decoder: False
  mask_attn: False
  use_effect_gating_token: False
  distributional_loss: energy
  init_from: null
  transformer_backbone_key: llama
  transformer_backbone_kwargs:
      bidirectional_attention: false
      max_position_embeddings: ${model.kwargs.cell_set_len}
      hidden_size: ${model.kwargs.hidden_dim}
      intermediate_size: 2784
      num_hidden_layers: 8
      num_attention_heads: 12
      num_key_value_heads: 12
      head_dim: 58
      use_cache: false
      attention_dropout: 0.0
      hidden_dropout: 0.0
      layer_norm_eps: 1e-6
      pad_token_id: 0
      bos_token_id: 1
      eos_token_id: 2
      tie_word_embeddings: false
      rotary_dim: 0
      use_rotary_embeddings: false
  lora:
      enable: false
      r: 16
      alpha: 32
      dropout: 0.05
      bias: none
      target: auto
      adapt_mlp: false
      task_type: FEATURE_EXTRACTION
      merge_on_eval: false

```

21./data3/fanpeishan/state/src/state/configs/training/default.yaml

```yaml
wandb_track: false
weight_decay: 0.0005
batch_size: 16
lr: 1e-4
max_steps: 40000
train_seed: 42
val_freq: 2000
ckpt_every_n_steps: 2000
gradient_clip_val: 10 # 0 means no clipping
loss_fn: mse
devices: 1  # Number of GPUs to use for training
strategy: auto  # DDP strategy for multi-GPU training
use_mfu: true
mfu_kwargs:
    available_flops: 60e12
    use_backward: true
    logging_interval: 10
    window_size: 2
cumulative_flops_use_backward: true
```

请你根据当前的输出，仔细判断输出中是否有不正确的地方，判断st模型的训练过程是否正确。提示：每一处不合理的地方都请指出来，并给出不合理的原因，以及解决方案。

```bash
(arc-state) (vcc) fanpeishan@ubun:/data3/fanpeishan/state$ export CUDA_VISIBLE_DEVICES=6
state tx train \                             
  name="run28_v1" \
  data.kwargs.toml_config_path="/data3/fanpeishan/state/for_state/run__commands/tomls/run28.toml" \
  data.output_dir="/data3/fanpeishan/state/for_state/run_results/run28" \
  model="state"
Seed set to 42
DEBUG: Batch Size passed to DataModule is: 16/data1/fanpeishan/State-Tahoe-Filtered/c37.h5ad/data1/fanpeishan/State-Tahoe-Filtered/c38.h5adProcessing c38:   0%|                              | 0/1 [00:00<?, ?it/s]
No cell barcode information found in /data1/fanpeishan/State-Tahoe-Filtered/c38.h5ad. Generating generic barcodes.
Processed c38: 2360694 train, 0 val, 0 test
Processing c38: 100%|████████████████████████| 1/1 [00:01<00:00,  1.16s/it]
c37 CVCL_1547 {'val': 2, 'test': 2}
Processing c37:   0%|| 0/1 [00:00<?, ?it/s]
No cell barcode information found in /data1/fanpeishan/State-Tahoe-Filtered/c37.h5ad. Generating generic barcodes.
Processed c37: 1817013 train, 54983 val, 54251 test
Processing c37: 100%|████████████████| 1/1 [00:00<00:00,  1.17it/s]
num_workers: 8
batch size: None
StateTransitionPerturbationModel(
  (loss_fn): SamplesLoss()
  (gene_decoder): LatentToGeneDecoder(
    (decoder): Sequential(
      (0): Linear(in_features=2000, out_features=1024, bias=True)
      (1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (2): GELU(approximate='none')
      (3): Dropout(p=0.1, inplace=False)
      (4): Linear(in_features=1024, out_features=1024, bias=True)
      (5): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (6): GELU(approximate='none')
      (7): Dropout(p=0.1, inplace=False)
      (8): Linear(in_features=1024, out_features=512, bias=True)
      (9): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (10): GELU(approximate='none')
      (11): Dropout(p=0.1, inplace=False)
      (12): Linear(in_features=512, out_features=2000, bias=True)
      (13): ReLU())
  )
  (pert_encoder): Sequential(
    (0): Linear(in_features=1137, out_features=696, bias=True)
  )
  (basal_encoder): Sequential(
    (0): Linear(in_features=2000, out_features=696, bias=True)
  )
  (transformer_backbone): LlamaModel(
    (embed_tokens): Embedding(32000, 696, padding_idx=0)
    (layers): ModuleList(
      (0-7): 8 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=696, out_features=696, bias=False)
          (k_proj): Linear(in_features=696, out_features=696, bias=False)
          (v_proj): Linear(in_features=696, out_features=696, bias=False)
          (o_proj): Linear(in_features=696, out_features=696, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=696, out_features=2784, bias=False)
          (up_proj): Linear(in_features=696, out_features=2784, bias=False)
          (down_proj): Linear(in_features=2784, out_features=696, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((696,), eps=1e-06)
        (post_attention_layernorm): LlamaRMSNorm((696,), eps=1e-06)
      )
    )
    (norm): LlamaRMSNorm((696,), eps=1e-06)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (project_out): Sequential(
    (0): Linear(in_features=696, out_features=2000, bias=True)
  )
  (relu): ReLU()
)
Model created. Estimated params size: 0.34 GB
ModelFLOPSUtilizationCallback: Using available flops: 60000000000000.0
ModelFLOPSUtilizationCallback: Using use_backward: True
ModelFLOPSUtilizationCallback: Using logging interval: 10
ModelFLOPSUtilizationCallback: Using cell set length: 512
ModelFLOPSUtilizationCallback: Using window size: 2
Building trainer with kwargs: {'accelerator': 'gpu', 'devices': 1, 'strategy': 'auto', 'max_steps': 40000, 'check_val_every_n_epoch': None, 'val_check_interval': 2000, 'logger': [<state.tx.utils.RobustCSVLogger object at 0x7fee77aaa6d0>], 'plugins': [], 'callbacks': [<lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint object at 0x7fee2e012310>, <lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint object at 0x7fee2e39b990>, <state.tx.callbacks.batch_speed_monitor.BatchSpeedMonitorCallback object at 0x7fee2dde7990>, <state.tx.callbacks.GradNormCallback object at 0x7fee775fb710>, <state.tx.callbacks.model_flops_utilization.ModelFLOPSUtilizationCallback object at 0x7fee2e195350>, <state.tx.callbacks.cumulative_flops.CumulativeFLOPSCallback object at 0x7fee77611010>], 'gradient_clip_val': 10, 'use_distributed_sampler': False}
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
Trainer built successfully
Model device: cpu
CUDA memory allocated: 0.00 GB
CUDA memory reserved: 0.00 GB
About to call trainer.fit() with checkpoint_path=None
ModelFLOPSUtilizationCallback: Initializing throughput tracker with world_size: 1
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [6]

  | Name                 | Type                | Params | Mode  | FLOPs
-----------------------------------------------------------------------------
0 | loss_fn              | SamplesLoss         | 0      | train | 0    
1 | gene_decoder         | LatentToGeneDecoder | 4.7 M  | train | 0    
2 | pert_encoder         | Sequential          | 792 K  | train | 0    
3 | basal_encoder        | Sequential          | 1.4 M  | train | 0    
4 | transformer_backbone | LlamaModel          | 84.3 M | train | 0    
5 | project_out          | Sequential          | 1.4 M  | train | 0    
6 | relu                 | ReLU                | 0      | train | 0    
-----------------------------------------------------------------------------
70.3 M    Trainable params
22.3 M    Non-trainable params
92.5 M    Total params
370.089   Total estimated model params size (MB)
133       Modules in train mode
0         Modules in eval mode
0         Total Flops
Epoch 0:  20%|███████████▌                  | 119/582 [06:41<26:00,  0.30it/s, v_num=0, mfu (%)=7.110]
```

提示：如果我给你的代码文件不完整，你对其中某些细节无法准确判断，请告诉我你需要的文件，我会发给你。请一定不要对你不知道的文件内容进行猜测。
