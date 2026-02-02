五个 state/src/state/tx/models 目录下的文件代码。

1./data3/fanpeishan/state/src/state/tx/models/base.py文件源代码：

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

2./data3/fanpeishan/state/src/state/tx/models/state_transition.py文件源代码：

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

3./data3/fanpeishan/state/src/state/tx/models/decoders.py文件源代码：

```python
import logging

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ...emb.finetune_decoder import Finetune

logger = logging.getLogger(__name__)


class FinetuneVCICountsDecoder(nn.Module):
    def __init__(
        self,
        genes,
        # model_loc="/large_storage/ctc/userspace/aadduri/vci/checkpoint/rda_tabular_counts_2048_new/step=950000.ckpt",
        # config="/large_storage/ctc/userspace/aadduri/vci/checkpoint/rda_tabular_counts_2048_new/tahoe_config.yaml",
        model_loc="/home/aadduri/vci_pretrain/vci_1.4.2.ckpt",
        config="/large_storage/ctc/userspace/aadduri/vci/checkpoint/large_1e-4_rda_tabular_counts_2048/crossds_config.yaml",
        read_depth=1200,
        latent_dim=1024,  # dimension of pretrained vci model
        hidden_dims=[512, 512, 512],  # hidden dimensions of the decoder
        dropout=0.1,
        basal_residual=False,
    ):
        super().__init__()
        self.genes = genes
        self.model_loc = model_loc
        self.config = config
        self.finetune = Finetune(OmegaConf.load(self.config))
        self.finetune.load_model(self.model_loc)
        self.read_depth = nn.Parameter(torch.tensor(read_depth, dtype=torch.float), requires_grad=False)
        self.basal_residual = basal_residual

        # layers = [
        #     nn.Linear(latent_dim, hidden_dims[0]),
        # ]

        # self.gene_lora = nn.Sequential(*layers)

        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], len(self.genes)),
            nn.ReLU(),
        )

        self.gene_decoder_proj = nn.Sequential(
            nn.Linear(len(self.genes), 128),
            nn.Linear(128, len(self.genes)),
        )

        self.binary_decoder = self.finetune.model.binary_decoder
        for param in self.binary_decoder.parameters():
            param.requires_grad = False

    def gene_dim(self):
        return len(self.genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, S, latent_dim].
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        batch_size, seq_len, latent_dim = x.shape
        x = x.view(batch_size * seq_len, latent_dim)

        # Get gene embeddings
        gene_embeds = self.finetune.get_gene_embedding(self.genes)

        # Handle RDA task counts
        use_rda = getattr(self.finetune.model.cfg.model, "rda", False)
        # Define your sub-batch size (tweak this based on your available memory)
        sub_batch_size = 16
        logprob_chunks = []  # to store outputs of each sub-batch

        for i in range(0, x.shape[0], sub_batch_size):
            # Get the sub-batch of latent vectors
            x_sub = x[i : i + sub_batch_size]

            # Create task_counts for the sub-batch if needed
            if use_rda:
                # task_counts_sub = torch.full(
                #     (x_sub.shape[0],), self.read_depth, device=x.device
                # )
                task_counts_sub = torch.ones((x_sub.shape[0],), device=x.device) * self.read_depth
            else:
                task_counts_sub = None

            # Compute merged embeddings for the sub-batch
            merged_embs_sub = self.finetune.model.resize_batch(x_sub, gene_embeds, task_counts_sub)

            # Run the binary decoder on the sub-batch
            logprobs_sub = self.binary_decoder(merged_embs_sub)

            # Squeeze the singleton dimension if needed
            if logprobs_sub.dim() == 3 and logprobs_sub.size(-1) == 1:
                logprobs_sub = logprobs_sub.squeeze(-1)

            # Collect the results
            logprob_chunks.append(logprobs_sub)

        # Concatenate the sub-batches back together
        logprobs = torch.cat(logprob_chunks, dim=0)

        # Reshape back to [B, S, gene_dim]
        decoded_gene = logprobs.view(batch_size, seq_len, len(self.genes))
        decoded_gene = decoded_gene + self.gene_decoder_proj(decoded_gene)
        # decoded_gene = torch.nn.functional.relu(decoded_gene)

        # # normalize the sum of decoded_gene to be read depth
        # decoded_gene = decoded_gene / decoded_gene.sum(dim=2, keepdim=True) * self.read_depth

        # decoded_gene = self.gene_lora(decoded_gene)
        # TODO: fix this to work with basal counts

        # add logic for basal_residual:
        decoded_x = self.latent_decoder(x)
        decoded_x = decoded_x.view(batch_size, seq_len, len(self.genes))

        # Pass through the additional decoder layers
        return decoded_gene + decoded_x

```

4./data3/fanpeishan/state/src/state/tx/models/utils.py文件源代码：

```python
from typing import Union

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, LlamaConfig, LlamaModel, PreTrainedModel

# LoRA / PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    TaskType = None  # type: ignore


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    activation: nn.Module = nn.ReLU,  # default to nn.ReLU class
) -> nn.Sequential:
    """
    Build an MLP of `n_layers` from `in_dim` to `out_dim`.
    ...
    """
    layers = []
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    if n_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())  # instantiate the class
        layers.append(nn.Dropout(dropout))

        # Intermediate layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())  # instantiate again
            layers.append(nn.Dropout(dropout))

        # Final layer
        layers.append(nn.Linear(hidden_dim, out_dim))

    return nn.Sequential(*layers)


def get_activation_class(name: str) -> nn.Module:
    """
    Given a string activation name, return the corresponding nn.Module class.

    Supported activation functions (add any more here):
    - ReLU
    - LeakyReLU
    - ELU
    - SELU
    - GELU
    """
    name = name.lower()

    if name == "relu":
        return nn.ReLU
    elif name == "leakyrelu":
        return nn.LeakyReLU
    elif name == "elu":
        return nn.ELU
    elif name == "selu":
        return nn.SELU
    elif name == "gelu":
        return nn.GELU
    # Add more as needed...
    else:
        raise ValueError(f"Unsupported activation function: {name}")


def get_loss_fn(loss: Union[str, nn.Module]) -> nn.Module:
    """
    Given a string loss function name, return the corresponding nn.Module class.

    Supported loss functions (add any more here):
    - MSELoss
    - L1Loss
    - SmoothL1Loss
    """
    if isinstance(loss, nn.Module):
        return loss

    loss = loss.lower()

    if loss == "mse":
        return nn.MSELoss()
    # Add more as needed...
    else:
        raise ValueError(f"Unsupported loss function: {loss}")


def get_transformer_backbone(key, kwargs) -> PreTrainedModel:
    kwargs = dict(kwargs or {})

    if key == "GPT2":
        config = GPT2Config(**kwargs)
        model = GPT2BidirectionalModel(config)

        # Zero out position embeddings and freeze them
        model.wpe.weight.requires_grad = False
        model.wte.weight.requires_grad = False
        model.wpe.weight.zero_()
        model.wte.weight.zero_()

        model_dim = config.n_embd
    elif key == "llama":
        bidirectional_attention = bool(kwargs.pop("bidirectional_attention", False))

        config = LlamaConfig(**kwargs)
        if bidirectional_attention:
            model = LlamaBidirectionalModel(config)
        else:
            model = LlamaModel(config)
        model_dim = config.hidden_size

        model.embed_tokens.weight.requires_grad = False
        model.embed_tokens.weight.zero_()
    else:
        raise ValueError(f"Unknown backbone key {key}")

    return model, model_dim


# -------------------------------
# LoRA utilities
# -------------------------------
def _default_lora_targets(backbone_key: str, adapt_mlp: bool) -> list[str]:
    """
    Choose target module names for LoRA injection based on backbone type.
    """
    k = backbone_key.lower()
    if k == "llama":
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if adapt_mlp:
            targets += ["gate_proj", "up_proj", "down_proj"]
        return targets
    if k == "gpt2":
        targets = ["c_attn", "c_proj"]
        if adapt_mlp:
            targets += ["mlp.c_fc", "mlp.c_proj"]
        return targets
    raise ValueError(f"Unsupported backbone for LoRA: {backbone_key}")


def apply_lora(model: PreTrainedModel, backbone_key: str, lora_cfg: dict | None) -> PreTrainedModel:
    """
    Apply LoRA adapters to a HuggingFace transformer model when enabled.
    If PEFT is unavailable or config is disabled, returns the original model.
    """
    if not lora_cfg or not lora_cfg.get("enable", False):
        return model

    if LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "peft is not installed but `lora.enable` is True. Add `peft` to dependencies."
        )

    target = lora_cfg.get("target", "auto")
    adapt_mlp = bool(lora_cfg.get("adapt_mlp", False))
    target_modules = (
        lora_cfg.get("target_modules")
        if target != "auto"
        else _default_lora_targets(backbone_key, adapt_mlp)
    )

    # Build PEFT LoRA config
    task_type_key = lora_cfg.get("task_type", "FEATURE_EXTRACTION")
    task_type = TaskType[task_type_key] if isinstance(task_type_key, str) else task_type_key

    config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias=lora_cfg.get("bias", "none"),
        target_modules=target_modules,
        task_type=task_type,
    )

    peft_model = get_peft_model(model, config)

    # Optional: print trainable params summary if available
    try:
        peft_model.print_trainable_parameters()
    except Exception:
        pass

    return peft_model


class NoRoPE(nn.Module):
    """
    A drop-in replacement for LlamaRotaryEmbedding that always returns:
      cos = all ones, sin = all zeros
    of shape (batch_size, seq_len, head_dim), so rotary has no effect.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor):
        # hidden_states: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _hidden_dim = hidden_states.shape

        # Create cos = ones, sin = zeros
        #   shape --> (batch_size, seq_len, head_dim)
        cos = hidden_states.new_ones(batch_size, seq_len, self.head_dim)
        sin = hidden_states.new_zeros(batch_size, seq_len, self.head_dim)
        return cos, sin


class LlamaBidirectionalModel(LlamaModel):
    """
    A drop-in replacement for LlamaModel with bidirectional attention.
    By overriding _update_causal_mask to return None, all tokens attend to each other.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.rotary_emb = NoRoPE(
            head_dim=config.head_dim,
        )
        
        # Explicitly disable causal attention
        self.config.is_causal = False
        # force every layer to be non-causal
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.is_causal = False   # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        output_attentions: bool = False,
    ):
        # By returning None, we disable any causal‐(look‐ahead) masking.
        # The only mask that remains is whatever "attention_mask" the user has passed
        # (e.g. padding‐mask), which will be handled by Flash/SDPA internally as non‐causal.
        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        cache_position: torch.LongTensor = None,
        **flash_attn_kwargs,
    ):
        flash_attn_kwargs["is_causal"] = False
        
        # If no attention_mask is provided, create an all-ones mask (no masking)
        # This ensures bidirectional attention with correct device/dtype
        if attention_mask is None:
            # Get batch size (B) and sequence length (S) from input_embeds if available, else from input_ids.
            # If neither is available, fall back to attention_mask=None and log a warning.
            B = None
            S = None
            if inputs_embeds is not None:
                B, S = inputs_embeds.size(0), inputs_embeds.size(1)
            if B and S:
                attention_mask = torch.ones((B, 1, S, S), dtype=torch.float, device=inputs_embeds.device)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )


class GPT2BidirectionalModel(GPT2Model):
    """
    A thin wrapper around GPT2Model that disables the causal (unidirectional) mask,
    allowing full bidirectional attention—and prints the internal bias mask each forward pass.
    """

    def __init__(self, config: GPT2Config):
        # Mark as not‐a‐decoder (for downstream utilities).
        config.is_decoder = False
        super().__init__(config)

        # Overwrite each attention's bias so no triangular masking occurs.
        for block in self.h:
            # block.attn.bias is a bool‐tensor of shape (1, 1, max_pos, max_pos).
            block.attn.bias.data.fill_(True)
            block.attn.is_causal = False

        def _no_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values,
            output_attentions: bool,
        ):
            return None

        self._update_causal_mask = _no_causal_mask.__get__(self, GPT2Model)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Determine sequence length for printing the relevant slice of bias
        if input_ids is not None:
            seq_len = input_ids.size(1)
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.size(1)
        else:
            seq_len = None  # If neither is given, we can’t infer seq_len

        if seq_len is not None:
            # Print the (1, 1, seq_len, seq_len) slice of the bias for the first block
            bias_mask = self.h[0].attn.bias[0, 0, :seq_len, :seq_len]
        #     print("Bias mask (block 0) slice [0,0,:seq_len,:seq_len]:")
        #     print(bias_mask)
        # else:
        #     print("Cannot infer sequence length to print bias mask.")

        # If a 2D attention_mask was provided, print its expanded 4D version:
        if attention_mask is not None:
            # Expand to (batch_size, 1, seq_len, seq_len)
            B, S = attention_mask.size()
            expanded = attention_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, S, S)
            # Convert to float mask (1→0.0, 0→-inf) just like GPT2 does internally
            neg_inf = torch.finfo(self.dtype).min
            float_mask = (1.0 - expanded.to(self.dtype)) * neg_inf
            # print(f"Expanded attention_mask (shape {expanded.shape}) → float mask:")
            # print(float_mask)

        # Finally, call the parent forward method
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

```

5./data3/fanpeishan/state/src/state/tx/models/decoders_nb.py文件源代码：

```python
# models/decoders_nb.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial


class NBDecoder(nn.Module):
    """
    scVI‑style decoder that maps a latent embedding (optionally with batch covariates)
    to the parameters of a negative‑binomial (or ZINB) distribution over raw counts.

    Y_ig ~ NB(μ_ig, θ_g)         where
      μ_ig = l_i * softplus(W_g z_i + b_g)
      θ_g  = softplus(r_g)       (gene‑specific inverse dispersion)

    Optionally, a zero‑inflation gate π_ig can be produced (not shown here).
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims=[1024, 256, 256],
        dropout: float = 0.0,
        use_zero_inflation: bool = False,
    ):
        super().__init__()
        modules = []
        in_features = latent_dim
        for h in hidden_dims:
            modules += [
                nn.Linear(in_features, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_features = h
        self.encoder = nn.Sequential(*modules)

        self.skip = nn.Identity() if in_features == latent_dim else nn.Linear(latent_dim, in_features, bias=False)
        self.post_norm = nn.LayerNorm(in_features)

        # Mean parameter
        self.px_scale = nn.Linear(in_features, gene_dim)

        self.l_encoder = nn.Linear(in_features, 1)

        # Gene‑specific inverse dispersion (log‑space, broadcasted)
        self.log_theta = nn.Parameter(torch.randn(gene_dim))

        # Optional zero‑inflation gate
        self.use_zero_inflation = use_zero_inflation
        if use_zero_inflation:
            self.px_dropout = nn.Linear(in_features, gene_dim)

    @property
    def theta(self):
        # softplus to keep positive
        return F.softplus(self.log_theta)

    def forward(self, z: torch.Tensor, log_library: torch.Tensor | None = None):
        """
        z:            [B, latent_dim]
        log_library:  [B, 1]           (optional – if None we predict it)
        returns μ, θ (and π if requested)
        """
        flat = False
        if z.dim() == 3:  # [B,S,D]  → flatten
            B, S, D = z.shape
            z = z.reshape(-1, D)
            flat = True

        h = self.encoder(z)  # [B* S, H]
        h = self.post_norm(h + self.skip(z))

        if log_library is None:
            log_library = self.l_encoder(h)  # [B* S, 1]
        px_scale = F.softplus(self.px_scale(h))  # [B* S, G]
        mu = torch.exp(log_library) * px_scale  # NB mean

        if self.use_zero_inflation:
            pi = torch.sigmoid(self.px_dropout(h))
            outs = (mu, self.theta, pi)
        else:
            outs = (mu, self.theta)

        if flat:  # reshape back to [B,S,*]
            mu = mu.reshape(B, S, -1)
            if self.use_zero_inflation:
                pi = pi.reshape(B, S, -1)
                return mu, self.theta, pi  # θ remains [G]
            else:
                return mu, self.theta
        return outs

    def gene_dim(self) -> int:
        return self.px_scale.out_features


def nb_nll(x, mu, theta, eps: float = 1e-6):
    """
    Negative‑binomial negative log‑likelihood.
        x, mu : [..., G]
        theta : [G] or [..., G]
    returns scalar
    """
    logits = (mu + eps).log() - (theta + eps).log()  # NB parameterisation
    dist = NegativeBinomial(total_count=theta, logits=logits)
    return -dist.log_prob(x).mean()

```

