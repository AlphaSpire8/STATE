import argparse as ap


def add_arguments_combo(parser: ap.ArgumentParser) -> None:
    """CLI for two-stage perturbation combination sweeps."""

    parser.add_argument("--model-dir", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path. If omitted, defaults to <model-dir>/checkpoints/last.ckpt "
            "(falling back to final.ckpt if needed)."
        ),
    )
    parser.add_argument("--adata", type=str, required=True, help="Path to input AnnData file (.h5ad).")
    parser.add_argument(
        "--embed-key",
        type=str,
        default=None,
        help="Optional key in adata.obsm for input features (defaults to adata.X).",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        required=True,
        help="Column in adata.obs containing perturbation labels.",
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        required=True,
        help="Label of the control perturbation (used to construct the base control set).",
    )
    parser.add_argument(
        "--cell-type",
        type=str,
        required=True,
        help="Target cell type value to filter before running the combo sweep.",
    )
    parser.add_argument(
        "--celltype-col",
        type=str,
        default=None,
        help=(
            "Optional column name in adata.obs for cell types. If omitted, attempts to detect using the "
            "training config or common fallbacks."
        ),
    )
    parser.add_argument(
        "--cell-set-len",
        type=int,
        default=None,
        help="Override the model cell_set_len when constructing the fixed control set.",
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        default=None,
        help=(
            "Optional batch column in adata.obs. If omitted, attempts to detect from training config "
            "or common fallbacks when the model uses a batch encoder."
        ),
    )
    parser.add_argument(
        "--inner-batch-size",
        type=int,
        default=1,
        help="Number of target perturbations to evaluate simultaneously in the second pass.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for control sampling.")
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help=(
            "Directory where per-perturbation AnnData outputs (.h5ad) are written."
            " Defaults to <adata>_combo/ alongside the input file."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")


def run_tx_combo(args: ap.Namespace) -> None:
    import logging
    import os
    import pickle
    import re

    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch
    import yaml

    from tqdm import tqdm

    from ...tx.models.state_transition import StateTransitionPerturbationModel

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    if args.quiet:
        logger.setLevel(logging.WARNING)

    def _load_config(cfg_path: str) -> dict:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _pick_first_present(columns: pd.Index, candidates: list[str | None]) -> str | None:
        for key in candidates:
            if key and key in columns:
                return key
        return None

    def _to_dense(matrix) -> np.ndarray:
        try:
            import scipy.sparse as sp  # type: ignore

            if sp.issparse(matrix):
                return matrix.toarray()
        except Exception:
            pass
        return np.asarray(matrix)

    def _normalize_pert_vector(raw_vec, expected_dim: int) -> torch.Tensor:
        if raw_vec is None:
            return torch.zeros(expected_dim, dtype=torch.float32)
        if torch.is_tensor(raw_vec):
            return raw_vec.detach().float()
        vec_np = np.asarray(raw_vec)
        return torch.tensor(vec_np, dtype=torch.float32)

    def _flatten_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
        return tensor

    def _tensor_to_numpy(tensor: torch.Tensor | None) -> np.ndarray | None:
        if tensor is None:
            return None
        flat = _flatten_tensor(tensor)
        if flat is None:
            return None
        return flat.detach().cpu().numpy().astype(np.float32)

    def _argmax_index_from_any(value, expected_dim: int | None = None) -> int | None:
        if value is None:
            return None
        try:
            if torch.is_tensor(value):
                if value.ndim == 0:
                    return int(value.item())
                if value.ndim == 1:
                    return int(torch.argmax(value).item())
                return None
        except Exception:
            return None
        try:
            arr = np.asarray(value)
            if arr.ndim == 0:
                return int(arr.item())
            if arr.ndim == 1:
                return int(arr.argmax())
        except Exception:
            pass
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
                if arr.ndim == 1:
                    return int(arr.argmax())
            except Exception:
                return None
        return None

    model_dir = os.path.abspath(args.model_dir)
    config_path = os.path.join(model_dir, "config.yaml")
    cfg = _load_config(config_path)

    var_dims_path = os.path.join(model_dir, "var_dims.pkl")
    if not os.path.exists(var_dims_path):
        raise FileNotFoundError(f"Missing var_dims.pkl at {var_dims_path}")
    with open(var_dims_path, "rb") as handle:
        var_dims = pickle.load(handle)

    input_dim = int(var_dims.get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError("input_dim missing from var_dims.pkl; cannot determine feature dimension")

    pert_dim = int(var_dims.get("pert_dim", 0))
    if pert_dim <= 0:
        raise ValueError("pert_dim missing from var_dims.pkl; cannot build perturbation embeddings")

    batch_dim_entry = var_dims.get("batch_dim")
    batch_dim = int(batch_dim_entry) if batch_dim_entry is not None else None

    pert_map_path = os.path.join(model_dir, "pert_onehot_map.pt")
    if not os.path.exists(pert_map_path):
        raise FileNotFoundError(f"Missing pert_onehot_map.pt at {pert_map_path}")
    pert_onehot_map = torch.load(pert_map_path, weights_only=False)

    batch_onehot_map_path = os.path.join(model_dir, "batch_onehot_map.pkl")
    batch_onehot_map = None
    if os.path.exists(batch_onehot_map_path):
        with open(batch_onehot_map_path, "rb") as handle:
            batch_onehot_map = pickle.load(handle)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_last = os.path.join(model_dir, "checkpoints", "last.ckpt")
        default_final = os.path.join(model_dir, "checkpoints", "final.ckpt")
        checkpoint_path = default_last if os.path.exists(default_last) else default_final
    elif not os.path.isabs(checkpoint_path):
        candidate = os.path.join(model_dir, checkpoint_path)
        checkpoint_path = candidate if os.path.exists(candidate) else checkpoint_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = next(model.parameters()).device
    cell_set_len = args.cell_set_len or getattr(model, "cell_sentence_len", 256)

    uses_batch_encoder = getattr(model, "batch_encoder", None) is not None
    if uses_batch_encoder and (batch_dim is None or batch_dim <= 0):
        raise ValueError("Model uses a batch encoder but batch_dim missing from var_dims.pkl")
    if uses_batch_encoder and batch_onehot_map is None:
        raise FileNotFoundError(
            "Model uses a batch encoder but batch_onehot_map.pkl was not found in the model directory"
        )

    logger.info("Loaded model from %s (cell_set_len=%d)", checkpoint_path, cell_set_len)

    adata = sc.read_h5ad(args.adata)

    data_kwargs = {}
    try:
        data_kwargs = cfg.get("data", {}).get("kwargs", {})  # type: ignore[assignment]
    except AttributeError:
        data_kwargs = {}

    celltype_col = args.celltype_col
    if celltype_col is None:
        cfg_celltype = None
        try:
            cfg_celltype = data_kwargs.get("cell_type_key")
        except Exception:
            cfg_celltype = None
        candidates = [
            cfg_celltype,
            "cell_type",
            "celltype",
            "cell_type_name",
            "celltype_name",
            "cellType",
            "ctype",
        ]
        celltype_col = _pick_first_present(adata.obs.columns, candidates)
    if celltype_col is None:
        raise ValueError("Could not determine cell type column; provide --celltype-col explicitly.")
    if celltype_col not in adata.obs:
        raise KeyError(f"Column '{celltype_col}' not found in adata.obs")

    if args.pert_col not in adata.obs:
        raise KeyError(f"Perturbation column '{args.pert_col}' not found in adata.obs")

    adata_ct = adata[adata.obs[celltype_col].astype(str) == str(args.cell_type)].copy()
    if adata_ct.n_obs == 0:
        raise ValueError(f"No cells found with cell type '{args.cell_type}' in column '{celltype_col}'")

    pert_series = adata_ct.obs[args.pert_col].astype(str)
    control_mask = pert_series == str(args.control_pert)
    control_indices = np.where(control_mask)[0]
    if len(control_indices) == 0:
        raise ValueError(
            f"No control cells with perturbation '{args.control_pert}' found in column '{args.pert_col}' "
            f"for cell type '{args.cell_type}'"
        )

    perts_all = pd.unique(pert_series)
    perts = [p for p in perts_all if p != str(args.control_pert)]
    if len(perts) == 0:
        raise ValueError("No non-control perturbations found in filtered AnnData")

    batch_indices_all: np.ndarray | None = None
    batch_col = args.batch_col if args.batch_col is not None else data_kwargs.get("batch_col")
    if uses_batch_encoder:
        candidate_batch_cols: list[str] = []
        if batch_col is not None:
            candidate_batch_cols.append(batch_col)
        if isinstance(data_kwargs.get("batch_col"), str):
            candidate_batch_cols.append(data_kwargs.get("batch_col"))
        candidate_batch_cols.extend(
            [
                "gem_group",
                "gemgroup",
                "batch",
                "donor",
                "plate",
                "experiment",
                "lane",
                "batch_id",
            ]
        )
        resolved_batch_col = next((col for col in candidate_batch_cols if col in adata_ct.obs), None)
        if resolved_batch_col is None:
            raise ValueError(
                "Model uses a batch encoder but no batch column was found. Provide --batch-col explicitly."
            )
        batch_col = resolved_batch_col
        raw_batch_labels = adata_ct.obs[batch_col].astype(str).values

        assert batch_onehot_map is not None
        label_to_idx: dict[str, int] = {}
        if isinstance(batch_onehot_map, dict):
            for key, value in batch_onehot_map.items():
                idx = _argmax_index_from_any(value, batch_dim)
                if idx is not None:
                    label_to_idx[str(key)] = idx

        if not label_to_idx and batch_dim is not None:
            unique_labels = sorted(set(raw_batch_labels))
            label_to_idx = {lab: min(i, batch_dim - 1) for i, lab in enumerate(unique_labels)}

        if not label_to_idx:
            raise ValueError("Unable to construct batch label mapping; batch_onehot_map is empty or invalid")

        fallback_idx = sorted(label_to_idx.values())[0]
        batch_indices_all = np.zeros(len(raw_batch_labels), dtype=np.int64)
        misses = 0
        for i, lab in enumerate(raw_batch_labels):
            idx = label_to_idx.get(lab)
            if idx is None:
                batch_indices_all[i] = fallback_idx
                misses += 1
            else:
                batch_indices_all[i] = idx

        if misses:
            logger.warning(
                "Batch column '%s': %d/%d labels missing from saved mapping; using fallback index %d",
                batch_col,
                misses,
                len(raw_batch_labels),
                fallback_idx,
            )
        logger.info(
            "Using batch column '%s' with %d unique mapped labels",
            batch_col,
            len(np.unique(batch_indices_all)),
        )

    cfg_embed_key = data_kwargs.get("embed_key")
    explicit_embed_key = args.embed_key is not None

    candidate_order: list[str | None] = []
    seen_keys: set[str | None] = set()

    def _append_candidate(key: str | None) -> None:
        if key in seen_keys:
            return
        seen_keys.add(key)
        candidate_order.append(key)

    if explicit_embed_key:
        _append_candidate(args.embed_key)
    else:
        if isinstance(cfg_embed_key, str):
            _append_candidate(cfg_embed_key)
        _append_candidate(None)
        for fallback_key in ("X_hvg", "X_state", "X_state_basal", "X_state_pred", "X_pca", "X_latent"):
            if fallback_key in adata_ct.obsm:
                _append_candidate(fallback_key)

    selection_errors: list[str] = []
    features = None
    used_embed_key: str | None = None

    for candidate in candidate_order:
        matrix = None
        label = "adata.X" if candidate is None else f"adata.obsm['{candidate}']"

        if candidate is None:
            matrix = _to_dense(adata_ct.X)
        else:
            if candidate not in adata_ct.obsm:
                if explicit_embed_key:
                    raise KeyError(f"Embedding key '{candidate}' not found in adata.obsm")
                selection_errors.append(f"{label} missing")
                continue
            matrix = np.asarray(adata_ct.obsm[candidate])

        if matrix.shape[0] != adata_ct.n_obs:
            msg = f"{label} row count {matrix.shape[0]} != filtered AnnData cells {adata_ct.n_obs}"
            if explicit_embed_key:
                raise ValueError(msg)
            selection_errors.append(msg)
            continue

        if matrix.shape[1] != input_dim:
            msg = f"{label} feature dimension {matrix.shape[1]} != model input_dim {input_dim}"
            if explicit_embed_key:
                raise ValueError(
                    msg
                    + ". Provide --embed-key pointing to a representation with matching dimension or preprocess the input."
                )
            selection_errors.append(msg)
            continue

        features = matrix
        used_embed_key = candidate
        break

    if features is None:
        tried = ", ".join(["adata.X" if c is None else f"adata.obsm['{c}']" for c in candidate_order]) or "(none)"
        detail = "; ".join(selection_errors) if selection_errors else "No suitable feature representation found."
        raise ValueError(
            f"Unable to find a feature matrix matching the model input dimension. Tried: {tried}. {detail}"
        )

    if used_embed_key is None:
        logger.info("Using adata.X (%d cells x %d features) as input features", features.shape[0], features.shape[1])
    else:
        logger.info(
            "Using adata.obsm['%s'] (%d cells x %d features) as input features",
            used_embed_key,
            features.shape[0],
            features.shape[1],
        )

    features = features.astype(np.float32, copy=False)

    rng = np.random.default_rng(args.seed)
    replace = len(control_indices) < cell_set_len
    sampled_idx = rng.choice(control_indices, size=cell_set_len, replace=replace)
    control_features = features[sampled_idx]

    default_vec = _normalize_pert_vector(pert_onehot_map.get(str(args.control_pert)), pert_dim)
    if default_vec.numel() != pert_dim:
        default_vec = torch.zeros(pert_dim, dtype=torch.float32)

    control_batch_tensor = None
    if batch_indices_all is not None:
        control_batch_tensor = torch.tensor(batch_indices_all[sampled_idx], dtype=torch.long, device=device)

    def _pert_batch_tensor(name: str) -> torch.Tensor:
        raw_vec = pert_onehot_map.get(name)
        vec = _normalize_pert_vector(raw_vec, pert_dim) if raw_vec is not None else default_vec
        if vec.dim() == 0:
            vec = vec.unsqueeze(0)
        vec = vec.reshape(-1)
        if vec.numel() != pert_dim:
            raise ValueError(f"Perturbation vector for '{name}' has incorrect dimension {vec.numel()} != {pert_dim}")
        return vec.unsqueeze(0).repeat(cell_set_len, 1).to(device)

    pert_batch_vectors = {name: _pert_batch_tensor(name) for name in perts}

    control_tensor = torch.tensor(control_features, dtype=torch.float32, device=device)

    use_counts: bool | None = None
    inner_batch_size = max(1, int(args.inner_batch_size))

    def _default_output_dir(path: str) -> str:
        base_dir = os.path.dirname(os.path.abspath(path))
        base_name = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(base_dir, f"{base_name}_combo")

    output_dir = args.output_folder or _default_output_dir(args.adata)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Writing per-perturbation combo outputs to %s", output_dir)

    def _sanitize_filename(label: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z_.-]+", "_", label)
        sanitized = sanitized.strip("._")
        return sanitized or "perturbation"

    used_output_names: dict[str, int] = {}
    written_files: list[str] = []

    with torch.no_grad():
        progress_total = len(perts) * len(perts)
        progress_bar = tqdm(
            total=progress_total,
            desc="Combo sweeps",
            unit="combo",
            disable=args.quiet,
        )
        for pert1 in perts:
            per_pert_X_blocks: list[np.ndarray] = []
            per_pert_latent_blocks: list[np.ndarray] = []
            per_pert_obs_rows: list[dict[str, str | int]] = []

            first_batch = {
                "ctrl_cell_emb": control_tensor.clone(),
                "pert_emb": pert_batch_vectors[pert1],
                "pert_name": [pert1] * cell_set_len,
            }
            if control_batch_tensor is not None:
                first_batch["batch"] = control_batch_tensor.clone()
            first_out = model.predict_step(first_batch, batch_idx=0, padded=False)
            first_latent_tensor = _flatten_tensor(first_out.get("preds"))
            if first_latent_tensor is None:
                raise RuntimeError("Model predict_step did not return 'preds' tensor")
            first_latent_tensor = first_latent_tensor.detach().to(device)

            for chunk_start in range(0, len(perts), inner_batch_size):
                chunk_perts = perts[chunk_start : chunk_start + inner_batch_size]
                chunk_size = len(chunk_perts)

                ctrl_chunk = torch.cat([first_latent_tensor.clone() for _ in chunk_perts], dim=0)
                pert_chunk = torch.cat([pert_batch_vectors[p] for p in chunk_perts], dim=0)
                names_chunk = [p for p in chunk_perts for _ in range(cell_set_len)]

                second_batch = {
                    "ctrl_cell_emb": ctrl_chunk,
                    "pert_emb": pert_chunk,
                    "pert_name": names_chunk,
                }

                if control_batch_tensor is not None:
                    batch_chunk = control_batch_tensor.repeat(chunk_size)
                    second_batch["batch"] = batch_chunk

                second_out = model.predict_step(second_batch, batch_idx=0, padded=True)

                latent_np = _tensor_to_numpy(second_out.get("preds"))
                counts_np = _tensor_to_numpy(second_out.get("pert_cell_counts_preds"))

                if latent_np is None:
                    raise RuntimeError("Second-stage prediction missing 'preds' output")

                latent_np = latent_np.reshape(chunk_size, cell_set_len, -1)
                counts_np = counts_np.reshape(chunk_size, cell_set_len, -1) if counts_np is not None else None

                if use_counts is None:
                    use_counts = counts_np is not None
                elif use_counts and counts_np is None:
                    raise RuntimeError("Inconsistent decoder outputs across perturbations; expected counts predictions")

                for idx_chunk, pert2 in enumerate(chunk_perts):
                    latent_slice = latent_np[idx_chunk].astype(np.float32)
                    if use_counts:
                        assert counts_np is not None
                        per_pert_X_blocks.append(counts_np[idx_chunk].astype(np.float32))
                    else:
                        per_pert_X_blocks.append(latent_slice)
                    per_pert_latent_blocks.append(latent_slice)

                    for cell_idx in range(cell_set_len):
                        per_pert_obs_rows.append({"pert1": pert1, "pert2": pert2, "cell_index": cell_idx})

                    progress_bar.update(1)

            X_matrix = (
                np.vstack(per_pert_X_blocks)
                if per_pert_X_blocks
                else np.empty((0, 0), dtype=np.float32)
            )
            latent_matrix = (
                np.vstack(per_pert_latent_blocks)
                if per_pert_latent_blocks
                else np.empty((0, 0), dtype=np.float32)
            )
            obs_df = pd.DataFrame(per_pert_obs_rows)

            feature_dim = 0
            if use_counts and X_matrix.size > 0:
                feature_dim = X_matrix.shape[1]
            elif latent_matrix.size > 0:
                feature_dim = latent_matrix.shape[1]
            elif X_matrix.size > 0:
                feature_dim = X_matrix.shape[1]

            gene_names = var_dims.get("gene_names")
            if (
                use_counts
                and feature_dim > 0
                and isinstance(gene_names, (list, tuple))
                and len(gene_names) == feature_dim
            ):
                var_index = pd.Index([str(name) for name in gene_names], name="gene")
            else:
                var_index = pd.Index([f"feature_{i}" for i in range(feature_dim)], name="feature")
            var_df = pd.DataFrame(index=var_index)

            combo_adata = ad.AnnData(X=X_matrix, obs=obs_df, var=var_df)
            combo_adata.obsm["latent_preds"] = latent_matrix
            combo_adata.uns["cell_type"] = str(args.cell_type)
            combo_adata.uns["perturbations"] = perts
            combo_adata.uns["pert1"] = pert1
            combo_adata.uns["control_pert"] = str(args.control_pert)
            combo_adata.uns["cell_set_len"] = cell_set_len
            combo_adata.uns["input_embed_key"] = used_embed_key if used_embed_key is not None else "X"
            if uses_batch_encoder and batch_col is not None:
                combo_adata.uns["batch_col"] = batch_col
            combo_adata.uns["inner_batch_size"] = inner_batch_size
            combo_adata.uns["sampled_control_indices"] = adata_ct.obs_names[sampled_idx].tolist()

            output_name = _sanitize_filename(pert1)
            if output_name in used_output_names:
                used_output_names[output_name] += 1
                output_name = f"{output_name}_{used_output_names[output_name]}"
            else:
                used_output_names[output_name] = 0

            output_path = os.path.join(output_dir, f"{output_name}.h5ad")
            combo_adata.write_h5ad(output_path)
            written_files.append(output_path)
            logger.info("Saved combos for %s with %d cells to %s", pert1, combo_adata.n_obs, output_path)

        progress_bar.close()

    logger.info("Finished writing %d combo files to %s", len(written_files), output_dir)
