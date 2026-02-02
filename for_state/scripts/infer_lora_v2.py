"""
infer_lora_v2.py

优化版推断脚本 - 修正了药物扰动预测的推断流程

关键改进：
1. 使用控制组均值表达作为所有细胞的输入（而非每个细胞自己的扰动后表达）
2. 正确模拟真实应用场景：预测扰动对控制状态细胞的影响

原始问题：原脚本使用扰动后细胞的表达作为输入，这导致 pearson_delta 指标下降，
因为模型输入已经包含了扰动信息，使得 delta 预测变得无意义。

用法示例：
python infer_lora_v2.py \
  --adata /path/to/data.h5ad \
  --model_dir /path/to/model_dir \
  --lora_path /path/to/lora_state.pth \
  --output /path/to/output.h5ad
"""

import argparse
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn


# ------------- LoRA module & helper functions (from finetune.py) -------------
class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear layer - used during inference to apply fine-tuned LoRA weights."""
    def __init__(self, orig: nn.Linear, r: int = 4, alpha: float = 1.0, bias: bool = True):
        super().__init__()
        self.orig = orig  # original linear layer
        in_f = orig.in_features
        out_f = orig.out_features
        self.r = r
        self.alpha = alpha
        # LoRA parameters
        if r > 0:
            # A: out x r ; B: r x in
            self.A = nn.Parameter(torch.zeros(out_f, r))
            self.B = nn.Parameter(torch.zeros(r, in_f))
            # scaling
            self.scaling = self.alpha / max(1, self.r)
        else:
            # no LoRA
            self.register_parameter("A", None)
            self.register_parameter("B", None)
            self.scaling = 0.0

    def forward(self, x):
        # original linear output
        out = self.orig(x)
        if self.r > 0:
            # compute low-rank update: (x @ B.T) @ A.T
            lora_update = (x @ self.B.t()) @ self.A.t()  # (batch, out)
            out = out + self.scaling * lora_update
        return out


def replace_linear_with_lora(model: nn.Module, r: int, alpha: float,
                             target_module_keywords=None) -> Tuple[nn.Module, Dict[str, nn.Module]]:
    """Replace nn.Linear layers with LoRALinear wrappers."""
    if target_module_keywords is not None:
        target_module_keywords = [kw.lower() for kw in target_module_keywords]

    lora_modules = {}

    # Build mapping name -> module
    name2mod = dict(model.named_modules())

    # For each module, check its immediate children to replace
    for full_name, module in list(name2mod.items()):
        for child_name, child in list(module.named_children()):
            # child is immediate child module
            if isinstance(child, nn.Linear):
                do_replace = False
                if target_module_keywords is None:
                    do_replace = True
                else:
                    # check full_name + child_name for keywords
                    candidate_name = (full_name + "." + child_name).lower() if full_name != "" else child_name.lower()
                    for kw in target_module_keywords:
                        if kw in candidate_name:
                            do_replace = True
                            break
                if do_replace:
                    lora = LoRALinear(child, r=r, alpha=alpha)
                    # replace on parent
                    setattr(module, child_name, lora)
                    full_child_name = (full_name + "." + child_name) if full_name != "" else child_name
                    lora_modules[full_child_name] = lora
    return model, lora_modules


def add_arguments_infer(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to model checkpoint (.ckpt). If not provided, will use model_dir/checkpoints/final.ckpt",
    )
    parser.add_argument("--adata", type=str, required=True, help="Path to input AnnData file (.h5ad)")
    parser.add_argument("--embed_key", type=str, default=None, help="Key in adata.obsm for input features")
    parser.add_argument(
        "--pert_col", type=str, default="drugname_drugconc", help="Column in adata.obs for perturbation labels"
    )
    parser.add_argument("--output", type=str, default=None, help="Path to output AnnData file (.h5ad)")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model_dir containing the config.yaml file and the pert_onehot_map.pt file that was saved during training.",
    )
    parser.add_argument(
        "--celltype_col", type=str, default=None, help="Column in adata.obs for cell type labels (optional)"
    )
    parser.add_argument(
        "--celltypes", type=str, default=None, help="Comma-separated list of cell types to include (optional)"
    )
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for inference (default: 1000)")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights file or directory (if dir, looks for lora_state.pth inside).",
    )
    # ===== V2 新增参数 =====
    parser.add_argument(
        "--use_ctrl_mean",
        action="store_true",
        default=True,
        help="[V2] Use control mean as input for all cells (default: True). "
             "This is the correct inference mode for perturbation prediction.",
    )
    parser.add_argument(
        "--no_ctrl_mean",
        action="store_true",
        help="[V2] Disable control mean mode and use original per-cell input (for comparison).",
    )
    parser.add_argument(
        "--control_label",
        type=str,
        default=None,
        help="[V2] Override control label for identifying control cells. "
             "If not provided, will use config default or '[('DMSO_TF', 0.0, 'uM')]'.",
    )


def run_tx_infer(args):
    import logging
    import os
    import pickle
    import sys

    import scanpy as sc
    import yaml
    from tqdm import tqdm

    # add your state src to path (adjust if needed)
    sys.path.append("/data0/home/qijinyin/Workspace/02_perturb/state/src")
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ===== V2: 确定是否使用控制组均值 =====
    use_ctrl_mean = args.use_ctrl_mean and not args.no_ctrl_mean
    logger.info(f"[V2] Control mean mode: {'ENABLED' if use_ctrl_mean else 'DISABLED'}")

    # ---------- Load config ----------
    def load_config(cfg_path: str) -> dict:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    config_path = os.path.join(args.model_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # ---------- Checkpoint path ----------
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(args.model_dir, "checkpoints", "final.ckpt")
        logger.info(f"No checkpoint provided, using default: {checkpoint_path}")

    # ---------- Load var_dims (pert dim) ----------
    var_dims_path = os.path.join(args.model_dir, "var_dims.pkl")
    with open(var_dims_path, "rb") as f:
        var_dims = pickle.load(f)
    pert_dim = var_dims["pert_dim"]
    logger.info(f"Perturbation dim: {pert_dim}")

    # ---------- Load model ----------
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    # keep original logic for cell_sentence_len and device
    cell_sentence_len = model.cell_sentence_len
    device = next(model.parameters()).device
    logger.info(f"Model cell_sentence_len={cell_sentence_len}, parameters device={device}")

    # ---------- Load LoRA (FIXED: properly rebuild LoRA layers and load weights) ----------
    if args.lora_path is not None:
        logger.info(f"Loading LoRA adapter from: {args.lora_path}")
        if os.path.isdir(args.lora_path):
            lora_file = os.path.join(args.lora_path, "lora_state.pth")
        else:
            lora_file = args.lora_path
        if not os.path.exists(lora_file):
            raise FileNotFoundError(f"LoRA weights not found at {lora_file}")

        logger.info(f"Found LoRA file: {lora_file} -> loading...")
        lora_data = torch.load(lora_file, map_location="cpu")

        # Extract LoRA state dict first (needed for both loading and config inference)
        lora_state = lora_data.get("lora_state", lora_data)

        # Extract LoRA config (rank and alpha) from saved file
        # Try to get from cfg first
        if "cfg" in lora_data:
            lora_rank = lora_data["cfg"].get("rank", 8)
            lora_alpha = lora_data["cfg"].get("alpha", 1.0)
        else:
            # Fallback: try to infer from first module's params
            # This handles the case where finetune_v2.py saves r/alpha per module
            lora_rank = 8
            lora_alpha = 1.0
            for module_name, params in lora_state.items():
                if isinstance(params, dict):
                    if "r" in params:
                        lora_rank = params["r"]
                    if "alpha" in params:
                        lora_alpha = params["alpha"]
                    break  # Only need to get from first module
        logger.info(f"LoRA config: rank={lora_rank}, alpha={lora_alpha}")

        # Apply LoRA wrappers to the model (same as finetune.py does)
        logger.info(f"Applying LoRA wrappers to model (rank={lora_rank}, alpha={lora_alpha})...")
        model, lora_modules = replace_linear_with_lora(model, r=lora_rank, alpha=lora_alpha)
        logger.info(f"Created {len(lora_modules)} LoRA modules")

        # Load LoRA weights into the corresponding modules
        loaded_count = 0
        for module_name, params in lora_state.items():
            if module_name in lora_modules:
                m = lora_modules[module_name]
                if isinstance(params, dict):
                    # finetune.py format: {"module.name": {"A": tensor, "B": tensor, "r": int, "alpha": float}}
                    if "A" in params and hasattr(m, "A") and m.A is not None:
                        m.A.data.copy_(params["A"].to(m.A.device))
                    if "B" in params and hasattr(m, "B") and m.B is not None:
                        m.B.data.copy_(params["B"].to(m.B.device))
                    loaded_count += 1
                    logger.debug(f"Loaded LoRA weights for module: {module_name}")
            else:
                logger.warning(f"LoRA module '{module_name}' not found in model, skipping...")

        logger.info(f"Successfully loaded LoRA weights for {loaded_count}/{len(lora_state)} modules")

        # Move model to device after LoRA application
        model = model.to(device)
        model.eval()

    # ---------- Load AnnData ----------
    logger.info(f"Loading AnnData from: {args.adata}")
    adata = sc.read_h5ad(args.adata)

    # Optionally filter by cell type
    if args.celltype_col is not None and args.celltypes is not None:
        celltypes = [ct.strip() for ct in args.celltypes.split(",")]
        if args.celltype_col not in adata.obs:
            raise ValueError(f"Column '{args.celltype_col}' not found in adata.obs.")
        initial_n = adata.n_obs
        adata = adata[adata.obs[args.celltype_col].isin(celltypes)].copy()
        logger.info(f"Filtered AnnData to {adata.n_obs} cells of types {celltypes} (from {initial_n} cells)")
    elif args.celltype_col is not None:
        if args.celltype_col not in adata.obs:
            raise ValueError(f"Column '{args.celltype_col}' not found in adata.obs.")
        logger.info(f"No cell type filtering applied, but cell type column '{args.celltype_col}' is available.")

    # ---------- Get input features ----------
    if args.embed_key in adata.obsm:
        X = adata.obsm[args.embed_key]
        logger.info(f"Using adata.obsm['{args.embed_key}'] as input features: shape {X.shape}")
    else:
        try:
            X = adata.X.toarray()
        except:
            X = adata.X
        logger.info(f"Using adata.X as input features: shape {X.shape}")

    # ===== V2 关键修改：计算控制组均值表达 =====
    # 确定控制组标签
    # 对于药物扰动数据集 (drugname_drugconc)，默认控制标签是 "[('DMSO_TF', 0.0, 'uM')]"
    if args.control_label is not None:
        # 用户显式指定了控制标签
        control_label = args.control_label
    elif args.pert_col == "drugname_drugconc":
        # 药物扰动数据集的标准控制标签格式
        control_label = "[('DMSO_TF', 0.0, 'uM')]"
    else:
        # 尝试从配置文件读取
        control_pert_cfg = cfg["data"]["kwargs"].get("control_pert", None)
        if control_pert_cfg:
            control_label = control_pert_cfg
        else:
            control_label = "[('DMSO_TF', 0.0, 'uM')]"
    
    logger.info(f"[V2] Control label: '{control_label}'")
    
    # 计算控制组均值（用于推断输入）
    pert_names = adata.obs[args.pert_col].values
    n_samples = len(pert_names)
    
    if use_ctrl_mean:
        ctrl_mask = pert_names == control_label
        n_ctrl = ctrl_mask.sum()
        
        if n_ctrl == 0:
            logger.warning(f"[V2] No control cells found with label '{control_label}'. "
                          f"Falling back to using first 10% of cells as pseudo-control.")
            # fallback: 使用前10%细胞作为伪控制组
            n_pseudo_ctrl = max(1, int(n_samples * 0.1))
            ctrl_X = X[:n_pseudo_ctrl]
            ctrl_mean = ctrl_X.mean(axis=0)
            logger.info(f"[V2] Using pseudo-control mean from first {n_pseudo_ctrl} cells")
        else:
            ctrl_X = X[ctrl_mask]
            ctrl_mean = ctrl_X.mean(axis=0)  # (gene_dim,)
            logger.info(f"[V2] Computed control mean from {n_ctrl} control cells")
        
        # 将控制组均值广播到所有细胞作为输入
        # 这是关键改进：所有细胞的推断输入都是相同的控制组均值
        X_input = np.tile(ctrl_mean, (n_samples, 1))  # (n_samples, gene_dim)
        logger.info(f"[V2] Using control mean as input for ALL {n_samples} samples")
        logger.info(f"[V2] Input shape: {X_input.shape}, Control mean norm: {np.linalg.norm(ctrl_mean):.4f}")
    else:
        # 原始模式：每个细胞使用自己的表达作为输入
        X_input = X
        logger.info(f"[V2] Using per-cell expression as input (original mode)")

    # ---------- Prepare perturbation tensor ----------
    pert_tensor = torch.zeros((n_samples, pert_dim), device="cpu")  # Keep on CPU initially
    logger.info(f"Perturbation tensor shape: {pert_tensor.shape}")

    # Load perturbation mapping
    pert_onehot_map_path = os.path.join(args.model_dir, "pert_onehot_map.pt")
    pert_onehot_map = torch.load(pert_onehot_map_path, map_location="cpu", weights_only=False)
    logger.info(f"Data module has {len(pert_onehot_map)} perturbations in mapping")

    unique_pert_names = sorted(set(pert_names))
    logger.info(f"AnnData has {len(unique_pert_names)} unique perturbations")

    # control perturb (用于未知扰动的fallback)
    control_pert = cfg["data"]["kwargs"].get("control_pert")
    if args.pert_col == "drugname_drugconc" and control_pert is None:
        control_pert = "[('DMSO_TF', 0.0, 'uM')]"
    logger.info(f"Control perturbation in data module: '{control_pert}'")

    matched_count = 0
    for idx, name in enumerate(pert_names):
        if name in pert_onehot_map:
            pert_tensor[idx] = pert_onehot_map[name]
            matched_count += 1
        else:
            if control_pert in pert_onehot_map:
                pert_tensor[idx] = pert_onehot_map[control_pert]
            else:
                first_pert = list(pert_onehot_map.keys())[0]
                pert_tensor[idx] = pert_onehot_map[first_pert]
    logger.info(f"Matched {matched_count} out of {n_samples} perturbations")

    # ---------- Inference batching ----------
    batch_size = cell_sentence_len  # follow original logic: the model expects batches of this length
    n_batches = (n_samples + batch_size - 1) // batch_size

    logger.info(
        f"Running inference on {n_samples} samples in {n_batches} batches of size {batch_size} (model's cell_sentence_len)..."
    )

    all_preds = []

    with torch.no_grad():
        progress_bar = tqdm(total=n_samples, desc="Processing samples", unit="samples")

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            current_batch_size = end_idx - start_idx

            # ===== V2 关键：使用 X_input（控制组均值）而非 X =====
            X_batch = torch.tensor(X_input[start_idx:end_idx], dtype=torch.float32).to(device)
            pert_batch = pert_tensor[start_idx:end_idx].to(device)
            pert_names_batch = pert_names[start_idx:end_idx].tolist()

            # Pad the batch to cell_sentence_len if it's the last incomplete batch (same behavior as original)
            if current_batch_size < cell_sentence_len:
                padding_size = cell_sentence_len - current_batch_size
                X_pad = torch.zeros((padding_size, X_batch.shape[1]), device=device)
                X_batch = torch.cat([X_batch, X_pad], dim=0)

                pert_pad = torch.zeros((padding_size, pert_batch.shape[1]), device=device)
                if control_pert in pert_onehot_map:
                    # ensure tensor on same device and shapes match
                    pert_pad[:] = pert_onehot_map[control_pert].to(device)
                else:
                    pert_pad[:, 0] = 1
                pert_batch = torch.cat([pert_batch, pert_pad], dim=0)

                pert_names_batch.extend([control_pert] * padding_size)

            # Prepare batch dict exactly like original
            batch = {
                "ctrl_cell_emb": X_batch,
                "pert_emb": pert_batch,
                "pert_name": pert_names_batch,
                "batch": torch.zeros((1, cell_sentence_len), device=device),
            }

            # Run model.predict_step as in original
            batch_preds = model.predict_step(batch, batch_idx=batch_idx, padded=False)

            if "pert_cell_counts_preds" in batch_preds and batch_preds["pert_cell_counts_preds"] is not None:
                pred_tensor = batch_preds["pert_cell_counts_preds"]
            else:
                pred_tensor = batch_preds["preds"]

            actual_preds = pred_tensor[:current_batch_size]
            all_preds.append(actual_preds.cpu().numpy())

            progress_bar.update(current_batch_size)

        progress_bar.close()

    # Concatenate and save
    preds_np = np.concatenate(all_preds, axis=0)
    
    # ===== V2: 保存额外的元信息 =====
    adata.X = preds_np
    adata.uns["infer_v2_info"] = {
        "use_ctrl_mean": use_ctrl_mean,
        "control_label": control_label,
        "n_ctrl_cells": int(ctrl_mask.sum()) if use_ctrl_mean else None,
        "ctrl_mean_norm": float(np.linalg.norm(ctrl_mean)) if use_ctrl_mean else None,
    }
    
    output_path = args.output or args.adata.replace(".h5ad", "_with_preds_v2.h5ad")
    adata.write_h5ad(output_path)
    logger.info(f"Saved predictions to {output_path} (in adata.X)")
    logger.info(f"[V2] Inference completed with control mean mode: {'ENABLED' if use_ctrl_mean else 'DISABLED'}")


def main():
    parser = argparse.ArgumentParser(
        description="[V2] Run inference on AnnData with a trained model checkpoint (with optional LoRA merge). "
                    "This version uses control mean as input for correct perturbation prediction."
    )
    add_arguments_infer(parser)
    args = parser.parse_args()
    run_tx_infer(args)


if __name__ == "__main__":
    main()
