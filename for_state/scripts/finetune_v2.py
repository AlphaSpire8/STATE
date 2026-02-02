#!/usr/bin/env python3
"""
优化版微调脚本 - 针对药物扰动预测的 pearson_delta 指标进行优化

关键改进：
1. 智能配对：支持按批次(batch)匹配控制组和扰动组细胞，减少批次效应的影响
2. Delta-Aware 损失函数：专门优化扰动效应(delta)的预测
3. 控制组均值：预计算控制组均值用于 delta 计算

用法示例：
python finetune_v2.py \
  --model_dir /path/to/model_dir \
  --checkpoint /path/to/checkpoint.ckpt \
  --adata /path/to/data.h5ad \
  --pert_col drugname_drugconc \
  --output_lora ./lora_state.pth \
  --epochs 5 \
  --batch_size 128 \
  --lr 5e-4 \
  --lora_rank 8 \
  --use_delta_loss \
  --batch_col plate  # 可选：指定批次列进行智能配对
"""

import argparse
import os
import pickle
from typing import Tuple, Dict, Any, Optional

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py
import pandas as pd  

# ------------- LoRA module & helper functions -------------
class LoRALinear(nn.Module):
    """低秩适配器(LoRA)模块，用于高效微调线性层"""
    def __init__(self, orig: nn.Linear, r: int = 4, alpha: float = 1.0, bias: bool = True):
        super().__init__()
        self.orig = orig  # 原始线性层
        in_f = orig.in_features
        out_f = orig.out_features
        self.r = r
        self.alpha = alpha
        # LoRA参数
        if r > 0:
            self.A = nn.Parameter(torch.zeros(out_f, r))
            self.B = nn.Parameter(torch.zeros(r, in_f))
            nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
            nn.init.zeros_(self.B)
            self.scaling = self.alpha / max(1, self.r)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
            self.scaling = 0.0

    def forward(self, x):
        """前向传播：原始输出 + 低秩更新"""
        out = self.orig(x)
        if self.r > 0:
            lora_update = (x @ self.B.t()) @ self.A.t()
            out = out + self.scaling * lora_update
        return out

def replace_linear_with_lora(model: nn.Module, r: int, alpha: float,
                             target_module_keywords=None) -> Tuple[nn.Module, Dict[str, nn.Module]]:
    """将指定线性层替换为LoRA模块"""
    if target_module_keywords is not None:
        target_module_keywords = [kw.lower() for kw in target_module_keywords]

    lora_modules = {}
    name2mod = dict(model.named_modules())

    for full_name, module in list(name2mod.items()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                do_replace = False
                if target_module_keywords is None:
                    do_replace = True
                else:
                    candidate_name = (full_name + "." + child_name).lower() if full_name != "" else child_name.lower()
                    for kw in target_module_keywords:
                        if kw in candidate_name:
                            do_replace = True
                            break
                if do_replace:
                    lora = LoRALinear(child, r=r, alpha=alpha)
                    setattr(module, child_name, lora)
                    full_child_name = (full_name + "." + child_name) if full_name != "" else child_name
                    lora_modules[full_child_name] = lora
    return model, lora_modules

def freeze_model_except_lora(model: nn.Module):
    """冻结所有参数，仅保留LoRA参数可训练"""
    for n, p in model.named_parameters():
        p.requires_grad = ".A" in n or ".B" in n

def collect_lora_parameters(model: nn.Module):
    """收集所有LoRA可训练参数"""
    return [p for n, p in model.named_parameters() if p.requires_grad]


# ===== Delta-Aware 损失函数 =====
class DeltaAwareLoss(nn.Module):
    """专门优化扰动效应 delta 的复合损失函数"""
    def __init__(self, lambda_delta: float = 1.0, lambda_pearson: float = 0.5, 
                 lambda_mse: float = 0.5):
        super().__init__()
        self.lambda_delta = lambda_delta
        self.lambda_pearson = lambda_pearson
        self.lambda_mse = lambda_mse
    
    def pearson_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 1 - Pearson 相关系数作为损失"""
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        target_centered = target - target.mean(dim=-1, keepdim=True)
        
        pred_std = pred_centered.norm(dim=-1, keepdim=True) + 1e-8
        target_std = target_centered.norm(dim=-1, keepdim=True) + 1e-8
        
        correlation = (pred_centered * target_centered).sum(dim=-1) / (
            pred_std.squeeze(-1) * target_std.squeeze(-1)
        )
        return 1 - correlation.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                ctrl_mean: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算组合损失及各组件值"""
        if ctrl_mean.dim() == 1:
            ctrl_mean = ctrl_mean.unsqueeze(0).expand_as(pred)
        
        delta_pred = pred - ctrl_mean
        delta_true = target - ctrl_mean
        
        delta_mse = F.mse_loss(delta_pred, delta_true)
        delta_pearson = self.pearson_loss(delta_pred, delta_true)
        mse = F.mse_loss(pred, target)
        
        total_loss = (
            self.lambda_delta * delta_mse + 
            self.lambda_pearson * delta_pearson + 
            self.lambda_mse * mse
        )
        return total_loss, {
            "delta_mse": delta_mse.item(), 
            "delta_pearson": delta_pearson.item(),
            "mse": mse.item(),
            "total": total_loss.item()
        }


# ===== 改进的 Dataset =====
class AnnDataPerturbationDatasetV2(Dataset):
    """支持智能批次配对和预计算控制组均值的数据集"""
    def __init__(self, adata, pert_onehot_map: Dict[str, torch.Tensor], 
                 pert_col: str = "drugname_drugconc",
                 embed_key: str = None, 
                 control_pert_override: str = None,
                 control_label: str = "[('DMSO_TF', 0.0, 'uM')]",
                 batch_col: Optional[str] = None):
        self.adata = adata
        
        pert_series = adata.obs[pert_col]
        if pd.api.types.is_categorical_dtype(pert_series):
            pert_series = pert_series.astype(str)
        self.pert_names = pert_series.values
        
        self.control_label = control_label
        self.batch_col = batch_col
        
        if embed_key is not None and embed_key in adata.obsm:
            self.X = adata.obsm[embed_key]
        else:
            try:
                self.X = adata.X.toarray()
            except:
                self.X = adata.X
        
        self.map = pert_onehot_map
        self.pert_dim = next(iter(pert_onehot_map.values())).shape[0]
        self.control_pert = control_pert_override
        
        self.ctrl_indices = np.where(self.pert_names == control_label)[0]
        self.pert_indices = np.where(self.pert_names != control_label)[0]
        
        if len(self.ctrl_indices) == 0:
            raise ValueError(f"未找到控制组细胞（control_label='{control_label}'）")
        if len(self.pert_indices) == 0:
            raise ValueError(f"未找到扰动组细胞")
        
        print(f"[Dataset] 控制组: {len(self.ctrl_indices)}, 扰动组: {len(self.pert_indices)}")
        
        ctrl_mean_np = self.X[self.ctrl_indices].mean(axis=0).astype(np.float32)
        self.ctrl_mean = torch.from_numpy(ctrl_mean_np)
        norm_val = torch.norm(self.ctrl_mean).item()
        print(f"[Dataset] 控制组均值 - 形状: {self.ctrl_mean.shape}, 范数: {norm_val:.4f}")
        
        if batch_col is not None and batch_col in adata.obs.columns:
            self.ctrl_groups = {}
            batch_values = adata.obs[batch_col].values
            for idx in self.ctrl_indices:
                batch_val = batch_values[idx]
                if batch_val not in self.ctrl_groups:
                    self.ctrl_groups[batch_val] = []
                self.ctrl_groups[batch_val].append(idx)
            self.pert_batch_values = batch_values[self.pert_indices]
            print(f"[Dataset] 按批次分组控制组 - 共 {len(self.ctrl_groups)} 个批次")
        else:
            self.ctrl_groups = None
            self.pert_batch_values = None
        
        self.n = len(self.pert_indices)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        pert_idx = self.pert_indices[idx]
        
        if self.ctrl_groups is not None and self.pert_batch_values is not None:
            batch_val = self.pert_batch_values[idx]
            if batch_val in self.ctrl_groups and self.ctrl_groups[batch_val]:
                ctrl_idx = np.random.choice(self.ctrl_groups[batch_val])
            else:
                ctrl_idx = np.random.choice(self.ctrl_indices)
        else:
            ctrl_idx = np.random.choice(self.ctrl_indices)
        
        ctrl_emb = torch.tensor(self.X[ctrl_idx], dtype=torch.float32)
        pert_cell_emb = torch.tensor(self.X[pert_idx], dtype=torch.float32)
        
        name = self.pert_names[pert_idx]
        if name in self.map:
            pert = self.map[name].clone().detach()
        else:
            if self.control_pert and self.control_pert in self.map:
                pert = self.map[self.control_pert].clone().detach()
            else:
                pert = next(iter(self.map.values())).clone().detach()
        
        return {
            "ctrl_cell_emb": ctrl_emb,
            "pert_emb": pert,
            "pert_cell_emb": pert_cell_emb,
            "ctrl_mean": self.ctrl_mean,
            "pert_name": name,
        }


# ===== 改进的 collate_fn =====
def collate_fn_v2(batch, cell_sentence_len: int, pert_dim: int, 
                  control_pert_vec: torch.Tensor = None):
    """批处理函数，支持动态填充和控制组均值传递"""
    current_batch_size = len(batch)
    feat_dim = batch[0]["ctrl_cell_emb"].shape[0]
    X_batch = torch.stack([b["ctrl_cell_emb"] for b in batch], dim=0)
    pert_batch = torch.stack([b["pert_emb"] for b in batch], dim=0)
    pert_names = [b["pert_name"] for b in batch]
    
    pert_cell_batch = torch.stack([b["pert_cell_emb"] for b in batch], dim=0) if batch[0]["pert_cell_emb"] is not None else None
    ctrl_mean = batch[0]["ctrl_mean"] if batch[0]["ctrl_mean"] is not None else None

    if current_batch_size < cell_sentence_len:
        pad_size = cell_sentence_len - current_batch_size
        X_pad = torch.zeros((pad_size, feat_dim), dtype=torch.float32)
        X_batch = torch.cat([X_batch, X_pad], dim=0)
        
        pert_pad = torch.zeros((pad_size, pert_dim), dtype=torch.float32)
        if control_pert_vec is not None:
            pert_pad[:] = control_pert_vec
        pert_batch = torch.cat([pert_batch, pert_pad], dim=0)
        pert_names.extend(["__PAD__"] * pad_size)
        
        if pert_cell_batch is not None:
            pert_cell_pad = torch.zeros((pad_size, pert_cell_batch.shape[1]), dtype=torch.float32)
            pert_cell_batch = torch.cat([pert_cell_batch, pert_cell_pad], dim=0)

    return {
        "ctrl_cell_emb": X_batch,
        "pert_emb": pert_batch,
        "pert_name": pert_names,
        "batch": torch.zeros((1, cell_sentence_len), dtype=torch.float32),
        "pert_cell_emb": pert_cell_batch,
        "ctrl_mean": ctrl_mean,
    }, current_batch_size


# ===== 改进的训练循环 =====
def train_lora_v2(model,
                  lora_modules: Dict[str, nn.Module],
                  train_loader: DataLoader,
                  device: torch.device,
                  epochs: int = 3,
                  lr: float = 1e-3,
                  weight_decay: float = 0.0,
                  save_every: int = 1,
                  out_dir: str = "./lora_out",
                  save_full_checkpoint: bool = False,
                  use_delta_loss: bool = True,
                  delta_loss_weights: Optional[Dict[str, float]] = None):
    """LoRA微调主循环，支持Delta-Aware损失"""
    os.makedirs(out_dir, exist_ok=True)
    
    freeze_model_except_lora(model)
    lora_params = collect_lora_parameters(model)
    if not lora_params:
        raise RuntimeError("未找到可训练的LoRA参数")

    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    
    delta_loss_fn = None
    if use_delta_loss:
        if delta_loss_weights is None:
            delta_loss_weights = {
                "lambda_delta": 1.0, 
                "lambda_pearson": 0.5, 
                "lambda_mse": 0.5
            }
        delta_loss_fn = DeltaAwareLoss(**delta_loss_weights).to(device)
        print(f"[训练] 启用 Delta-Aware 损失，权重: {delta_loss_weights}")
    else:
        print("[训练] 使用原始损失函数")

    model.to(device)
    model.train()
    
    history = {
        "epoch_loss": [],
        "delta_mse": [],
        "delta_pearson": [],
        "mse": []
    }

    for epoch in range(epochs):
        total_loss = 0.0
        total_delta_mse = 0.0
        total_delta_pearson = 0.0
        total_mse = 0.0
        cnt = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (batch_dict, actual_bs) in pbar:
            batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_dict.items()}
            
            optimizer.zero_grad()
            loss = None
            loss_components = {}
            
            if use_delta_loss and delta_loss_fn is not None:
                try:
                    preds = model.predict_step(batch_dict, batch_idx, padded=False)
                    pred_tensor = preds.get("pert_cell_counts_preds", preds.get("preds"))
                    
                    pred_tensor = pred_tensor[:actual_bs]
                    target = batch_dict["pert_cell_emb"][:actual_bs]
                    ctrl_mean = batch_dict["ctrl_mean"]
                    
                    loss, loss_components = delta_loss_fn(pred_tensor, target, ctrl_mean)
                except Exception as e:
                    print(f"[警告] Delta损失计算失败: {e}，回退到原始损失")
            
            if loss is None:
                try:
                    out = model.training_step(batch_dict, batch_idx)
                    loss = out["loss"] if isinstance(out, dict) and "loss" in out else out
                except Exception as e:
                    try:
                        preds = model(batch_dict)
                        loss = torch.mean(preds["preds"] ** 2) if isinstance(preds, dict) and "preds" in preds else torch.mean(preds ** 2)
                    except Exception as e2:
                        raise RuntimeError("模型前向传播失败") from e2

            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=device)

            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            total_loss += step_loss
            
            if loss_components:
                total_delta_mse += loss_components.get("delta_mse", 0)
                total_delta_pearson += loss_components.get("delta_pearson", 0)
                total_mse += loss_components.get("mse", 0)
            
            cnt += 1
            postfix = {"loss": f"{(total_loss/cnt):.4e}"}
            if loss_components:
                postfix["δ_mse"] = f"{(total_delta_mse/cnt):.4e}"
                postfix["δ_prs"] = f"{(total_delta_pearson/cnt):.4f}"
            pbar.set_postfix(postfix)

        epoch_loss = total_loss / cnt
        history["epoch_loss"].append(epoch_loss)
        
        if loss_components and cnt > 0:
            history["delta_mse"].append(total_delta_mse / cnt)
            history["delta_pearson"].append(total_delta_pearson / cnt)
            history["mse"].append(total_mse / cnt)
        
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4e}")
        if loss_components:
            print(f"  Delta MSE: {(total_delta_mse/cnt):.4e}, "
                  f"Delta Pearson: {(total_delta_pearson/cnt):.4f}, "
                  f"MSE: {(total_mse/cnt):.4e}")

        lora_state = {}
        for name, m in lora_modules.items():
            lora_state[name] = {}
            if hasattr(m, "A") and m.A is not None:
                lora_state[name]["A"] = m.A.detach().cpu()
            if hasattr(m, "B") and m.B is not None:
                lora_state[name]["B"] = m.B.detach().cpu()
            lora_state[name]["r"] = m.r
            lora_state[name]["alpha"] = m.alpha

        save_data = {
            "epoch": epoch + 1, 
            "lora_state": lora_state,
            "history": history,
            "use_delta_loss": use_delta_loss,
            "delta_loss_weights": delta_loss_weights
        }
        torch.save(save_data, os.path.join(out_dir, f"lora_epoch{epoch+1}.pth"))
        
        if save_full_checkpoint:
            torch.save({"epoch": epoch + 1, "model_state": model.state_dict()}, 
                      os.path.join(out_dir, f"full_model_epoch{epoch+1}.pth"))

    return os.path.join(out_dir, f"lora_epoch{epochs}.pth"), history


# ------------- CLI & main -------------
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LoRA微调脚本")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--adata", type=str, required=True)
    parser.add_argument("--embed_key", type=str, default=None)
    parser.add_argument("--pert_col", type=str, default="target_gene")
    parser.add_argument("--output_lora", type=str, default="./lora_state.pth")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--target_modules", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./lora_out")
    parser.add_argument("--save_full_checkpoint", action="store_true")
    
    # V2新增参数
    parser.add_argument("--batch_col", type=str, default=None)
    parser.add_argument("--use_delta_loss", action="store_true")
    parser.add_argument("--no_delta_loss", action="store_true")
    parser.add_argument("--lambda_delta", type=float, default=1.0)
    parser.add_argument("--lambda_pearson", type=float, default=0.5)
    parser.add_argument("--lambda_mse", type=float, default=0.5)
    parser.add_argument("--control_label", type=str, default=None)
    
    return parser.parse_args()


def main():
    """主函数：加载模型、数据，执行LoRA微调"""
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        import sys
        sys.path.append("/data3/fanpeishan/state/src")
        from state.tx.models.state_transition import StateTransitionPerturbationModel
    except Exception:
        try:
            from state.tx.models.state_transition import StateTransitionPerturbationModel
        except Exception as e:
            raise RuntimeError("无法导入模型") from e

    import yaml
    cfg_path = os.path.join(args.model_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件缺失: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = args.checkpoint or os.path.join(args.model_dir, "checkpoints", "final.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点缺失: {ckpt_path}")

    print(f"[加载] 模型: {ckpt_path}")
    model = StateTransitionPerturbationModel.load_from_checkpoint(ckpt_path)
    model.eval()
    cell_sentence_len = model.cell_sentence_len
    
    pert_map_path = os.path.join(args.model_dir, "pert_onehot_map.pt")
    if not os.path.exists(pert_map_path):
        raise FileNotFoundError(f"扰动映射缺失: {pert_map_path}")
    pert_onehot_map = torch.load(pert_map_path, map_location="cpu", weights_only=False)

    print(f"[加载] 数据: {args.adata}")
    adata = sc.read_h5ad(args.adata) if args.adata.endswith(".h5ad") else sc.read(args.adata)

    # 确定控制组标签
    control_label = args.control_label or {
        "drugname_drugconc": "[('DMSO_TF', 0.0, 'uM')]",
        "perturbation_2": "Control"
    }.get(args.pert_col, cfg["data"]["kwargs"].get("control_pert", "[('DMSO_TF', 0.0, 'uM')]"))
    
    print(f"[配置] 控制组标签: '{control_label}'")
    use_delta_loss = args.use_delta_loss and not args.no_delta_loss
    print(f"[配置] Delta损失: {'启用' if use_delta_loss else '禁用'}")
    print(f"[配置] 批次列: {args.batch_col if args.batch_col else '未指定'}")

    ds = AnnDataPerturbationDatasetV2(
        adata, 
        pert_onehot_map, 
        pert_col=args.pert_col,
        embed_key=args.embed_key, 
        control_pert_override=control_label,
        control_label=control_label,
        batch_col=args.batch_col
    )
    
    def collate_wrapper(batch):
        return collate_fn_v2(
            batch, 
            cell_sentence_len=cell_sentence_len, 
            pert_dim=ds.pert_dim,
            control_pert_vec=pert_onehot_map.get(control_label, None)
        )
    
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_wrapper)

    # 配置LoRA
    target_keywords = [kw.strip() for kw in args.target_modules.split(",") if kw.strip()] if args.target_modules else None

    print(f"[LoRA] 配置 - rank={args.lora_rank}, alpha={args.lora_alpha}")
    model, lora_modules = replace_linear_with_lora(model, r=args.lora_rank, alpha=args.lora_alpha,
                                                  target_module_keywords=target_keywords)
    print(f"[LoRA] 替换模块数量: {len(lora_modules)}")

    freeze_model_except_lora(model)
    lora_params = collect_lora_parameters(model)
    print(f"[LoRA] 可训练参数: {len(lora_params)}")

    # 训练配置
    delta_loss_weights = {
        "lambda_delta": args.lambda_delta,
        "lambda_pearson": args.lambda_pearson,
        "lambda_mse": args.lambda_mse
    }

    print("[启动] LoRA微调")
    out_path, history = train_lora_v2(
        model, 
        lora_modules, 
        dataloader, 
        device=device,
        epochs=args.epochs, 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        out_dir=args.save_dir, 
        save_full_checkpoint=args.save_full_checkpoint,
        use_delta_loss=use_delta_loss,
        delta_loss_weights=delta_loss_weights
    )

    # 保存最终状态
    final_state = {}
    for name, m in lora_modules.items():
        final_state[name] = {
            "r": m.r,
            "alpha": m.alpha
        }
        if hasattr(m, "A") and m.A is not None:
            final_state[name]["A"] = m.A.detach().cpu()
        if hasattr(m, "B") and m.B is not None:
            final_state[name]["B"] = m.B.detach().cpu()

    save_data = {
        "lora_state": final_state, 
        "cfg": {"rank": args.lora_rank, "alpha": args.lora_alpha},
        "history": history,
        "use_delta_loss": use_delta_loss,
        "delta_loss_weights": delta_loss_weights,
        "batch_col": args.batch_col,
        "control_label": control_label
    }
    torch.save(save_data, args.output_lora)
    print(f"[完成] LoRA状态已保存至: {args.output_lora}")


if __name__ == "__main__":
    main()