import warnings
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from tqdm import tqdm
warnings.filterwarnings("ignore")

N_HVG = 2000      # 评估基因数 
K_DE  = 100       # Top K 差异基因
MIN_CELLS = 3     # 最小细胞数阈值
# ==================================

def ensure_dense(adata, name="data"):
    if not isinstance(adata.X, np.ndarray):
        print(f"[INFO] Converting {name} sparse matrix to dense...")
        adata.X = adata.X.toarray()
    return adata

def align_genes(pred, real):
    common = pred.var_names.intersection(real.var_names)
    if len(common) < 1000:
        raise ValueError(f"Too few common genes: {len(common)}")
    return pred[:, common].copy(), real[:, common].copy()

def select_hvg(real, pred, n_top=N_HVG):

    print(f"[INFO] Selecting Top {n_top} HVGs based on REAL data...")
    try:
        sc.pp.highly_variable_genes(real, n_top_genes=n_top, subset=False, flavor='seurat')
        hvg_mask = real.var['highly_variable'].values
    except:
        print("[WARN] Seurat HVG failed, falling back to variance ranking.")
        vars = np.var(real.X, axis=0)
        top_idx = np.argsort(vars)[-n_top:]
        hvg_mask = np.zeros(real.n_vars, dtype=bool)
        hvg_mask[top_idx] = True

    return real[:, hvg_mask].copy(), pred[:, hvg_mask].copy()

def get_pseudobulk(adata, group_col, min_cells=MIN_CELLS):
    counts = adata.obs[group_col].value_counts()
    valid_groups = counts[counts >= min_cells].index
    
    subset = adata[adata.obs[group_col].isin(valid_groups)]
    
    # 转换为 DataFrame 进行聚合
    df = pd.DataFrame(
        subset.X, 
        index=subset.obs[group_col].astype(str).values,
        columns=subset.var_names
    )
    return df.groupby(level=0).mean()

def _worker_real_de(adata_subset, group, ctrl, k):
    try:
        sc.tl.rank_genes_groups(
            adata_subset, 
            groupby='temp_group', 
            reference=ctrl, 
            groups=[group], 
            method='wilcoxon', 
            use_raw=False
        )
        
        res = adata_subset.uns['rank_genes_groups']
        genes = res['names'][group]
        logfc = res['logfoldchanges'][group]
        
        df = pd.DataFrame({'gene': genes, 'abs_logfc': np.abs(logfc)})
        top_genes = df.sort_values('abs_logfc', ascending=False).head(k)['gene'].tolist()
        
        return group, top_genes
    except Exception:
        return group, []

def get_real_de_parallel(adata, group_col, ctrl, k=K_DE, min_cells=MIN_CELLS, n_jobs=10):
    """
    并行差异分析
    """
    print(f"[INFO] Running Parallel DE on REAL data (Wilcoxon, n_jobs={n_jobs})...")
    
    counts = adata.obs[group_col].value_counts()
    valid_groups = counts[counts >= min_cells].index.tolist()
    
    if ctrl not in valid_groups:
        if ctrl in adata.obs[group_col].values:
            valid_groups.append(ctrl)
        else:
            return {}

    test_groups = [g for g in valid_groups if g != ctrl]
    
    subset = adata[adata.obs[group_col].isin(valid_groups)].copy()
    subset.obs['temp_group'] = subset.obs[group_col].astype('category')
    
    # 并行计算
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_worker_real_de)(subset, g, ctrl, k) 
        for g in tqdm(test_groups, desc="Real DE Analysis")
    )
    
    return {g: genes for g, genes in results if genes}

def get_pred_top_k_diff(pred_bulk_df, ctrl, k=K_DE):
    print(f"[INFO] Fast Ranking on PRED data (Simple Diff, Top {k})...")
    
    if ctrl not in pred_bulk_df.index:
        return {}
    
    ctrl_vec = pred_bulk_df.loc[ctrl].values
    valid_perts = [p for p in pred_bulk_df.index if p != ctrl]
    
    de_results = {}
    
    for pert in tqdm(valid_perts, desc="Pred Fast Ranking"):
        pert_vec = pred_bulk_df.loc[pert].values
        diff = pert_vec - ctrl_vec
        
        top_indices = np.argsort(np.abs(diff))[-k:][::-1]
        
        top_genes = pred_bulk_df.columns[top_indices].tolist()
        de_results[pert] = top_genes
        
    return de_results

def main():
    p = argparse.ArgumentParser(description="Optimized scBenchmark Evaluation (Drug/Gene)")
    p.add_argument("--pred", required=True, help="Path to prediction h5ad")
    p.add_argument("--real", required=True, help="Path to ground truth h5ad")
    p.add_argument("--pert_col", default="drugname_drugconc", help="Column name for perturbation (e.g., drug, target_gene)")
    p.add_argument("--ctrl", default="[('DMSO_TF', 0.0, 'uM')]", help="Control group label")
    p.add_argument("--n_jobs", type=int, default=3, help="CPU cores for Real DE analysis")
    args = p.parse_args()

    print("Loading data...")
    pred = sc.read_h5ad(args.pred)
    real = sc.read_h5ad(args.real)
    pred, real = align_genes(pred, real)
    
    pred = ensure_dense(pred, "pred")
    real = ensure_dense(real, "real")

    real, pred = select_hvg(real, pred, n_top=N_HVG)

    pert_col = args.pert_col
    ctrl = args.ctrl
    real.obs[pert_col] = real.obs[pert_col].astype(str)
    pred.obs[pert_col] = pred.obs[pert_col].astype(str)
    
    common_perts = sorted(list(set(real.obs[pert_col]) & set(pred.obs[pert_col])))
    if ctrl not in common_perts:
        raise ValueError(f"Control '{ctrl}' missing in intersection!")

    print("[INFO] Computing Pseudobulk...")
    real_bulk_df = get_pseudobulk(real, pert_col)
    pred_bulk_df = get_pseudobulk(pred, pert_col)
    
    if ctrl not in real_bulk_df.index or ctrl not in pred_bulk_df.index:
        raise ValueError("Control group filtered out due to low cell count!")

    real_de_dict = get_real_de_parallel(real, pert_col, ctrl, n_jobs=args.n_jobs)
    
    pred_de_dict = get_pred_top_k_diff(pred_bulk_df, ctrl, k=K_DE)

    metrics = {'mse': [], 'pcc_delta': []}
    overlap_list = []
    
    eval_perts = [p for p in real_bulk_df.index if p != ctrl and p in pred_bulk_df.index]
    
    real_ctrl_vec = real_bulk_df.loc[ctrl].values
    pred_ctrl_vec = pred_bulk_df.loc[ctrl].values
    
    print(f"Calculating metrics for {len(eval_perts)} perturbations...")
    
    for pert in eval_perts:
        y_real = real_bulk_df.loc[pert].values
        y_pred = pred_bulk_df.loc[pert].values
        
        # Self-Normalized Delta
        delta_real = y_real - real_ctrl_vec
        delta_pred = y_pred - pred_ctrl_vec
        
        metrics['mse'].append(mean_squared_error(y_real, y_pred))
        
        if np.std(delta_real) > 1e-9 and np.std(delta_pred) > 1e-9:
            r, _ = stats.pearsonr(delta_real, delta_pred)
            metrics['pcc_delta'].append(r)
        else:
            metrics['pcc_delta'].append(0)
            
        # Overlap (Recall) 
        if pert in real_de_dict and pert in pred_de_dict:
            real_genes = set(real_de_dict[pert])
            pred_genes = set(pred_de_dict[pert])
            
            if len(real_genes) > 0:
                overlap = len(real_genes & pred_genes) / len(real_genes)
                overlap_list.append(overlap)

    print("\n" + "="*60)
    print(f"  FINAL EVAL_DRUG REPORT (Optimized)")
    print(f"  HVG: {N_HVG} | DE Ranking: Top {K_DE} (Abs LogFC)")
    print("="*60)
    print(f"1. PCC-Delta (Accuracy)     : {np.mean(metrics['pcc_delta']):.4f}")
    
    if overlap_list:
        print(f"2. Common-DEGs (Recall)     : {np.mean(overlap_list):.4f}")
    else:
        print(f"2. Common-DEGs (Recall)     : NaN (Real signal too weak or data mismatch)")
        
    print(f"3. MSE (Error)              : {np.mean(metrics['mse']):.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()