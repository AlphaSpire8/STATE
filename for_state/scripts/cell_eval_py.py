#!/usr/bin/env python3
"""
使用 cell-eval 计算药物扰动评估指标
"""

from cell_eval import MetricsEvaluator, DESortBy
import polars as pl

# ============================================================
# 1. 初始化评估器
# ============================================================
evaluator = MetricsEvaluator(
    adata_pred="/data3/fanpeishan/state/for_state/finetune_pipeline/results/test2/c37_prep_preprocess_infer.h5ad",       # 预测数据路径
    adata_real="/data3/fanpeishan/state/for_state/run_results/run14/c37_prep.h5ad",       # 真实数据路径
    control_pert="[('DMSO_TF', 0.0, 'uM')]",               # 控制扰动名称
    pert_col="drugname_drugconc",                      # 扰动列名
    de_method="wilcoxon",                 # Wilcoxon 检验（与 eval_drug.py 一致）
    num_threads=12,                       # 使用所有 CPU 核心
    outdir="/data3/fanpeishan/state/for_state/finetune_pipeline/results/test2/run_results_python",              # 输出目录
)

# ============================================================
# 2. 配置指标参数
# ============================================================
metric_configs = {
    "overlap_at_100": {
        "fdr_threshold": 1.0,                    # 设置为 1.0 以尽量避免 FDR 过滤
        "sort_by": DESortBy.ABS_FOLD_CHANGE,     # 使用绝对值 LogFC 排序
    }
}

# ============================================================
# 3. 运行评估
# ============================================================
results, agg_results = evaluator.compute(
    profile="full",
    metric_configs=metric_configs,
    basename="eval_results.csv",
    write_csv=True,
)

# ============================================================
# 4. 提取三个关键指标
# ============================================================
# results 是 polars DataFrame，包含每个扰动的详细结果
# 需要筛选出三个指标

# 方法1：直接从 results 提取
pearson_delta_series = results["pearson_delta"]
overlap_100_series = results["overlap_at_100"]
mse_series = results["mse"]

# 计算平均值
pearson_mean = pearson_delta_series.mean()
overlap_mean = overlap_100_series.mean()
mse_mean = mse_series.mean()

# ============================================================
# 5. 输出结果（与 eval_drug.py 格式一致）
# ============================================================
print("\n" + "="*60)
print(" DRUG PERTURBATION EVALUATION REPORT")
print("="*60)
print(f"1. PCC-Delta:   {pearson_mean:.4f}")
print(f"2. Overlap@100: {overlap_mean:.4f}")
print(f"3. MSE:         {mse_mean:.6f}")
print("="*60 + "\n")
