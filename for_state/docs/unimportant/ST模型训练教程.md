# ST 模型训练教程

## 目录

1. [概述](#1-概述)
2. [环境准备](#2-环境准备)
3. [数据准备](#3-数据准备)
4. [配置文件编写](#4-配置文件编写)
5. [训练流程](#5-训练流程)
6. [推理/预测](#6-推理预测)
7. [评估](#7-评估)
8. [常见问题和调试技巧](#8-常见问题和调试技巧)

---

## 1. 概述

### 1.1 什么是 ST 模型？

**ST (State Transition) 模型**是一个用于**细胞扰动响应预测**的深度学习模型。它可以：

- **预测基因敲除/药物处理后的细胞状态变化**
- **学习扰动对细胞转录组的影响**
- **支持零样本（zero-shot）和少样本（few-shot）学习**
- **跨细胞类型泛化**

### 1.2 模型架构概览

ST 模型采用基于 Transformer 的架构，主要组件包括：

- **Cell Encoder**: 将细胞表达谱编码为高维表征
- **Perturbation Encoder**: 编码扰动信息（基因/药物）
- **Transformer Backbone**: 处理细胞集合的序列信息
- **Decoder**: 预测扰动后的细胞状态

### 1.3 应用场景

- 药物筛选和虚拟实验
- 基因功能预测
- 细胞类型特异性扰动响应研究
- 跨数据集的扰动效应迁移学习

---

## 2. 环境准备

### 2.1 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: 3.9-3.11
- **GPU**: NVIDIA GPU with CUDA support (推荐 16GB+ VRAM)
- **内存**: 至少 32GB RAM (推荐 64GB+)

### 2.2 依赖安装

#### 方法 1: 使用 uv（推荐）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目
git clone <your-state-repository>
cd state

# 安装依赖（uv 会自动创建虚拟环境）
uv sync
```

#### 方法 2: 使用 pip

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装项目
pip install -e .
```

### 2.3 验证安装

```bash
# 检查 state 命令是否可用
uv run state --help

# 或者激活虚拟环境后
state --help
```

应该看到类似输出：

```
usage: state [-h] {tx,emb} ...

STATE: Single-cell Transcriptomics Analysis Toolkit
...
```

---

## 3. 数据准备

### 3.1 数据格式要求

ST 模型使用 **AnnData** 格式（`.h5ad` 文件）作为输入。数据应满足以下要求：

#### 必需字段

1. **`adata.X`**: 表达矩阵（cells × genes）
   - 推荐使用 log-normalized 数据
   - 或使用 HVG (Highly Variable Genes) 特征

2. **`adata.obs` 必需列**:
   - `perturbation_column`: 扰动标签（如基因名、药物名）
   - `cell_type_column`: 细胞类型
   - `batch_column`: 批次信息（如实验批次、plate）
   - `control_label`: 对照组标签（如 "non-targeting", "DMSO"）

3. **`adata.var`**: 基因元数据
   - `gene_names`: 基因名称（作为 index）
   - 可选：`highly_variable`: 标记 HVG

#### 可选字段

- **`adata.obsm['X_hvg']`**: HVG 表达矩阵（用于 gene 输出空间）
- **`adata.uns`**: 其他元数据

### 3.2 数据预处理步骤

#### 步骤 1: 过滤扰动效果

使用 `filter_on_target_knockdown` 命令过滤低效扰动：

```bash
filter_on_target_knockdown \
    --perturbation-column "gene" \
    --control-label "non-targeting" \
    --output "data_filtered.h5ad" \
    --preprocess \
    "data_raw.h5ad"
```

**参数说明**:
- `--perturbation-column`: 扰动列名
- `--control-label`: 对照标签
- `--preprocess`: 自动进行标准化和 log 转换
- `--output`: 输出文件路径

#### 步骤 2: 合并多个数据集（可选）

如果有多个 h5ad 文件需要合并：

```python
import anndata as ad
import scanpy as sc

# 读取文件
files = [
    'dataset1_filtered.h5ad',
    'dataset2_filtered.h5ad',
    'dataset3_filtered.h5ad'
]
adatas = [ad.read_h5ad(f) for f in files]

# 合并（inner join 保留共同基因）
adata_merged = ad.concat(adatas, join='inner')
adata_merged.obs_names_make_unique()

# 保存
adata_merged.write('merged_data.h5ad')
```

#### 步骤 3: 计算 HVG（如果需要）

```python
import scanpy as sc
from scipy import sparse

# 假设数据已经 log-normalized
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# 提取 HVG 数据到 obsm
hvg_data = adata.X[:, adata.var['highly_variable']]
if sparse.issparse(hvg_data):
    hvg_data = hvg_data.toarray()
adata.obsm['X_hvg'] = hvg_data

# 保存
adata.write('processed_data.h5ad')
```

### 3.3 组织数据目录

推荐的数据目录结构：

```
project_root/
├── data/
│   ├── raw/
│   │   └── original_data.h5ad
│   ├── processed/
│   │   ├── dataset1.h5ad
│   │   ├── dataset2.h5ad
│   │   └── merged.h5ad
│   └── configs/
│       └── training_config.toml
```

---

## 4. 配置文件编写

### 4.1 TOML 配置文件结构

训练配置使用 **TOML** 格式，包含四个主要部分：

```toml
[datasets]      # 数据集路径
[training]      # 训练集划分
[zeroshot]      # 零样本划分（按细胞类型）
[fewshot]       # 少样本划分（按扰动列表）
```

### 4.2 配置部分详解

#### 4.2.1 `[datasets]` - 数据集路径

定义数据集名称和路径映射：

```toml
[datasets]
# 格式：数据集名称 = "路径"
# 支持通配符（glob pattern）

# 单个文件
my_dataset = "/path/to/data.h5ad"

# 多个文件（使用通配符）
replogle = "/data/replogle/{cell1,cell2,cell3}.h5ad"

# 或使用目录
tahoe = "/data/tahoe/*.h5ad"
```

**注意**:
- 路径必须是**绝对路径**
- 支持 bash 风格通配符：`{a,b,c}` 或 `*`
- 数据集名称将用于后续配置引用

#### 4.2.2 `[training]` - 训练集划分

指定哪些数据集用于训练：

```toml
[training]
# 格式：数据集名称 = "train"
my_dataset = "train"
replogle = "train"
```

**默认行为**:
- 标记为 `"train"` 的数据集的**所有细胞类型**都会进入训练集
- 除非被 `[zeroshot]` 或 `[fewshot]` 覆盖

#### 4.2.3 `[zeroshot]` - 零样本划分

按**细胞类型**划分验证集/测试集：

```toml
[zeroshot]
# 格式：数据集名称.细胞类型 = "val" 或 "test"

# 将 dataset1 的 celltype_a 作为验证集
"dataset1.celltype_a" = "val"

# 将 dataset2 的 celltype_b 作为测试集
"dataset2.celltype_b" = "test"
```

**说明**:
- 整个细胞类型的**所有扰动**都会被移到指定集合
- 用于测试模型的跨细胞类型泛化能力

#### 4.2.4 `[fewshot]` - 少样本划分

按**扰动列表**精细划分：

```toml
[fewshot]
# 格式：数据集名称.细胞类型.扰动列名 = { val = [...], test = [...] }

[fewshot."dataset1.celltype_a"]
val = ["GENE1", "GENE2", "GENE3"]
test = ["GENE4", "GENE5", "GENE6"]
```

**说明**:
- 允许在同一细胞类型内划分不同扰动
- `val` 和 `test` 都是可选的
- 列表中的扰动会从训练集移除

### 4.3 完整示例配置

参见项目中的 [`example_training_config.toml`](./example_training_config.toml) 文件。

### 4.4 配置优先级

配置的应用顺序（优先级从低到高）：

1. `[training]`: 默认所有数据进入训练集
2. `[zeroshot]`: 覆盖特定细胞类型
3. `[fewshot]`: 覆盖特定扰动

---

## 5. 训练流程

### 5.1 基本训练命令

```bash
uv run state tx train \
    data.kwargs.toml_config_path="path/to/config.toml" \
    data.kwargs.pert_col="gene" \
    data.kwargs.cell_type_key="cell_type" \
    data.kwargs.batch_col="batch" \
    data.kwargs.control_pert="non-targeting" \
    training.max_steps=40000 \
    output_dir="output" \
    name="my_experiment"
```

### 5.2 关键参数说明

#### 数据参数 (`data.kwargs.*`)

| 参数 | 说明 | 示例 |
|------|------|------|
| `toml_config_path` | TOML 配置文件路径 | `"configs/train.toml"` |
| `pert_col` | 扰动列名 | `"gene"` / `"drug"` |
| `cell_type_key` | 细胞类型列名 | `"cell_type"` / `"cell_line"` |
| `batch_col` | 批次列名 | `"batch"` / `"gem_group"` / `"plate"` |
| `control_pert` | 对照标签 | `"non-targeting"` / `"DMSO"` |
| `output_space` | 输出空间 | `"gene"` (HVG) / `"all"` (全基因) |
| `num_workers` | 数据加载线程数 | `8` |
| `perturbation_features_file` | 扰动特征文件路径 | `"ESM2_features.pt"` |

**注意**:
- `pert_col`, `cell_type_key`, `batch_col` 必须与数据中的列名匹配
- `control_pert` 必须与数据中的对照标签完全一致

#### 训练参数 (`training.*`)

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `max_steps` | 最大训练步数 | 40000 | 40000-80000 |
| `batch_size` | 批次大小 | 16 | 32-64 |
| `lr` | 学习率 | 1e-4 | 1e-4 ~ 1e-3 |
| `ckpt_every_n_steps` | 保存检查点间隔 | 2000 | 2000-4000 |
| `val_freq` | 验证频率 | 2000 | 2000 |
| `gradient_clip_val` | 梯度裁剪 | 10 | 10 |

#### 模型参数 (`model.kwargs.*`)

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `cell_set_len` | 细胞序列长度 | 512 | 64-512 |
| `hidden_dim` | 隐藏层维度 | 696 | 128-696 |
| `batch_encoder` | 是否使用批次编码器 | False | True/False |
| `predict_residual` | 预测残差 | True | True |

**模型大小选择**:

```bash
# 小模型（快速原型）
model=state_sm  # small model

# 中等模型（推荐）
model=state  # default model

# 大模型（性能优先）
model=state_lg  # large model（需要更多 GPU 内存）
```

#### Wandb 参数 (`wandb.*`)

| 参数 | 说明 | 示例 |
|------|------|------|
| `project` | 项目名称 | `"st_training"` |
| `entity` | 团队/用户名 | `"your_team"` |
| `tags` | 实验标签 | `"[exp1,baseline]"` |

### 5.3 完整训练示例

#### 示例 1: 基本训练（Replogle 数据集）

```bash
uv run state tx train \
    data.kwargs.toml_config_path="configs/replogle.toml" \
    data.kwargs.num_workers=4 \
    data.kwargs.output_space="gene" \
    data.kwargs.batch_col="gem_group" \
    data.kwargs.pert_col="gene" \
    data.kwargs.cell_type_key="cell_line" \
    data.kwargs.control_pert="non-targeting" \
    training.max_steps=80000 \
    training.ckpt_every_n_steps=2000 \
    training.batch_size=64 \
    training.lr=1e-3 \
    model.kwargs.cell_set_len=64 \
    model.kwargs.hidden_dim=128 \
    model.kwargs.batch_encoder=True \
    model=state \
    wandb.tags="[replogle_baseline]" \
    output_dir="results" \
    name="replogle_exp1"
```

#### 示例 2: 使用预训练模型微调

```bash
uv run state tx train \
    data.kwargs.toml_config_path="configs/finetune.toml" \
    model.kwargs.init_from="path/to/pretrained/checkpoint.ckpt" \
    training.max_steps=20000 \
    training.lr=5e-5 \
    output_dir="results" \
    name="finetuned_model"
```

#### 示例 3: 使用扰动特征（如 ESM2 嵌入）

```bash
uv run state tx train \
    data.kwargs.toml_config_path="configs/vcc.toml" \
    data.kwargs.perturbation_features_file="data/ESM2_pert_features.pt" \
    data.kwargs.pert_col="target_gene" \
    training.max_steps=40000 \
    model=state_sm \
    output_dir="results" \
    name="with_esm2_features"
```

### 5.4 训练输出

训练会在输出目录创建以下文件：

```
output_dir/experiment_name/
├── config.yaml                    # 完整训练配置
├── checkpoints/                   # 检查点目录
│   ├── last.ckpt                 # 最新检查点
│   ├── step=2000.ckpt            # 定期保存的检查点
│   ├── step=4000.ckpt
│   └── final.ckpt                # 最终检查点
├── data_module.torch              # 数据模块状态
├── cell_type_onehot_map.pkl       # 细胞类型编码映射
├── pert_onehot_map.pt             # 扰动编码映射
├── batch_onehot_map.pkl           # 批次编码映射
├── var_dims.pkl                   # 变量维度信息
└── wandb_path.txt                 # Wandb 运行路径（如果启用）
```

### 5.5 监控训练

#### 使用 Wandb（推荐）

```bash
# 启用 wandb
uv run state tx train \
    use_wandb=true \
    wandb.project="my_project" \
    wandb.entity="my_team" \
    ... # 其他参数
```

然后访问 [wandb.ai](https://wandb.ai) 查看实时训练曲线。

#### 查看日志

```bash
# 训练过程中的输出会显示：
# - 当前步数和损失
# - 每个 epoch 的进度
# - 验证集性能（如果配置了验证集）
# - GPU 内存使用
```

---

## 6. 推理/预测

### 6.1 推理命令

训练完成后，使用 `state tx infer` 进行推理：

```bash
uv run state tx infer \
    --adata "data/test_data.h5ad" \
    --output "predictions.h5ad" \
    --model-dir "results/my_experiment" \
    --checkpoint "results/my_experiment/checkpoints/final.ckpt" \
    --pert-col "gene"
```

### 6.2 推理参数说明

| 参数 | 说明 | 必需 |
|------|------|------|
| `--adata` | 输入数据路径 | ✓ |
| `--output` | 输出文件路径 | ✓ |
| `--model-dir` | 模型目录 | ✓ |
| `--checkpoint` | 检查点路径 | ✓ |
| `--pert-col` | 扰动列名 | ✓ |
| `--batch-col` | 批次列名 | ✗ |
| `--control-pert` | 对照标签 | ✗ |

### 6.3 推理输出

推理结果保存在输出的 `.h5ad` 文件中：

- **`adata.X`**: 预测的细胞状态（与输入形状相同）
- **`adata.obs`**: 保留输入的所有元数据
- **`adata.var`**: 保留基因信息

### 6.4 使用预测命令

也可以使用 `state tx predict` 命令：

```bash
uv run state tx predict \
    --output_dir "results/my_experiment" \
    --checkpoint "last.ckpt"
```

这会自动使用训练时的数据配置进行预测。

---

## 7. 评估

### 7.1 使用 cell-eval 评估

推荐使用 `cell-eval` 工具评估预测结果：

```bash
cell-eval run \
    -ap predictions.h5ad \
    -ar ground_truth.h5ad \
    -o results/ \
    --control-pert "non-targeting" \
    --pert-col "gene" \
    --profile minimal \
    --celltype-col "cell_type" \
    --batch-size 1024 \
    --num-threads 16
```

### 7.2 评估参数说明

| 参数 | 说明 |
|------|------|
| `-ap` | 预测数据路径 |
| `-ar` | 真实数据路径 |
| `-o` | 输出目录 |
| `--control-pert` | 对照标签 |
| `--pert-col` | 扰动列名 |
| `--profile` | 评估配置（`minimal`/`full`） |
| `--celltype-col` | 细胞类型列名 |

### 7.3 评估指标

`cell-eval` 会计算多种指标：

- **R² (R-squared)**: 总体预测准确性
- **MSE (Mean Squared Error)**: 均方误差
- **Pearson Correlation**: 基因间相关性
- **Per-perturbation metrics**: 每个扰动的性能
- **Per-celltype metrics**: 每个细胞类型的性能

### 7.4 查看评估结果

评估结果保存在输出目录：

```
results/
├── {celltype}_agg_results.csv     # 聚合结果
├── {celltype}_pert_results.csv    # 每个扰动的结果
└── plots/                         # 可视化图表（如果配置了）
```

读取结果：

```python
import pandas as pd

# 读取聚合结果
results = pd.read_csv('results/celltype_agg_results.csv')

# 查看平均性能
print(results[results.statistic == 'mean'])
```

---

## 8. 常见问题和调试技巧

### 8.1 内存问题

#### 问题：OOM (Out of Memory) 错误

**解决方案**:

1. **减少批次大小**:
   ```bash
   training.batch_size=16  # 或更小
   ```

2. **减少 cell_set_len**:
   ```bash
   model.kwargs.cell_set_len=64  # 从 512 降到 64
   ```

3. **使用梯度累积**（如果支持）

4. **减少 num_workers**:
   ```bash
   data.kwargs.num_workers=2
   ```

### 8.2 数据加载问题

#### 问题：找不到数据文件

**检查清单**:
- TOML 配置中的路径是否为**绝对路径**
- 文件是否真实存在
- 通配符语法是否正确

**调试命令**:
```bash
# 验证通配符扩展
ls -l /path/to/data/{file1,file2}.h5ad
```

#### 问题：列名不匹配

**错误示例**:
```
KeyError: 'gene' not in adata.obs
```

**解决方案**:
```python
import anndata as ad

adata = ad.read_h5ad('data.h5ad')
print(adata.obs.columns)  # 查看所有列名

# 确保配置中的列名与数据匹配
# data.kwargs.pert_col="actual_column_name"
```

### 8.3 训练不收敛

#### 可能原因和解决方案

1. **学习率过大**:
   ```bash
   training.lr=1e-5  # 降低学习率
   ```

2. **数据未正确归一化**:
   - 确保使用 `--preprocess` 标志预处理数据
   - 或手动进行 log-normalization

3. **梯度爆炸**:
   ```bash
   training.gradient_clip_val=1.0  # 更严格的梯度裁剪
   ```

4. **数据不平衡**:
   - 检查每个扰动的样本数
   - 考虑使用数据增强或重采样

### 8.4 配置文件错误

#### 问题：TOML 语法错误

**常见错误**:

```toml
# ❌ 错误：字符串未加引号
[datasets]
example = /path/to/data.h5ad

# ✓ 正确
[datasets]
example = "/path/to/data.h5ad"
```

```toml
# ❌ 错误：键包含特殊字符未加引号
[fewshot]
dataset.celltype = {...}

# ✓ 正确
[fewshot]
"dataset.celltype" = {...}
```

**验证 TOML 文件**:
```python
import toml

try:
    config = toml.load('config.toml')
    print("配置文件格式正确！")
except Exception as e:
    print(f"配置文件错误: {e}")
```

### 8.5 推理问题

#### 问题：模型输入特征不匹配

**错误示例**:
```
Expected input features: 2000, but got: 18080
```

**解决方案**:

如果训练时使用了 `output_space="gene"` (HVG)，推理数据也需要 HVG 特征：

```python
import pickle
import anndata as ad

# 读取训练时保存的 HVG 列表
hvg_names = pickle.load(open('results/my_exp/var_dims.pkl', 'rb'))['gene_names']

# 准备推理数据
adata_test = ad.read_h5ad('test_data.h5ad')
adata_test.var.index = hvg_names
adata_test.X = adata_test.obsm['X_hvg']
adata_test.write('test_data_prepared.h5ad')
```

### 8.6 性能优化技巧

#### 1. 多 GPU 训练

```bash
uv run state tx train \
    training.devices=4 \
    training.strategy="ddp" \
    ... # 其他参数
```

#### 2. 混合精度训练

混合精度训练会自动启用（对于支持的模型）。

#### 3. 数据加载优化

```bash
data.kwargs.num_workers=8        # 增加数据加载线程
data.kwargs.pin_memory=true      # 固定内存（加速 CPU->GPU 传输）
```

#### 4. 检查点管理

```bash
# 只保留最近N个检查点（节省磁盘空间）
# 修改 src/state/tx/utils.py 中的 checkpoint callback 配置
```

### 8.7 调试模式

启用详细日志：

```bash
# 设置 Python 日志级别
export PYTHONLOGLEVEL=DEBUG

# 运行训练
uv run state tx train ...
```

检查数据加载：

```python
# 在训练前手动检查数据
from cell_load.data_modules import PerturbationDataModule
import toml

config = toml.load('config.toml')
dm = PerturbationDataModule(
    toml_config_path='config.toml',
    pert_col='gene',
    # ... 其他参数
)
dm.setup(stage='fit')

# 检查训练集大小
train_dl = dm.train_dataloader()
print(f"训练批次数: {len(train_dl)}")
print(f"批次大小: {train_dl.batch_size}")

# 检查一个批次
batch = next(iter(train_dl))
print(f"批次键: {batch.keys()}")
print(f"X 形状: {batch['X'].shape}")
```

### 8.8 常用检查清单

训练前检查：

- [ ] 配置文件路径正确且文件存在
- [ ] 数据文件路径正确且可访问
- [ ] 所有列名（pert_col, cell_type_key, batch_col）与数据匹配
- [ ] 对照标签（control_pert）在数据中存在
- [ ] GPU 内存充足（至少 16GB 推荐）
- [ ] 有足够的磁盘空间保存检查点

推理前检查：

- [ ] 模型检查点文件存在
- [ ] 推理数据格式与训练数据一致
- [ ] 如果使用 HVG，确保推理数据也有 X_hvg
- [ ] 列名与训练时一致

---

## 附录

### A. 参考资源

- **项目文档**: [`for_state/docs/`](../important/)
- **教程 Notebook**: [`for_state/run__commands/tutorial/`](../../run__commands/tutorial/)
- **配置示例**: [`examples/`](../../../examples/)

### B. 相关命令

```bash
# 查看完整帮助
state tx train --help

# 推理帮助
state tx infer --help

# 预处理帮助
filter_on_target_knockdown --help

# 评估帮助
cell-eval run --help
```

### C. 术语表

- **ST (State Transition)**: 状态转换模型
- **HVG (Highly Variable Genes)**: 高变异基因
- **Perturbation**: 扰动（基因敲除、药物处理等）
- **Zero-shot**: 零样本学习（在未见过的细胞类型上测试）
- **Few-shot**: 少样本学习（使用少量样本训练）
- **AnnData**: Annotated Data Matrix，单细胞数据标准格式
- **TOML**: Tom's Obvious Minimal Language，配置文件格式

---

## 联系与支持

如有问题，请参考：

1. 项目 GitHub Issues
2. 内部文档：[`for_state/docs/important/`](../important/)
3. 代码示例：[`for_state/run__commands/tutorial/`](../../run__commands/tutorial/)

---

**版本**: 1.0  
**最后更新**: 2025-12-28
