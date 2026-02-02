# MMD 脚本说明（for_state/scripts/MMD.py）

本文档对仓库中 `for_state/scripts/MMD.py` 的功能、实现细节、数据流、依赖、潜在问题与改进建议做详尽说明，便于阅读、维护与复现。

**高层目的与适用场景**
- 目的：实现基于 MMD（Maximum Mean Discrepancy，使用 RBF kernel）的方法，将一个预训练模型的“源域 control”分布与目标（H1）control 对齐。对齐通过学习一个简单的恒定表达 shift（对每个基因的加法偏移）来完成。最后用适配后的 control 生成对目标 perturbations 的预测，并保存结果。
- 适用场景：已有预训练模型（包括模型目录、config、var_dims.pkl、pert_onehot_map.pt 和 checkpoint）；有源域与目标的 AnnData（`.h5ad`）数据，且 obs 中有 `target_gene` 指示 control（值为 `'non-targeting'`）。
- 运行入口：脚本顶部类 `Cfg` 包含默认路径与超参；主流程在 `main()`，会依次加载模型、提取控制（control）embedding、微调 shift adapter、保存 adapter、并对目标 perturbations 做预测输出 CSV。

**主要函数逐一说明**
- `pairwise_distances_sq(x, y=None)`
  - 角色：计算样本集合间的平方欧氏距离矩阵 d^2。
  - 输入：`x: Tensor [n, d]`，可选 `y: Tensor [m, d]`。
  - 输出：`d2: Tensor [n, m]`，用 `clamp(min=0.0)` 避免数值负值。
  - 依赖：矩阵乘法 `torch.mm`

- `rbf_kernel(x, y=None, sigma=1.0)`
  - 角色：基于平方距离计算 RBF kernel 矩阵 K = exp(-d2 / (2*sigma^2))。
  - 输入/输出：参见上。
  - 敏感点：`sigma` 取值对 kernel 数值行为影响大（过小导致近似身份矩阵，过大导致近似常数矩阵）。

- `mmd_rbf(x, y, sigma=1.0)`
  - 角色：使用 RBF kernel 计算两个样本集合之间的 MMD 值（近似无偏或改良式）。
  - 计算要点：构造 Kxx, Kyy, Kxy，去掉对角线项（若 n>1/m>1），然后按 (n*(n-1)) 等归一化并组合为 scalar。
  - 边界条件：处理 n<1 或 m<1 的情形并返回 0；除数处加 `1e-8` 以避免除零。
  - 注意：对小样本（n=1 或 m=1）数值稳定性较差，且该实现为单尺度核（单个 sigma）。

- `median_sigma(emb, subsample=1000, fallback=1.0)`
  - 角色：用 median heuristic（经验中值）估计 RBF 的 sigma。随机抽样最多 `subsample` 个点计算成对距离的 median 的平方根。
  - 随机性：使用 `torch.randperm`，未设随机种子。
  - 边界：若无有效距离则返回 `fallback`。

- `load_model_and_metadata(cfg: Cfg)`
  - 角色：读取模型配置信息并实例化并加载预训练 checkpoint，返回 model、var_dims、gene_names、pert_map。
  - 关键文件：`config.yaml`（OmegaConf）、`var_dims.pkl`（pickle，需包含 `gene_names`）、`pert_onehot_map.pt`（torch 保存的 dict-like）。
  - 依赖仓库代码：`from src.state.tx.utils import get_lightning_module`，调用 `get_lightning_module` 构造 Lightning 模块并通过 checkpoint 的 `state_dict` 加载参数（`strict=False`）。
  - 运行后副作用：把 model 移动到 `cfg.device` 并调用 `model.eval()`，同时把参数 `requires_grad=False`（冻结）。

- `get_control_embeddings(adata_path, model, cfg: Cfg, sample_limit=None)`
  - 角色：从单个 `.h5ad` 中筛选 control（`obs['target_gene']=='non-targeting'`），从表达矩阵 X 计算 model 的 basal encoder embedding 并返回。（注意返回值为 embedding）
  - 输入/输出：输入路径、model、配置；输出 `emb: Tensor`（CPU tensor，detached）。
  - 边界条件：若缺少 `target_gene` 或没有 control 细胞则抛出 RuntimeError。
  - 注意：对稀疏矩阵尝试 `.toarray()`，若数据非常大会占用大量内存。

- `get_control_embeddings_multi(adata_paths, model, cfg: Cfg, sample_limit=None)`
  - 角色：支持多个 h5ad 文件（脚本支持一种基于大括号的简单展开或直接 glob ），为每个文件提取 control embeddings 并合并返回。
  - 实现细节：若路径中有 `{}`，脚本先尝试 `glob.glob(path.replace('{','').replace('}',''))`，若无匹配则手工按大括号内逗号分割拼接。对每个文件若没有 control，会 `WARNING` 并跳过。若最终无任何 embeddings 则抛错。
  - 输出：合并后的 embeddings（CPU），可选按 `sample_limit` 随机截断。

- `finetune_shift(cfg: Cfg, model, src_embs, tgt_X)`
  - 角色：学习一个长度为表达维度的可训练偏移向量 `shift`（nn.Parameter），优化目标为将加上 shift 的目标原始表达（或输入）通过 model.basal_encoder 的 embedding 与源域 embedding 在 MMD（RBF）上对齐。
  - 训练细节：optimizer = Adam([shift], lr=cfg.adapter_lr)，迭代 `cfg.adapter_epochs`，mini-batch 从 `src_embs` 随机抽样并从 `tgt_X` 随机抽取同样大小样本，计算 fake_emb = model.basal_encoder(tgt_batch) 并最小化 `cfg.weight_mmd * mmd_rbf(fake_emb, src_batch, sigma)`。
  - 输出：返回一个 `ShiftAdapter` 对象（持有训练后 shift 的 CPU 版本，提供 `__call__`、`to(device)`、`state_dict()`）。
  - 注意：训练中对 `sigma` 使用 `median_sigma(src_embs)`；若 `src_embs` 很大或在 CPU/GPU 频繁移动，会有性能开销。

- `predict(cfg: Cfg, model, adapter, pert_map, var_dims)`
  - 角色：对 `cfg.target_pert_csv` 指定的 perturbation 列表，使用目标 control（应用 adapter 后取均值为 control template），结合 `pert_map` 中对应的 perturbation embedding，调用 `model.predict_step` 得到每个 perturbation 的细胞级预测并对若干虚拟细胞取平均，最终返回 `pandas.DataFrame`：每行对应一个 `target_gene`，列为基因名（来自 `var_dims['gene_names']`）。
  - 输入：支持 `target_pert_csv` 为 CSV（需 `target_gene` 列）或 `.h5ad`（需在 obs 中含 `target_gene`）。
  - 模型接口假设：`model.predict_step(batch, batch_idx=0, padded=False)` 返回 dict，优先使用 `out.get('pert_cell_counts_preds')` 或 `out['preds']` 作为预测张量。
  - 细节：对每个 perturbation 用 `cell_sentence_len`（若有）分块生成 batch 直到累积 `cfg.n_cells_per_pert` 个伪细胞，最终取平均。

- `main()`
  - 角色：脚本默认流程，使用 `Cfg()` 的默认路径：加载模型/元数据，提取源域 control embeddings（`get_control_embeddings_multi`），提取目标 control（`get_control_embeddings` 被用于 `tgt_X`），训练 adapter 保存为 `cfg.save_adapter_path`，调用 `predict` 并把结果写到硬编码输出路径 `/data/h1_predicted_perturbations.csv`。

**数据流（概览）**
1. 配置：`Cfg`（包含模型路径、数据路径、超参如学习率、batch、epoch、sigma 初始值等）。
2. 加载模型与元数据（`load_model_and_metadata`）：读取 `config.yaml`、`var_dims.pkl`、`pert_onehot_map.pt`，并通过仓库内模型工厂构造模型并加载 checkpoint。模型被移到 `cfg.device` 且置为 eval，参数冻结。
3. 提取源域 control embeddings（`get_control_embeddings_multi`）：读取多个 h5ad 文件中 `obs['target_gene']=='non-targeting'` 的样本，过 model.basal_encoder 得到 embedding（合并与采样）。
4. 准备目标 control（`get_control_embeddings`）：脚本当前实现对目标 control 也调用该函数，得到的返回值为 embedding（见“语义矛盾”节）。
5. 训练 shift（`finetune_shift`）：使用源域 embedding 与目标（当前脚本传入的 `tgt_X`）训练 `shift`，通过将目标表达加上 shift 并经 `model.basal_encoder` 来计算 fake_emb，与 src_embs 计算 MMD 并反向更新 shift。
6. 保存 adapter 并进行预测（`predict`）：在预测流程中，从目标 control 的原始表达读取 X（脚本在 `predict` 内直接读取 adata 并取 X，随后做 `X_shift = adapter(X)`），把平均 control embedding 与每个 perturbation 的 perturbation embedding（来自 `pert_onehot_map.pt`）传入 `model.predict_step` 来得到预测并计算均值，输出 CSV。

**关键依赖文件与格式要求**
- 必备文件与其预期结构：
  - `config.yaml`：用于构造模型（OmegaConf）。
  - `var_dims.pkl`（pickle）：至少包含 `gene_names`（模型输出列名）。
  - `pert_onehot_map.pt`（torch 保存）：dict-like，key 为 perturbation 名（字符串），value 为与模型兼容的 perturbation embedding tensor。
  - checkpoint（`cfg.model_ckpt`）：包含模型 `state_dict`（或直接 state_dict）用于加载权重。
  - 源 / 目标数据：AnnData `.h5ad` 文件，且 `adata.obs` 中要有 `target_gene` 字段；以 `'non-targeting'` 标记 control 细胞；表达矩阵可能为稀疏矩阵（脚本尝试 `.toarray()`）。
  - 目标 perturbation 列表：CSV 文件（含列 `target_gene`），或 `.h5ad`（obs 中含 `target_gene`）。

- 外部 Python 包依赖：`torch`、`torch.nn`、`torch.optim`、`numpy`、`pickle`、`pandas`、`scanpy`、`omegaconf`、`math`、`glob`。
- 仓库内部依赖：`src.state.tx.utils.get_lightning_module`（用于实例化模型），模型实现需提供 `basal_encoder`、`predict_step`、可选 `cell_sentence_len` 等接口。

**潜在问题、边界情形与数值/性能注意点**
1. 语义矛盾（重要）：
   - `get_control_embeddings` 返回 embedding（通过 `model.basal_encoder(X)`），但是 `finetune_shift` 的实现期望 `tgt_X` 是原始表达（脚本将 `tgt_X` 直接用于 `tgt_batch = tgt_X[tgt_idx] + shift`，然后 `fake_emb = model.basal_encoder(tgt_batch)`）。如果 `tgt_X` 已经是 embedding，则对 embedding 加上 shift 并再编码会产生语义错误（double encoding）。需确认设计预期：
     - 方案 A（必要时推荐）：`get_control_embeddings` 分离为 `get_control_raw_X`（返回原始 X）与 `get_control_embeddings`（返回 embedding）；并确保 `finetune_shift` 接收 raw X。
     - 当前脚本看似把源域当 embedding（src_embs）与目标 raw X（tgt_X）做对齐，但实际代码中 `tgt_X` 被赋值为 embedding（在 main 中 `tgt_X = get_control_embeddings(...)`），这会引发训练目标混淆。
2. 内存与稀疏矩阵：对稀疏 `adata.X` 使用 `.toarray()` 会把整个矩阵展开到内存，易 OOM，尤其在大数据上。应考虑分批读取或支持稀疏计算流。
3. CPU/GPU 拷贝开销：代码在多个地方将 tensors 在 CPU/GPU 间移动（例如在 get_control_embeddings 返回 CPU 的 embedding），随后在训练中 `src_embs.to(device)` 再转回 GPU。频繁移动影响性能。建议：保留设备一致性参数或返回原始设备并按需要进行转换。
4. 路径与大括号展开脆弱：`get_control_embeddings_multi` 中对 `{a,b}` 的展开逻辑先尝试 `glob.glob`，再手工分解拼接，处理不如标准 shell brace expansion 或 `glob` 的通配能力强，可能导致路径解析错误。建议改为直接接受 list 或使用 `glob.glob` 与通配符 `*`。
5. 随机性与可复现性：脚本使用 `torch.randperm`、`torch.randint`、`torch.randperm` 等但未设置随机种子，结果不可复现。建议添加 `seed` 参数并在运行开始设置 `torch.manual_seed(seed)`、`np.random.seed(seed)` 等。
6. MMD 数值稳定性：`median_sigma` 可能返回很小的 sigma（或 0），导致 RBF kernel 数值退化；脚本在 MMD 归一化时对除数加 `1e-8`，但更稳健的做法是对 sigma 做下界 clamp（如 `max(sigma, 1e-6)`）。同时单尺度 kernel 有局限，可考虑多尺度核加权。
7. 小样本问题：当样本数 n 或 m 非常小时（例如 1 或 2），MMD 估计方差大，训练不稳定，应发出警告或跳过不可靠更新。
8. 模型 API 假设：`predict_step` 返回的键名（`pert_cell_counts_preds` 或 `preds`）可能因模型改变而不同，需确保模型对该输入字典与输出键名兼容。

**改进建议（优先级排序，短小可行）**
1. 解决 raw X vs embedding 的语义矛盾：把数据读取函数拆分为 `get_control_raw_X` 与 `get_control_embeddings`，并在 `main` 中明确传入 `tgt_X_raw` 给 `finetune_shift`。
2. 增加 `seed` 参数（cfg）以保证训练与采样可复现。
3. 改进 `median_sigma`：对返回值 clamp 至一个下界（如 1e-6 或 1e-3），并当 subsample 样本不足时退化到 `fallback` 并打印警告。
4. 优化内存/性能：对大 AnnData 使用分批读取或在计算 embedding 时直接在设备上分批编码，避免 .toarray() 全量展开。
5. 改善路径处理：删除自制的大括号拼接，改为接受明确的文件列表或使用 `glob` + 通配符。并对不存在的文件路径做更友好提示。

**建议的最小测试用例**
- 使用一个极简 mock model（实现 `basal_encoder` 返回输入的线性变换，`predict_step` 返回 `{'preds': input}` 的最简单版本），并用一个小的虚拟 AnnData（例如 10 个细胞、5 个基因，稀疏或密集皆可），测试整个 `finetune_shift` + `predict` 流程的端到端行为，验证形状与保存文件是否正确。

**进一步问题（最多五个）**
1. 你期望 `finetune_shift` 接收的 `tgt_X` 是原始表达（raw X）还是 embedding？当前实现存在不一致，请确认首选行为。  
2. 是否需要我把这份说明直接写入仓库（我已经生成文件），或者你还需我同时提交一个注释版脚本/单元测试？  
3. 你是否希望结果可复现（我可以添加 `seed` 到 `Cfg` 并统一设置随机种子）？  
4. 目标环境是否有 GPU，并希望训练 adapter 时保持 tensors 全程在 GPU 上以提高性能？  
5. 你期望支持大规模稀疏 `.h5ad` 文件的内存友好处理吗（需要做分批/流式读取）？

---

文件已生成：
- [for_state/docs/unimportant/MMD说明.md](for_state/docs/unimportant/MMD说明.md)

你希望我接下来做哪一项：同步修改脚本以修正 raw X vs embedding 的矛盾，或添加可复现性（seed）与改进的 median_sigma 实现？