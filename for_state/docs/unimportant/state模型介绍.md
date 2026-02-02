##

扰动类型：
基因敲除（Knockout， CRISPRko）基因敲减（Knockdown， CRISPRi）基因激活（Activation， CRISPRa）位点突变（Mutation，CRISPR base editing ）可以实现多位点突变多基因扰动（Combinatorial Perturbation）

扰动组技术可以被分为四种类型（不做讨论）
基于RNAseq的基于ATACseq的基于 Proteome probing的基于图像的

perturb-seq的数据资源：
1. PerturBase
2. scPerturb

E-distance在扰动量化中的作用：
在单细胞扰动实验中，E-distance主要用于以下方面： （1）量化扰动强度：E-distance可以衡量扰动前后细胞表达谱的变化程度，从而量化扰动的强度。 （2）比较不同扰动的效果：E-distance能够计算不同扰动条件下细胞表达谱之间的距离，从而比较不同扰动的效果。 （3）识别扰动目标与机制：如果两种扰动导致相似的E-distance变化，可能表明它们作用于相同的细胞通路或分子靶点。

单细胞扰动预测AI框架：STAMP 
STAMP是一种基于子任务分解进行单细胞扰动预测的创新AI范式，相较于大模型预训练-微调以及动态学习基因表征等策略，STAMP以一种插件的形式可以适配任意基因表征，具有高效、灵活、普适等优势，同时也为该领域的系统评估提供了可借鉴的新思路。

##

state

2.研究基础
   
细胞对扰动的响应是理解生物机制和筛选潜在药物靶点的关键。扰动类型包括基因干预（如CRISPR、RNAi）和化学处理（如小分子药物），这些扰动不仅用于诱导特定表型，还能揭示基因、通路与细胞表型之间的因果关系。

2.1一些有代表性的计算方法与AI模型:
scgen predicts single-cell perturbation responses. Nature methods, 2019

Predicting cellular responses to complex perturbations in high-throughput screens. Molecular systems biology,  2023

Learning single-cell perturbation responses using neural optimal transport. Nature methods, 2023

Predicting transcriptional outcomes of novel multigene perturbations with gears. Nature Biotechnology, 2024

scgpt: toward building a foundation model for single-cell multi-omics using generative ai. Nature Methods,  2024.

Large-scale foundation model on single-cell transcriptomics. Nature methods, 2024

2.2Benchmarking 对比不同方法的扰动预测性能
Benchmarking machine learning models for cellular perturbation analysis. arXiv , 2024

A largescale benchmark for network inference from single-cell perturbation data. arXiv , 2022

A systematic comparison of single-cell perturbation response prediction models. bioRxiv, 2024

Benchmarking single-cell foundation models for perturbation effect prediction. bioRxiv, 2024

2.3 两个噪音
生物异质性：研究人群中内部存在的异质性，使得真正的扰动差异被掩盖， 单细胞RNA测序破坏了细胞，无法直接观察扰动前的状态，导致扰动效果需要通过群体比较推断

技术噪声： 实验条件（如测序深度、试剂批次）的差异会掩盖真实的扰动信号，限制模型的泛化能力。

3.STATE的架构与训练

SE 模块训练不使用扰动数据,而是使用1.67亿单细胞转录组观察数据.
ST 使用高变基因的log1P表达值进行训练，对SE输出的细胞细胞添加扰动标签。ST 目标在于预测未见过的细胞类型对已知扰动的反应。

3.1 SE模块
核心目标是压缩细胞的表达维度（每个细胞捕捉2048个基因），捕获细胞状态的核心特征信息.
通过SE 这种训练模式，可以为ST提供噪声更少，生物学意义更强的输入，并且由于在跨数据集上进行训练，降低了不同平台的批次问题。

3.2 ST模块
负责预测细胞群体在扰动下的转录组动态变化.
过结合"集合注意力机制"和"分布匹配优化"，实现了对单细胞异质性和扰动效应的建模。

4.Cell-Eval评估框架
超越了传统的、基于原始表达计数的简单指标（如均方误差、相关系数），引入了一套更贴近生物学意义、更具解释性的指标，重点关注模型在"差异表达基因（DEG）预测"和"扰动效应强度估计"方面的能力。

4.1基因表达水平 (Gene Expression Counts)

    评估内容： 预测的单个基因在扰动后表达水平变化的准确性。

   - 扰动判别分数 (Perturbation Discrimination Score):对于一个给定的预测扰动效应，计算它与所有真实扰动效应的曼哈顿距离

   - 皮尔逊Δ相关系数 (Pearson Delta Correlation - PACorr):计算预测的扰动效应与真实效应之间的皮尔逊相关系数

4.2差异表达分析 (Differential Expression - DE)

    评估内容： 预测模型识别哪些基因因扰动而显著差异表达（DEGs），以及预测这些基因表达变化方向（上/下调）和幅度（log fold change）的准确性。

   - AUPRC (Area Under the Precision-Recall Curve):使用模型预测的基因 p 值（或 -log10(adj p-value)）作为分数，真实 DEGs（adj p-value < 0.05）作为正例，绘制精确率-召回率曲线（PR曲线），计算曲线下面积。

   - log fold change (logFC) Spearman 相关系数:针对真实显著 的 DEGs，计算预测的 logFC 与真实的 logFC 之间的 Spearman 秩相关系数
  
   - DEG 重叠准确率 (DE Overlap Accuracy):对每个扰动，分别取预测和真实数据中 top-k 个最显著的 DEGs，计算它们的交集大小占 k 的比例
  
   - Top-k 精确率 (Precision at k):对每个扰动，计算模型预测的 top-k 个 DEGs 中，有多少个是真实显著 的 DEGs 的比例
  
   - 方向一致性 (Directionality Agreement - DirAgree):对预测和真实都显著的基因，计算预测的 FC的正负与真实符号一致的基因所占的比例

4.3扰动效应大小(Perturbation Effect Size)

    评估内容： 模型预测整个转录组受扰动影响的整体强度或广度的准确性（例如，某个药物是引起全局重编程还是只影响少数基因）。

   - 效应大小 Spearman 相关系数 (SizeCorr):计算预测的扰动效应大小与真实的效应大小之间的 Spearman 秩相关系数