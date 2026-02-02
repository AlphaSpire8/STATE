请根据以下要求，调整代码：
1.去掉tqdm进度条
2.在终端展示每个药物的被查询到的时候的名称，如果没有被查询到，请显示"未查询到"
3.将“smiles = comps[0].canonical_smiles or comps[0].isomeric_smiles”替换为“smiles = comps[0].connectivity_smiles”