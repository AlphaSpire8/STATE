import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import sys

# =============== 第一步：安全获取SMILES ===============
try:
    comps = pcp.get_compounds("aspirin", 'name') 
    if not comps:
        raise ValueError("PubChem未找到匹配药物")
    
    smiles = comps[0].connectivity_smiles  
    if not smiles:
        raise ValueError("获取的SMILES为空")
    print(f"✓ SMILES: {smiles}")
except Exception as e:
    sys.exit(f"✗ PubChem查询失败: {e}. ")

# =============== 第二步：安全构建分子 ===============
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    sys.exit(f"✗ RDKit无法解析SMILES: {smiles}. 建议验证SMILES有效性")

# =============== 第三步：生成256维Morgan指纹 ===============
try:
    fp_bitvect = AllChem.GetMorganFingerprintAsBitVect(
        mol, 
        radius=2,      
        nBits=256     
    )
except Exception as e:
    sys.exit(f"✗ 指纹生成失败: {e}")

# =============== 第四步：安全转换为numpy数组 ===============
fp_array = np.zeros((256,), dtype=np.int8)  # 与nBits严格匹配
DataStructs.ConvertToNumpyArray(fp_bitvect, fp_array)

# =============== 验证输出 ===============
print(f"指纹维度: {fp_array.shape}")
print(f"非零位: {np.sum(fp_array)} ({np.sum(fp_array)/256:.1%})")
print(f"二值验证: 唯一值={np.unique(fp_array)}")  # 应输出 [0 1]
