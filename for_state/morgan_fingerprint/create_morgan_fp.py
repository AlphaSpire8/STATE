import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import DataStructs
# 引入新的指纹生成器模块
from rdkit.Chem import rdFingerprintGenerator 
import numpy as np
import os
import re

# ================= 配置路径 =================
INPUT_CSV = '/data3/fanpeishan/state/for_state/morgan_fingerprint/merged_drug_names_preview.csv'
OUTPUT_DIR = '/data3/fanpeishan/state/for_state/morgan_fingerprint/'
OUTPUT_FILENAME = 'drug_morgan_fingerprints.csv'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# ================= 核心处理函数 =================
def get_morgan_fingerprint(drug_name, n_bits=256, radius=2):
    """
    输入药物名称，返回：
    1. 256维numpy数组 (特征向量)
    2. 状态 ("Success" / "Fail")
    3. 实际查询成功的名称 (如果是失败则为 None)
    """
    # 初始化全0向量
    zero_fp = np.zeros((n_bits,), dtype=np.int8)
    
    if pd.isna(drug_name) or str(drug_name).strip() == "":
        return zero_fp, "Fail", None

    # --- 1. 准备搜索列表 (原名 + 去括号名) ---
    search_names = [str(drug_name)]
    if "(" in str(drug_name):
        clean_name = re.sub(r'\s*\(.*?\)', '', str(drug_name)).strip()
        if clean_name:
            search_names.append(clean_name)
    
    smiles = None
    found_name = None

    # --- 2. 循环尝试查询 ---
    for name_query in search_names:
        try:
            comps = pcp.get_compounds(name_query, 'name')
            if comps:
                # 保持之前的修改：使用 connectivity_smiles
                smiles = comps[0].connectivity_smiles
                if smiles:
                    found_name = name_query
                    break 
        except Exception:
            continue 
            
    if not smiles:
        return zero_fp, "Fail", None

    # --- 3. 构建分子并生成指纹 ---
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return zero_fp, "Fail", found_name 

    try:
        # 【修改点：使用新版生成器消除警告】
        # 使用 rdFingerprintGenerator 替代 AllChem.GetMorganFingerprintAsBitVect
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp_bitvect = morgan_gen.GetFingerprint(mol)
        
        # 转换为 Numpy 数组
        fp_array = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp_bitvect, fp_array)
        return fp_array, "Success", found_name
    except Exception as e:
        # 为了调试方便，这里可以打印一下具体的错误，但为了保持输出整洁暂时不打印
        return zero_fp, "Fail", found_name

# ================= 主程序逻辑 =================
def main():
    sum_success = 0
    sum_name_changed = 0
    sum_unfound = 0
    print(f"正在读取输入文件: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    col_name = 'value'
    if col_name not in df.columns:
        col_name = df.columns[0]
        print(f"警告: 未找到 'value' 列，默认使用第一列 '{col_name}'")

    drug_names = df[col_name].values
    print(f"共加载 {len(drug_names)} 个药物名称。开始处理...\n")

    fp_list = []
    
    # 循环处理
    for i, name in enumerate(drug_names):
        fp, status, found_name = get_morgan_fingerprint(name)
        fp_list.append(fp)
        
        # 终端展示查询结果
        if status == "Success":
            sum_success += 1
            if found_name != name:
                sum_name_changed += 1
                print("name changed")
            print(f"[{i+1}/{len(drug_names)}] 原始名: {name} -> 查询结果: {found_name}")
        else:
            sum_unfound += 1
            print("unfound")
            print(f"[{i+1}/{len(drug_names)}] 原始名: {name} -> 未查询到")

    # --- 结果整合与保存 ---
    fp_df = pd.DataFrame(fp_list, columns=[f'fp_{i}' for i in range(256)])
    out_df = pd.concat([df[[col_name]].reset_index(drop=True), fp_df], axis=1)

    print(f"\n处理完成！共查询成功 {sum_success} 个药物。")
    print(f"其中 {sum_name_changed} 个名称发生了变化。")
    print(f"未查询到 {sum_unfound} 个药物。")
    out_df.to_csv(OUTPUT_PATH, index=False)
    print("保存完毕。")

if __name__ == "__main__":
    main()