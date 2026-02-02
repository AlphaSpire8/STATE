# state/for_state/run_commands/testing/prepare_test_data.py
import anndata as ad
import pickle

def prepare_test_data(data_in, data_out, var_dims_path):
    """
    准备单细胞测试数据：筛选高变基因并进行标准化处理
    
    Args:
        data_in: 输入数据路径 (.h5ad)
        data_out: 输出数据路径 (.h5ad)
        var_dims_path: 模型高变基因列表文件路径 (var_dims.pkl)
    """
    # 1. 加载测试数据
    print(f"加载测试数据: {data_in}")
    adata = ad.read_h5ad(data_in)
    print(f"原始数据维度: {adata.shape}")

    # 2. 加载高变基因列表
    with open(var_dims_path, 'rb') as f:
        hvg_names = pickle.load(f)['gene_names']
    print(f"模型高变基因数量: {len(hvg_names)}")

    # 3. 筛选测试集中存在的高变基因
    valid_genes = [g for g in hvg_names if g in adata.var_names]
    print(f"匹配基因数: {len(valid_genes)}/{len(hvg_names)}")
    adata = adata[:, valid_genes].copy()

    # 4. 使用Scanpy进行标准化 (替换原第五步)
    # import scanpy as sc  
    # sc.pp.normalize_total(adata, target_sum=10000)
    # print(f"标准化后（总和10000）- 最大值: {adata.X.max():.4f}")

    # 5. 进行log1p变换 
    # sc.pp.log1p(adata)
    # print(f"log1p后- 最大值: {adata.X.max():.4f}")

    # 6. 删除所有obsm字段（包括旧X_hvg）
    adata.obsm.clear()
    print("已删除所有obsm字段")
    
    # # 7. 创建新obsm字段X_hvg
    # adata.obsm['X_hvg'] = adata.X.copy()

    # 7. 创建新 obsm 字段 X_hvg（确保为稠密 ndarray）
    print("转换 X 为稠密数组...")
    if hasattr(adata.X, "toarray"):
        X_dense = adata.X.toarray()
    else:
        X_dense = np.array(adata.X)  # 兜底转换
    adata.obsm['X_hvg'] = X_dense

    print(f"X_hvg 类型: {type(adata.obsm['X_hvg'])}")
    print(f"X_hvg 形状: {adata.obsm['X_hvg'].shape}")

    # 8. 保存数据
    print(f"保存测试数据: {data_out}")
    adata.write_h5ad(data_out)

if __name__ == '__main__':
    prepare_test_data(
    data_in="/data1/fanpeishan/State-Tahoe-Filtered/c36.h5ad",
    data_out="/data3/fanpeishan/state/for_state/data/State-Tahoe-Filtered-processed/c36_prep.h5ad",
    var_dims_path="/data3/fanpeishan/state/for_state/models/ST-Tahoe/var_dims.pkl"
)
    prepare_test_data(
    data_in="/data1/fanpeishan/State-Tahoe-Filtered/c39.h5ad",
    data_out="/data3/fanpeishan/state/for_state/data/State-Tahoe-Filtered-processed/c39_prep.h5ad",
    var_dims_path="/data3/fanpeishan/state/for_state/models/ST-Tahoe/var_dims.pkl"
)