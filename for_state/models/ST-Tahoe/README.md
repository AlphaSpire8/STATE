State Transition model checkpoints are meant to be used with the [Arc State](https://pypi.org/project/arc-state/) package.

State is distributed via [`uv`](https://docs.astral.sh/uv). To train models on your own AnnDatas, please see the [repository](https://github.com/ArcInstitute/state) README.

## Running inference from pypi

```bash
uv tool install arc-state
```

To generate predictions:

```code
state tx infer --model_dir <ST-Tahoe_PATH> --pert_col drugname_drugconc --adata <INPUT_ADATA>.h5ad --output <OUTPUT_DIR>
```

This will group the cells in your input AnnData by the specified `pert_col` and run inference with a pretrained ST checkpoint.

## Running from source

```bash
# Clone repo
git clone github.com:arcinstitute/state
cd state

# Initialize venv
uv venv

# Install
uv tool install -e .
```

To generate embeddings given an AnnData:

```code
state tx infer --model_dir <ST-Tahoe_PATH> --pert_col drugname_drugconc --adata <INPUT_ADATA>.h5ad --output <OUTPUT_DIR>
```

For model licenses please see `MODEL_ACCEPTABLE_USE_POLICY.md`, `MODEL_LICENSE.md`, and `LICENSE.md`.