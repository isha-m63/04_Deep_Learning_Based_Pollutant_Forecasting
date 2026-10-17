# Setup

## Requirements
- Python 3.10+ (Docker image uses 3.11)
- `torch`, `torchdiffeq`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `seaborn`, `pyyaml`, `streamlit`
- The demo's pinned versions are in `requirements-demo.txt`. PyTorch is installed separately there.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchdiffeq pandas numpy matplotlib scikit-learn seaborn pyyaml streamlit
```

## Configuration

`config.yaml` holds the paths and hyperparameters. **All paths are absolute Windows paths**
(`C:/02_AtmPhy_ML/S4_AQ_PINN/...`) from the original machine. Change them to paths in your checkout, for example
`data/processed/`, before running notebooks or scripts. Hyperparameters:

| Key | Value |
|---|---|
| `training.batch_size / epochs / learning_rate` | 32 / 50 / 0.001 |
| `model.lstm` | input 6, hidden 64, 2 layers, dropout 0.2, output 3 |
| `model.neural_ode` | input 6, hidden 64, intermediate 64, output 3 |
| `model.pinn` | input 6, hidden 64, 2 layers, output 3 |

## Reproduce the pipeline

The notebooks are the working path (see [known-issues.md](known-issues.md) for why the scripts' `__main__` blocks
don't run as written):

1. `notebooks/data_cleaning_eda.ipynb` → `data/processed/cleaned_data.csv`
2. `notebooks/preprocessing.ipynb` → `data/processed/{X,y}_{train,test}_norm.csv`
3. `notebooks/model_training_lstm.ipynb`, `..._neural_ode.ipynb`, `..._pinn.ipynb` → `weights/*.pth`, loss curves,
   metrics and plots

The training functions can also be imported and called directly:

```python
from utils.data_loader import get_data_loaders
from training.train_lstm import train_lstm
train_loader, test_loader = get_data_loaders(config, batch_size=32)
train_lstm(train_loader, test_loader, input_size=6, output_size=3, device=device,
           hidden_size=64, num_layers=2, dropout=0.2, epochs=50, lr=1e-3,
           save_path="weights/trained_lstm_model.pth", losses_save_path="evaluation/lstm/")
```

`train_neural_ode` takes `hidden_size` and `intermediate_size`; `train_pinn` takes `hidden_size`, `num_layers`,
`lambda_phys` (default 1.0) and `delta_t` (default 1.0).

## Run the demo without retraining

```bash
streamlit run streamlit_app.py
```

It uses the files in `demo_assets/` only.
