# Graph Signal Separation with Learnable Spectral Filters (GSS-LSF)

[![Paper](https://img.shields.io/badge/Paper-IEEE%20SPL-blue)](https://doi.org/10.1109/LSP.2026.3705721)
[![Preprint](https://img.shields.io/badge/Preprint-arXiv-b31b1b)](https://arxiv.org/abs/2604.24185)
[![Institution](https://img.shields.io/badge/EPFL-IMOS-red)](https://imos.epfl.ch/)

**Keivan Faghih Niresi, Dorina Thanou, and Olga Fink**  
*IEEE Signal Processing Letters*, 2026.

This repository accompanies the paper **“Graph Signal Separation With Learnable Spectral Filters”**, which studies recovery of latent graph signals from a single noisy mixture by using source-specific graphs and unsupervised learnable spectral filtering.

> **Implementation status:** This release implements the paper's **fixed-random-input learnable spectral filter** trained with Adam and includes a closed-form truncated graph Fourier least-squares.


## Contents

```text
.
├── Demo.ipynb                 # Real-world data comparison
├── sensor_data.pt             # Small paired signals used by the illustration
├── spectral_lsf/
│   ├── __init__.py
│   ├── spectral.py             # Graphs, learnable spectral gains, LS separation, oracle sweep
│   └── plotting.py             # Visualization utilities
├── requirements.txt
├── CITATION.cff
└── README.md
```

## Setup

Python 3.10+ is recommended. In a new environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter lab
```

Open `Demo.ipynb` and run its cells from top to bottom. The notebook uses the included `sensor_data.pt`.

## Minimal Python example

```python
import torch
from spectral_lsf import build_graph, closed_form_spectral, make_noisy_mixture, seed_everything

seed_everything(42)
data = torch.load("sensor_data.pt", weights_only=True)
x_temperature = data["x_temperature"]
x_pressure = data["x_pressure"]

mixed, _ = make_noisy_mixture(x_temperature, x_pressure, noise_std=0.2)
L_temperature = build_graph(x_temperature, k=5)
L_pressure = build_graph(x_pressure, k=5)

# GSS-LSF (paper's learnable spectral-filter parameterization)
from spectral_lsf import learnable_spectral_filters
result = learnable_spectral_filters(L_temperature, L_pressure, mixed, k1=5, k2=5)
x_temperature_hat, x_pressure_hat = result.temperature, result.pressure

# Closed-form truncated-basis baseline for GSS-LSF
baseline_temperature, baseline_pressure = closed_form_spectral(
    L_temperature, L_pressure, mixed, k1=5, k2=5
)
```

## Citation

If you use or build on this research, please cite:

```bibtex
@article{niresi2026graph,
  title   = {Graph Signal Separation With Learnable Spectral Filters},
  author  = {Faghih Niresi, Keivan and Thanou, Dorina and Fink, Olga},
  journal = {IEEE Signal Processing Letters},
  year    = {2026},
  doi     = {10.1109/LSP.2026.3705721}
}
```
