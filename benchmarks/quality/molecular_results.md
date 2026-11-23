# Training quality: molecular models

mlx-graphs (MLX, GPU) vs PyTorch Geometric (torch, CPU), trained by
`molecular_quality.py` with the same architectures, hyperparameters,
data and mini-batch order. Test metric at the best validation epoch,
mean ± std over seeds; time per epoch is the median training time of
an epoch (the first one, with warm-up, excluded). PyG runs without
pyg-lib / torch-cluster / torch-sparse (pure-torch radius graph and
triplets). Peak memory: `mx.get_peak_memory()` with MLX's buffer
cache bounded to 1 GB. Shared machine: the times vary by ~30%.

| model | dataset | metric | PyG | mlx-graphs | PyG s/epoch | mlx-graphs s/epoch | mlx-graphs peak memory (GB) | seeds |
|---|---|---|---|---|---|---|---|---|
| schnet | QM9 (gap) 10000/1000/1000, 10 epochs | test MAE (eV) | 0.3771 ± 0.0039 | 0.3840 ± 0.0191 | 6.4 | 3.2 | 0.25 | 2/2 |
| dimenet++ | QM9 (gap) 5000/1000/1000, 6 epochs | test MAE (eV) | 0.2699 | 0.2787 | 16.4 | 9.5 | 1.38 | 1/1 |
| attentivefp | ESOL, 100 epochs | test RMSE (log mol/L) | 0.6064 ± 0.0575 | 0.5926 ± 0.0338 | 0.5 | 0.2 | 0.16 | 3/3 |
