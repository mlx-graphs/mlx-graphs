# Benchmarking mlx-graphs vs torch-geometric on a number of ops

Benchmarks are generated by measuring the runtime of some `mlx-graphs` ops/layers, along with their equivalent in [torch_geometric](https://github.com/pyg-team/pytorch_geometric) (PyG) with `mps` and `cpu` backends. For each layer, we measure the runtime of multiple experiments.

For each op we compute 2 benchmarks based on these experiments:

* Detailed benchmark: provides the runtime of each experiment.
* Average runtime benchmark: computes the mean of experiments. Easier to navigate, with fewer details.

A comprehensive benchmark of core `mlx` operations vs `torch` operations is also available [here]((https://github.com/TristanBilot/mlx-benchmark)).

Results are stored in `results` and are provided for different processors.


## How to contribute
### Running experiments
Install benchmark dependenceis
```
pip install '.[benchmarks]'
```
Run
```
python launcher.py
```

> [!TIP]
> Since the mps backend is not optimized, feel free to run the experiments with `--include-mps=False`.

> [!WARNING]
> Running all the experiments takes about 20 minutes on an M3Pro and it's pretty compute intensive.

### Saving results
Create a new file in `results` with file name
```
<processor_model>_<# performance CPU cores>P_<# efficiency CPU cores>E_<# GPU cores>GPU_<GB or RAM>G.md
```
For example, for an M3Pro with 5 performance and 6 efficiency CPU cores, 14 GPU cores and 18GB of RAM the file name will be `M3PRO_5P_6E_14GPU_18G.md`.

Then paste the results in the new file.
