# Benchmarking mlx-graphs, PyG and DGL on training loops
Here we benchmark mlx-graphs against PyG and DGL in terms of training speed over some datasets.

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


### Saving results
Create a new file in `results` with file name
```
<processor_model>_<# performance CPU cores>P_<# efficiency CPU cores>E_<# GPU cores>GPU_<GB or RAM>G.md
```
For example, for an M3Pro with 5 performance and 6 efficiency CPU cores, 14 GPU cores and 18GB of RAM the file name will be `M3PRO_5P_6E_14GPU_18G.md`.

Then paste the results in the new file.
