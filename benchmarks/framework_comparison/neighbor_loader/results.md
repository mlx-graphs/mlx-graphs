# NeighborLoader Benchmark Results

Platform: macOS-15.6.1

mlx version: 0.30.6

mlx-graphs version: 0.0.8

mlx-cluster version: 0.0.7

torch version: 2.8.0

torch_geometric version: 2.7.0

## Configuration

- Number of batches per timing run: 50
- Timing repeats (min taken): 5

## Results


| Config | Graph Size | Batch Size | Neighbors | MLX-Graphs (s) | PyG (s) | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| Small (Cora-sized) | 2,708 nodes | 256 | [10, 5] | 0.0063 | 0.0029 | 0.45x |
| Medium | 10,000 nodes | 512 | [15, 10, 5] | 0.0624 | 0.0391 | 0.63x |
| Large | 100,000 nodes | 1024 | [25, 10] | 0.4274 | 0.2407 | 0.56x |


## Notes

- Speedup > 1.0x means MLX-Graphs is faster than PyG
- Times are minimum of 5 runs for 50 batch iterations
- MLX uses GPU, PyG uses CPU (Apple Silicon does not support CUDA)
