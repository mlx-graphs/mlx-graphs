# mlx-graphs vs PyTorch Geometric

Machine: Apple M1 Pro, 32 GB, macOS 27.0. mlx 0.32.4, mlx-graphs 0.1.0, mlx-cluster 0.1.2, torch 2.15.0, torch_geometric 2.8.1, Python 3.12.10.

Times are medians in milliseconds; lower is better. Speedup is the PyG time divided by the mlx-graphs time.

mlx-graphs trains GNNs **6.0× faster than PyG on the Mac's GPU (MPS)** and **3.5× faster than PyG on the CPU** (median over 34 training benchmarks, up to 125×), and reaches the same accuracy.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/headline_dark.png">
  <img alt="Time of PyG on MPS and on CPU relative to mlx-graphs for one layer, node classification and graph classification, and the test accuracy of each framework" src="figures/headline_light.png">
</picture>

## One layer, forward + backward

Random graphs, 128 input and output features, 4 heads for GAT/GATv2.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/layers_speedup_dark.png">
  <img alt="Forward and backward time per layer and graph size, relative to mlx-graphs, for mlx-graphs, PyG on MPS and PyG on CPU" src="figures/layers_speedup_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/layers_time_dark.png">
  <img alt="Time of one forward and backward pass per layer and graph size, for mlx-graphs, PyG on MPS and PyG on CPU" src="figures/layers_time_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/layers_memory_dark.png">
  <img alt="Peak memory per layer on the largest graph, for mlx-graphs, PyG on MPS and PyG on CPU" src="figures/layers_memory_light.png">
</picture>

| graph | layer | mlx-graphs | PyG MPS | PyG CPU | speedup vs MPS | speedup vs CPU | peak GB mlx / MPS / CPU |
|---|---|---:|---:|---:|---:|---:|---|
| 10,000 nodes, 100,000 edges | GCN | 1.56 | 30.93 | 12.94 | 19.8× | 8.3× | 0.1 / 1.3 / 0.4 |
| 10,000 nodes, 100,000 edges | GraphSAGE | 1.75 | 12.26 | 6.83 | 7.0× | 3.9× | 0.1 / 1.3 / 0.4 |
| 10,000 nodes, 100,000 edges | GIN | 2.19 | 12.63 | 7.66 | 5.8× | 3.5× | 0.1 / 1.3 / 0.4 |
| 10,000 nodes, 100,000 edges | GAT | 4.73 | 43.26 | 22.96 | 9.1× | 4.9× | 0.2 / 1.3 / 0.5 |
| 10,000 nodes, 100,000 edges | GATv2 | 12.47 | 59.54 | 35.96 | 4.8× | 2.9× | 0.6 / 1.3 / 0.6 |
| 50,000 nodes, 500,000 edges | GCN | 6.56 | 154.28 | 65.80 | 23.5× | 10.0× | 0.3 / 1.4 / 1.0 |
| 50,000 nodes, 500,000 edges | GraphSAGE | 6.02 | 70.53 | 34.22 | 11.7× | 5.7× | 0.3 / 1.4 / 0.7 |
| 50,000 nodes, 500,000 edges | GIN | 7.92 | 71.47 | 35.90 | 9.0× | 4.5× | 0.4 / 1.4 / 0.7 |
| 50,000 nodes, 500,000 edges | GAT | 21.46 | 217.38 | 101.78 | 10.1× | 4.7× | 0.8 / 2.4 / 1.5 |
| 50,000 nodes, 500,000 edges | GATv2 | 61.66 | 304.93 | 178.51 | 4.9× | 2.9× | 2.3 / 2.4 / 1.9 |
| 200,000 nodes, 2,000,000 edges | GCN | 36.36 | 1,904.53 | 270.39 | 52.4× | 7.4× | 1.0 / 3.5 / 3.1 |
| 200,000 nodes, 2,000,000 edges | GraphSAGE | 29.56 | 1,182.08 | 127.65 | 40.0× | 4.3× | 0.9 / 2.4 / 2.0 |
| 200,000 nodes, 2,000,000 edges | GIN | 36.27 | 1,180.31 | 142.91 | 32.5× | 3.9× | 1.3 / 2.4 / 2.1 |
| 200,000 nodes, 2,000,000 edges | GAT | 87.74 | 1,940.39 | 394.28 | 22.1× | 4.5× | 2.5 / 5.3 / 5.1 |
| 200,000 nodes, 2,000,000 edges | GATv2 | 276.49 | 2,307.72 | 691.32 | 8.3× | 2.5× | 6.9 / 7.2 / 6.9 |

## Full-batch node classification

Median time per training epoch over 3 seeds.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/node_classification_dark.png">
  <img alt="Full-batch node classification: time per training epoch per dataset and model, for mlx-graphs, PyG on MPS and PyG on CPU" src="figures/node_classification_light.png">
</picture>

| dataset | model | mlx-graphs | PyG MPS | PyG CPU | speedup vs MPS | speedup vs CPU |
|---|---|---:|---:|---:|---:|---:|
| Cora | GCN | 1.66 | 4.99 | 5.23 | 3.0× | 3.1× |
| Cora | GraphSAGE | 2.49 | 19.17 | 9.99 | 7.7× | 4.0× |
| Cora | GAT | 3.17 | 7.48 | 8.69 | 2.4× | 2.7× |
| CiteSeer | GCN | 2.56 | 4.96 | 9.06 | 1.9× | 3.5× |
| CiteSeer | GraphSAGE | 4.18 | 42.16 | 20.63 | 10.1× | 4.9× |
| CiteSeer | GAT | 3.79 | 7.20 | 12.62 | 1.9× | 3.3× |
| PubMed | GCN | 3.04 | 28.35 | 20.27 | 9.3× | 6.7× |
| PubMed | GraphSAGE | 4.51 | 61.38 | 35.50 | 13.6× | 7.9× |
| PubMed | GAT | 8.27 | 38.50 | 32.41 | 4.7× | 3.9× |
| Amazon Photo | GCN | 2.95 | 53.90 | 21.68 | 18.3× | 7.4× |
| Amazon Photo | GraphSAGE | 3.72 | 201.53 | 46.22 | 54.2× | 12.4× |
| Amazon Photo | GAT | 12.82 | 67.08 | 46.06 | 5.2× | 3.6× |
| Amazon Computers | GCN | 5.14 | 115.96 | 39.62 | 22.6× | 7.7× |
| Amazon Computers | GraphSAGE | 5.89 | 733.16 | 87.88 | 124.5× | 14.9× |
| Amazon Computers | GAT | 24.92 | 139.50 | 83.44 | 5.6× | 3.3× |
| Coauthor CS | GCN | 19.61 | 44.65 | 71.39 | 2.3× | 3.6× |
| Coauthor CS | GraphSAGE | 37.83 | timed out | 382.52 | – | 10.1× |
| Coauthor CS | GAT | 26.35 | 62.21 | 93.25 | 2.4× | 3.5× |
| Coauthor Physics | GCN | 41.14 | 207.01 | 144.18 | 5.0× | 3.5× |
| Coauthor Physics | GraphSAGE | 77.04 | out of memory | out of memory | – | – |
| Coauthor Physics | GAT | 57.52 | 249.97 | 189.33 | 4.3× | 3.3× |
| ogbn-arxiv | GCN | 33.28 | 1,040.92 | 252.86 | 31.3× | 7.6× |
| ogbn-arxiv | GraphSAGE | 38.62 | 1,667.99 | 664.60 | 43.2× | 17.2× |
| ogbn-arxiv | GAT | 238.04 | 1,318.93 | 477.29 | 5.5× | 2.0× |

### Full-batch node classification: training quality

Do the models train the same in both frameworks? Training loss at every epoch, and test accuracy at the epoch with the best validation accuracy on the dataset's validation split, mean ± standard deviation over 3 seeds. Final loss is the mean training loss over the last 10 epochs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/node_loss_dark.png">
  <img alt="Full-batch node classification: training loss per epoch for each dataset and model, mlx-graphs, PyG on MPS and PyG on CPU overlaid" src="figures/node_loss_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/node_accuracy_dark.png">
  <img alt="Full-batch node classification: test accuracy per dataset and model, mean and range over seeds, for mlx-graphs, PyG on MPS and PyG on CPU" src="figures/node_accuracy_light.png">
</picture>

| dataset | model | test acc. mlx-graphs | PyG MPS | PyG CPU | final loss mlx-graphs | PyG MPS | PyG CPU |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | GCN | 0.807 ± 0.009 | 0.804 ± 0.007 | 0.804 ± 0.003 | 0.000 | 0.000 | 0.000 |
| Cora | GraphSAGE | 0.799 ± 0.009 | 0.789 ± 0.000 | 0.783 ± 0.003 | 0.000 | 0.000 | 0.000 |
| Cora | GAT | 0.792 ± 0.014 | 0.778 ± 0.038 | 0.775 ± 0.036 | 0.001 | 0.001 | 0.001 |
| CiteSeer | GCN | 0.662 ± 0.001 | 0.652 ± 0.002 | 0.661 ± 0.004 | 0.000 | 0.000 | 0.000 |
| CiteSeer | GraphSAGE | 0.664 ± 0.007 | 0.662 ± 0.004 | 0.656 ± 0.009 | 0.000 | 0.000 | 0.000 |
| CiteSeer | GAT | 0.656 ± 0.015 | 0.663 ± 0.022 | 0.651 ± 0.005 | 0.007 | 0.001 | 0.003 |
| PubMed | GCN | 0.766 ± 0.002 | 0.767 ± 0.002 | 0.768 ± 0.007 | 0.002 | 0.002 | 0.002 |
| PubMed | GraphSAGE | 0.750 ± 0.008 | 0.756 ± 0.007 | 0.749 ± 0.007 | 0.000 | 0.000 | 0.000 |
| PubMed | GAT | 0.739 ± 0.006 | 0.755 ± 0.008 | 0.756 ± 0.010 | 0.001 | 0.001 | 0.001 |
| Amazon Photo | GCN | 0.903 ± 0.006 | 0.906 ± 0.005 | 0.901 ± 0.011 | 0.125 | 0.113 | 0.114 |
| Amazon Photo | GraphSAGE | 0.908 ± 0.005 | 0.903 ± 0.015 | 0.884 ± 0.031 | 0.059 | 0.108 | 0.093 |
| Amazon Photo | GAT | 0.888 ± 0.015 | 0.898 ± 0.016 | 0.895 ± 0.013 | 0.017 | 0.016 | 0.013 |
| Amazon Computers | GCN | 0.799 ± 0.003 | 0.835 ± 0.003 | 0.813 ± 0.006 | 0.192 | 0.135 | 0.166 |
| Amazon Computers | GraphSAGE | 0.783 ± 0.008 | 0.776 ± 0.017 | 0.802 ± 0.015 | 0.118 | 0.184 | 0.096 |
| Amazon Computers | GAT | 0.826 ± 0.028 | 0.812 ± 0.027 | 0.823 ± 0.006 | 0.037 | 0.023 | 0.020 |
| Coauthor CS | GCN | 0.903 ± 0.002 | 0.902 ± 0.005 | 0.899 ± 0.002 | 0.000 | 0.000 | 0.000 |
| Coauthor CS | GraphSAGE | 0.892 ± 0.006 | timed out | 0.900 ± 0.005 | 0.000 | timed out | 0.000 |
| Coauthor CS | GAT | 0.888 ± 0.003 | 0.886 ± 0.006 | 0.881 ± 0.006 | 0.002 | 0.003 | 0.001 |
| Coauthor Physics | GCN | 0.928 ± 0.005 | 0.931 ± 0.000 | 0.932 ± 0.001 | 0.000 | 0.000 | 0.000 |
| Coauthor Physics | GraphSAGE | 0.919 ± 0.008 | out of memory | out of memory | 0.000 | out of memory | out of memory |
| Coauthor Physics | GAT | 0.910 ± 0.007 | 0.895 ± 0.020 | 0.905 ± 0.011 | 0.000 | 0.000 | 0.002 |
| ogbn-arxiv | GCN | 0.633 ± 0.006 | 0.628 ± 0.005 | 0.630 ± 0.005 | 1.396 | 1.437 | 1.434 |
| ogbn-arxiv | GraphSAGE | 0.642 ± 0.004 | 0.641 ± 0.003 | 0.640 ± 0.004 | 1.341 | 1.355 | 1.361 |
| ogbn-arxiv | GAT | 0.649 ± 0.004 | 0.649 ± 0.001 | 0.650 ± 0.001 | 1.314 | 1.308 | 1.314 |

## Graph classification

Median time per training epoch over 3 seeds.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/graph_classification_dark.png">
  <img alt="Graph classification: time per training epoch per dataset and model, for mlx-graphs, PyG on MPS and PyG on CPU" src="figures/graph_classification_light.png">
</picture>

| dataset | model | mlx-graphs | PyG MPS | PyG CPU | speedup vs MPS | speedup vs CPU |
|---|---|---:|---:|---:|---:|---:|
| MUTAG | GCN | 5.18 | 40.24 | 19.35 | 7.8× | 3.7× |
| MUTAG | GraphSAGE | 5.89 | 36.79 | 15.95 | 6.2× | 2.7× |
| MUTAG | GIN | 6.78 | 33.29 | 17.07 | 4.9× | 2.5× |
| PROTEINS | GCN | 35.93 | 378.50 | 130.32 | 10.5× | 3.6× |
| PROTEINS | GraphSAGE | 39.06 | 283.90 | 107.62 | 7.3× | 2.8× |
| PROTEINS | GIN | 43.84 | 294.62 | 116.43 | 6.7× | 2.7× |
| NCI1 | GCN | 112.83 | 575.59 | 427.19 | 5.1× | 3.8× |
| NCI1 | GraphSAGE | 130.53 | 452.78 | 387.00 | 3.5× | 3.0× |
| NCI1 | GIN | 143.00 | 467.05 | 412.30 | 3.3× | 2.9× |
| DD | GCN | 161.95 | 1,502.28 | 523.73 | 9.3× | 3.2× |
| DD | GraphSAGE | 183.95 | 1,204.86 | 449.05 | 6.5× | 2.4× |
| DD | GIN | 187.84 | 1,097.42 | 456.76 | 5.8× | 2.4× |

### Graph classification: training quality

Do the models train the same in both frameworks? Training loss at every epoch, and test accuracy at the epoch with the best validation accuracy on a random 80/10/10 train/validation/test split, mean ± standard deviation over 3 seeds. Final loss is the mean training loss over the last 10 epochs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/graph_loss_dark.png">
  <img alt="Graph classification: training loss per epoch for each dataset and model, mlx-graphs, PyG on MPS and PyG on CPU overlaid" src="figures/graph_loss_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/graph_accuracy_dark.png">
  <img alt="Graph classification: test accuracy per dataset and model, mean and range over seeds, for mlx-graphs, PyG on MPS and PyG on CPU" src="figures/graph_accuracy_light.png">
</picture>

| dataset | model | test acc. mlx-graphs | PyG MPS | PyG CPU | final loss mlx-graphs | PyG MPS | PyG CPU |
|---|---|---:|---:|---:|---:|---:|---:|
| MUTAG | GCN | 0.600 ± 0.041 | 0.617 ± 0.143 | 0.650 ± 0.122 | 0.482 | 0.492 | 0.491 |
| MUTAG | GraphSAGE | 0.600 ± 0.122 | 0.667 ± 0.062 | 0.633 ± 0.125 | 0.457 | 0.464 | 0.470 |
| MUTAG | GIN | 0.667 ± 0.094 | 0.667 ± 0.103 | 0.617 ± 0.143 | 0.413 | 0.405 | 0.408 |
| PROTEINS | GCN | 0.747 ± 0.030 | 0.741 ± 0.013 | 0.729 ± 0.028 | 0.607 | 0.605 | 0.607 |
| PROTEINS | GraphSAGE | 0.729 ± 0.028 | 0.711 ± 0.011 | 0.723 ± 0.019 | 0.603 | 0.594 | 0.596 |
| PROTEINS | GIN | 0.729 ± 0.041 | 0.750 ± 0.007 | 0.738 ± 0.015 | 0.614 | 0.615 | 0.626 |
| NCI1 | GCN | 0.671 ± 0.015 | 0.667 ± 0.012 | 0.666 ± 0.008 | 0.613 | 0.615 | 0.615 |
| NCI1 | GraphSAGE | 0.693 ± 0.030 | 0.685 ± 0.005 | 0.689 ± 0.005 | 0.603 | 0.608 | 0.609 |
| NCI1 | GIN | 0.586 ± 0.078 | 0.629 ± 0.015 | 0.637 ± 0.025 | 0.655 | 0.651 | 0.634 |
| DD | GCN | 0.709 ± 0.069 | 0.725 ± 0.038 | 0.751 ± 0.045 | 0.359 | 0.365 | 0.358 |
| DD | GraphSAGE | 0.737 ± 0.044 | 0.773 ± 0.055 | 0.720 ± 0.073 | 0.314 | 0.300 | 0.307 |
| DD | GIN | 0.697 ± 0.034 | 0.714 ± 0.038 | 0.717 ± 0.040 | 0.171 | 0.176 | 0.167 |
