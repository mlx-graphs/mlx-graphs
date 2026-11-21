# mlx-graphs vs PyTorch Geometric

Machine: Apple M1 Pro, 32 GB, macOS 27.0. mlx 0.32.4, mlx-graphs 0.0.11, mlx-cluster 0.1.2, torch 2.15.0, torch_geometric 2.8.1, Python 3.12.10.

Times are medians in milliseconds; lower is better. Speedup is the PyG time divided by the mlx-graphs time.

mlx-graphs trains GNNs **4.1× faster than PyG on the Mac's GPU (MPS)** and **2.4× faster than PyG on the CPU** (median over 34 training benchmarks, up to 48×), and reaches the same accuracy.

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
| 10,000 nodes, 100,000 edges | GCN | 2.88 | 30.59 | 12.59 | 10.6× | 4.4× | 0.1 / 1.3 / 0.4 |
| 10,000 nodes, 100,000 edges | GraphSAGE | 2.04 | 12.18 | 6.78 | 6.0× | 3.3× | 0.1 / 1.3 / 0.4 |
| 10,000 nodes, 100,000 edges | GIN | 2.42 | 12.35 | 7.75 | 5.1× | 3.2× | 0.1 / 1.3 / 0.4 |
| 10,000 nodes, 100,000 edges | GAT | 7.55 | 42.72 | 22.12 | 5.7× | 2.9× | 0.4 / 1.3 / 0.5 |
| 10,000 nodes, 100,000 edges | GATv2 | 19.16 | 58.78 | 34.55 | 3.1× | 1.8× | 0.8 / 1.3 / 0.6 |
| 50,000 nodes, 500,000 edges | GCN | 12.97 | 152.96 | 61.80 | 11.8× | 4.8× | 0.2 / 1.4 / 1.0 |
| 50,000 nodes, 500,000 edges | GraphSAGE | 9.07 | 71.29 | 31.26 | 7.9× | 3.4× | 0.3 / 1.4 / 0.7 |
| 50,000 nodes, 500,000 edges | GIN | 10.79 | 71.28 | 35.36 | 6.6× | 3.3× | 0.4 / 1.4 / 0.7 |
| 50,000 nodes, 500,000 edges | GAT | 36.49 | 217.33 | 94.56 | 6.0× | 2.6× | 1.7 / 2.4 / 1.5 |
| 50,000 nodes, 500,000 edges | GATv2 | 94.54 | 305.08 | 163.53 | 3.2× | 1.7× | 2.8 / 2.4 / 1.9 |
| 200,000 nodes, 2,000,000 edges | GCN | 58.07 | 1,893.67 | 259.78 | 32.6× | 4.5× | 0.9 / 3.5 / 3.1 |
| 200,000 nodes, 2,000,000 edges | GraphSAGE | 39.08 | 1,176.97 | 128.46 | 30.1× | 3.3× | 0.8 / 2.4 / 2.0 |
| 200,000 nodes, 2,000,000 edges | GIN | 45.14 | 1,180.46 | 137.83 | 26.2× | 3.1× | 1.2 / 2.4 / 2.1 |
| 200,000 nodes, 2,000,000 edges | GAT | 182.53 | 1,943.10 | 395.59 | 10.6× | 2.2× | 6.7 / 5.3 / 5.1 |
| 200,000 nodes, 2,000,000 edges | GATv2 | 484.63 | 2,301.12 | 670.32 | 4.7× | 1.4× | 11.1 / 7.2 / 6.9 |

## Full-batch node classification

Median time per training epoch over 3 seeds.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/node_classification_dark.png">
  <img alt="Full-batch node classification: time per training epoch per dataset and model, for mlx-graphs, PyG on MPS and PyG on CPU" src="figures/node_classification_light.png">
</picture>

| dataset | model | mlx-graphs | PyG MPS | PyG CPU | speedup vs MPS | speedup vs CPU |
|---|---|---:|---:|---:|---:|---:|
| Cora | GCN | 2.29 | 4.97 | 5.08 | 2.2× | 2.2× |
| Cora | GraphSAGE | 2.58 | 19.09 | 9.86 | 7.4× | 3.8× |
| Cora | GAT | 3.15 | 7.33 | 8.60 | 2.3× | 2.7× |
| CiteSeer | GCN | 3.09 | 5.09 | 8.93 | 1.6× | 2.9× |
| CiteSeer | GraphSAGE | 4.24 | 41.95 | 20.61 | 9.9× | 4.9× |
| CiteSeer | GAT | 3.89 | 7.20 | 12.49 | 1.9× | 3.2× |
| PubMed | GCN | 5.31 | 28.25 | 19.93 | 5.3× | 3.8× |
| PubMed | GraphSAGE | 6.20 | 61.59 | 35.48 | 9.9× | 5.7× |
| PubMed | GAT | 10.62 | 38.23 | 32.71 | 3.6× | 3.1× |
| Amazon Photo | GCN | 13.00 | 54.14 | 21.64 | 4.2× | 1.7× |
| Amazon Photo | GraphSAGE | 6.77 | 201.86 | 45.15 | 29.8× | 6.7× |
| Amazon Photo | GAT | 21.66 | 66.99 | 46.31 | 3.1× | 2.1× |
| Amazon Computers | GCN | 28.02 | 115.75 | 39.38 | 4.1× | 1.4× |
| Amazon Computers | GraphSAGE | 15.19 | 734.29 | 91.77 | 48.3× | 6.0× |
| Amazon Computers | GAT | 45.06 | 139.64 | 82.53 | 3.1× | 1.8× |
| Coauthor CS | GCN | 20.74 | 44.76 | 72.74 | 2.2× | 3.5× |
| Coauthor CS | GraphSAGE | 37.75 | timed out | 356.45 | – | 9.4× |
| Coauthor CS | GAT | 28.17 | 62.42 | 90.59 | 2.2× | 3.2× |
| Coauthor Physics | GCN | 43.55 | 206.94 | 143.77 | 4.8× | 3.3× |
| Coauthor Physics | GraphSAGE | 78.78 | out of memory | out of memory | – | – |
| Coauthor Physics | GAT | 62.91 | 249.98 | 190.98 | 4.0× | 3.0× |
| ogbn-arxiv | GCN | 398.35 | 1,034.69 | 246.65 | 2.6× | 0.6× |
| ogbn-arxiv | GraphSAGE | 368.10 | 1,668.62 | 673.56 | 4.5× | 1.8× |
| ogbn-arxiv | GAT | 578.13 | 1,311.15 | 457.99 | 2.3× | 0.8× |

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
| Cora | GCN | 0.805 ± 0.011 | 0.804 ± 0.007 | 0.804 ± 0.003 | 0.000 | 0.000 | 0.000 |
| Cora | GraphSAGE | 0.783 ± 0.005 | 0.789 ± 0.000 | 0.783 ± 0.003 | 0.000 | 0.000 | 0.000 |
| Cora | GAT | 0.795 ± 0.014 | 0.778 ± 0.038 | 0.775 ± 0.036 | 0.001 | 0.001 | 0.001 |
| CiteSeer | GCN | 0.675 ± 0.003 | 0.652 ± 0.002 | 0.661 ± 0.004 | 0.000 | 0.000 | 0.000 |
| CiteSeer | GraphSAGE | 0.667 ± 0.006 | 0.662 ± 0.004 | 0.656 ± 0.009 | 0.000 | 0.000 | 0.000 |
| CiteSeer | GAT | 0.654 ± 0.021 | 0.663 ± 0.022 | 0.651 ± 0.005 | 0.000 | 0.001 | 0.003 |
| PubMed | GCN | 0.765 ± 0.003 | 0.767 ± 0.002 | 0.768 ± 0.007 | 0.004 | 0.002 | 0.002 |
| PubMed | GraphSAGE | 0.750 ± 0.005 | 0.756 ± 0.007 | 0.749 ± 0.007 | 0.001 | 0.000 | 0.000 |
| PubMed | GAT | 0.738 ± 0.007 | 0.755 ± 0.008 | 0.756 ± 0.010 | 0.001 | 0.001 | 0.001 |
| Amazon Photo | GCN | 0.907 ± 0.007 | 0.905 ± 0.005 | 0.901 ± 0.011 | 0.078 | 0.113 | 0.114 |
| Amazon Photo | GraphSAGE | 0.755 ± 0.076 | 0.900 ± 0.012 | 0.884 ± 0.031 | 0.435 | 0.105 | 0.093 |
| Amazon Photo | GAT | 0.889 ± 0.018 | 0.894 ± 0.012 | 0.895 ± 0.013 | 0.021 | 0.014 | 0.013 |
| Amazon Computers | GCN | 0.817 ± 0.002 | 0.826 ± 0.015 | 0.813 ± 0.006 | 0.076 | 0.133 | 0.166 |
| Amazon Computers | GraphSAGE | 0.679 ± 0.027 | 0.776 ± 0.017 | 0.802 ± 0.015 | 0.452 | 0.184 | 0.096 |
| Amazon Computers | GAT | 0.844 ± 0.007 | 0.812 ± 0.032 | 0.823 ± 0.006 | 0.039 | 0.024 | 0.020 |
| Coauthor CS | GCN | 0.904 ± 0.006 | 0.902 ± 0.005 | 0.899 ± 0.002 | 0.000 | 0.000 | 0.000 |
| Coauthor CS | GraphSAGE | 0.893 ± 0.003 | timed out | 0.900 ± 0.005 | 0.000 | timed out | 0.000 |
| Coauthor CS | GAT | 0.886 ± 0.003 | 0.887 ± 0.007 | 0.881 ± 0.006 | 0.001 | 0.002 | 0.001 |
| Coauthor Physics | GCN | 0.925 ± 0.004 | 0.931 ± 0.000 | 0.932 ± 0.001 | 0.000 | 0.000 | 0.000 |
| Coauthor Physics | GraphSAGE | 0.923 ± 0.004 | out of memory | out of memory | 0.000 | out of memory | out of memory |
| Coauthor Physics | GAT | 0.905 ± 0.014 | 0.894 ± 0.021 | 0.905 ± 0.011 | 0.001 | 0.000 | 0.002 |
| ogbn-arxiv | GCN | 0.625 ± 0.005 | 0.628 ± 0.005 | 0.630 ± 0.005 | 1.478 | 1.437 | 1.434 |
| ogbn-arxiv | GraphSAGE | 0.649 ± 0.003 | 0.641 ± 0.003 | 0.640 ± 0.004 | 1.318 | 1.355 | 1.361 |
| ogbn-arxiv | GAT | 0.649 ± 0.005 | 0.650 ± 0.001 | 0.650 ± 0.001 | 1.310 | 1.308 | 1.314 |

## Graph classification

Median time per training epoch over 3 seeds.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/graph_classification_dark.png">
  <img alt="Graph classification: time per training epoch per dataset and model, for mlx-graphs, PyG on MPS and PyG on CPU" src="figures/graph_classification_light.png">
</picture>

| dataset | model | mlx-graphs | PyG MPS | PyG CPU | speedup vs MPS | speedup vs CPU |
|---|---|---:|---:|---:|---:|---:|
| MUTAG | GCN | 8.76 | 41.51 | 19.59 | 4.7× | 2.2× |
| MUTAG | GraphSAGE | 7.54 | 37.89 | 15.89 | 5.0× | 2.1× |
| MUTAG | GIN | 8.32 | 33.46 | 17.22 | 4.0× | 2.1× |
| PROTEINS | GCN | 54.48 | 381.69 | 131.25 | 7.0× | 2.4× |
| PROTEINS | GraphSAGE | 49.26 | 293.27 | 108.00 | 6.0× | 2.2× |
| PROTEINS | GIN | 53.42 | 300.78 | 116.69 | 5.6× | 2.2× |
| NCI1 | GCN | 182.73 | 579.00 | 428.90 | 3.2× | 2.3× |
| NCI1 | GraphSAGE | 164.33 | 456.69 | 386.92 | 2.8× | 2.4× |
| NCI1 | GIN | 179.14 | 469.62 | 414.38 | 2.6× | 2.3× |
| DD | GCN | 181.88 | 1,509.40 | 515.60 | 8.3× | 2.8× |
| DD | GraphSAGE | 195.26 | 1,195.13 | 443.00 | 6.1× | 2.3× |
| DD | GIN | 203.12 | 1,085.51 | 461.54 | 5.3× | 2.3× |

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
| MUTAG | GCN | 0.600 ± 0.082 | 0.600 ± 0.122 | 0.650 ± 0.122 | 0.498 | 0.493 | 0.491 |
| MUTAG | GraphSAGE | 0.633 ± 0.125 | 0.700 ± 0.108 | 0.633 ± 0.125 | 0.464 | 0.464 | 0.470 |
| MUTAG | GIN | 0.633 ± 0.143 | 0.650 ± 0.108 | 0.617 ± 0.143 | 0.392 | 0.406 | 0.408 |
| PROTEINS | GCN | 0.759 ± 0.019 | 0.744 ± 0.011 | 0.729 ± 0.028 | 0.612 | 0.607 | 0.607 |
| PROTEINS | GraphSAGE | 0.726 ± 0.030 | 0.714 ± 0.039 | 0.723 ± 0.019 | 0.596 | 0.596 | 0.596 |
| PROTEINS | GIN | 0.735 ± 0.030 | 0.747 ± 0.044 | 0.738 ± 0.015 | 0.612 | 0.607 | 0.626 |
| NCI1 | GCN | 0.656 ± 0.005 | 0.663 ± 0.011 | 0.666 ± 0.008 | 0.624 | 0.615 | 0.615 |
| NCI1 | GraphSAGE | 0.693 ± 0.016 | 0.681 ± 0.004 | 0.689 ± 0.005 | 0.594 | 0.605 | 0.609 |
| NCI1 | GIN | 0.673 ± 0.020 | 0.657 ± 0.018 | 0.637 ± 0.025 | 0.625 | 0.646 | 0.634 |
| DD | GCN | 0.709 ± 0.017 | 0.739 ± 0.025 | 0.751 ± 0.045 | 0.462 | 0.360 | 0.358 |
| DD | GraphSAGE | 0.703 ± 0.055 | 0.751 ± 0.045 | 0.720 ± 0.073 | 0.301 | 0.305 | 0.307 |
| DD | GIN | 0.703 ± 0.097 | 0.655 ± 0.049 | 0.717 ± 0.040 | 0.208 | 0.329 | 0.167 |
