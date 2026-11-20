# mlx-graphs vs PyTorch Geometric

Machine: Apple M5 Pro, 64 GB, macOS 27.2.1. mlx 0.32.4, mlx-graphs 0.0.10, mlx-cluster 0.1.2, torch 2.15.0, torch_geometric 2.8.1, Python 3.12.14.

Times are medians in milliseconds; lower is better. Speedup is the PyG time divided by the mlx-graphs time.

mlx-graphs trains GNNs **4.3× faster than PyG on the Mac's GPU (MPS)** and **4.4× faster than PyG on the CPU** (median over 35 training benchmarks, up to 71×), and reaches the same accuracy.

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
| 10,000 nodes, 100,000 edges | GCN | 1.53 | 12.38 | 9.79 | 8.1× | 6.4× | 0.1 / 1.4 / 0.4 |
| 10,000 nodes, 100,000 edges | GraphSAGE | 0.88 | 4.64 | 7.19 | 5.3× | 8.2× | 0.1 / 1.4 / 0.4 |
| 10,000 nodes, 100,000 edges | GIN | 1.63 | 4.98 | 7.48 | 3.0× | 4.6× | 0.2 / 1.4 / 0.4 |
| 10,000 nodes, 100,000 edges | GAT | 4.50 | 17.42 | 18.28 | 3.9× | 4.1× | 0.4 / 1.4 / 0.5 |
| 10,000 nodes, 100,000 edges | GATv2 | 12.74 | 24.41 | 23.99 | 1.9× | 1.9× | 0.9 / 1.4 / 0.6 |
| 50,000 nodes, 500,000 edges | GCN | 6.97 | 91.37 | 51.84 | 13.1× | 7.4× | 0.3 / 1.4 / 1.0 |
| 50,000 nodes, 500,000 edges | GraphSAGE | 5.31 | 33.86 | 27.89 | 6.4× | 5.3× | 0.3 / 1.4 / 0.7 |
| 50,000 nodes, 500,000 edges | GIN | 6.23 | 34.78 | 30.88 | 5.6× | 5.0× | 0.4 / 1.4 / 0.7 |
| 50,000 nodes, 500,000 edges | GAT | 22.87 | 94.14 | 79.99 | 4.1× | 3.5× | 1.8 / 2.4 / 1.5 |
| 50,000 nodes, 500,000 edges | GATv2 | 69.74 | 131.92 | 128.77 | 1.9× | 1.8× | 2.9 / 2.4 / 1.9 |
| 200,000 nodes, 2,000,000 edges | GCN | 33.29 | 479.68 | 227.54 | 14.4× | 6.8× | 0.9 / 3.6 / 3.1 |
| 200,000 nodes, 2,000,000 edges | GraphSAGE | 21.93 | 228.93 | 110.39 | 10.4× | 5.0× | 0.9 / 2.5 / 2.0 |
| 200,000 nodes, 2,000,000 edges | GIN | 26.99 | 231.44 | 128.33 | 8.6× | 4.8× | 1.3 / 2.5 / 2.1 |
| 200,000 nodes, 2,000,000 edges | GAT | 138.34 | 506.46 | 327.25 | 3.7× | 2.4× | 6.7 / 5.4 / 5.1 |
| 200,000 nodes, 2,000,000 edges | GATv2 | 350.40 | 657.07 | 559.80 | 1.9× | 1.6× | 10.3 / 7.3 / 6.9 |

## Full-batch node classification

Median time per training epoch over 3 seeds.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/node_classification_dark.png">
  <img alt="Full-batch node classification: time per training epoch per dataset and model, for mlx-graphs, PyG on MPS and PyG on CPU" src="figures/node_classification_light.png">
</picture>

| dataset | model | mlx-graphs | PyG MPS | PyG CPU | speedup vs MPS | speedup vs CPU |
|---|---|---:|---:|---:|---:|---:|
| Cora | GCN | 1.10 | 1.45 | 3.24 | 1.3× | 3.0× |
| Cora | GraphSAGE | 0.91 | 6.55 | 6.71 | 7.2× | 7.3× |
| Cora | GAT | 1.27 | 2.22 | 5.17 | 1.8× | 4.1× |
| CiteSeer | GCN | 1.46 | 2.12 | 6.32 | 1.5× | 4.3× |
| CiteSeer | GraphSAGE | 1.84 | 14.36 | 14.78 | 7.8× | 8.0× |
| CiteSeer | GAT | 1.78 | 2.75 | 7.42 | 1.5× | 4.2× |
| PubMed | GCN | 1.87 | 6.08 | 10.32 | 3.2× | 5.5× |
| PubMed | GraphSAGE | 2.13 | 23.93 | 18.00 | 11.2× | 8.5× |
| PubMed | GAT | 4.10 | 11.99 | 18.04 | 2.9× | 4.4× |
| Amazon Photo | GCN | 1.74 | 15.61 | 12.49 | 9.0× | 7.2× |
| Amazon Photo | GraphSAGE | 1.76 | 82.14 | 36.86 | 46.6× | 20.9× |
| Amazon Photo | GAT | 6.33 | 21.53 | 25.99 | 3.4× | 4.1× |
| Amazon Computers | GCN | 2.88 | 36.64 | 23.62 | 12.7× | 8.2× |
| Amazon Computers | GraphSAGE | 2.91 | 207.14 | 70.79 | 71.1× | 24.3× |
| Amazon Computers | GAT | 13.01 | 51.67 | 60.33 | 4.0× | 4.6× |
| Coauthor CS | GCN | 6.34 | 17.96 | 46.97 | 2.8× | 7.4× |
| Coauthor CS | GraphSAGE | 10.61 | 663.35 | 281.49 | 62.5× | 26.5× |
| Coauthor CS | GAT | 9.99 | 31.00 | 58.46 | 3.1× | 5.8× |
| Coauthor Physics | GCN | 13.19 | 65.07 | 106.79 | 4.9× | 8.1× |
| Coauthor Physics | GraphSAGE | 23.21 | out of memory | out of memory | – | – |
| Coauthor Physics | GAT | 23.64 | 92.66 | 147.00 | 3.9× | 6.2× |
| ogbn-arxiv | GCN | 19.79 | 431.33 | 187.55 | 21.8× | 9.5× |
| ogbn-arxiv | GraphSAGE | 20.92 | 457.44 | 490.06 | 21.9× | 23.4× |
| ogbn-arxiv | GAT | 118.24 | 526.73 | 369.97 | 4.5× | 3.1× |

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
| Cora | GCN | 0.804 ± 0.011 | 0.804 ± 0.007 | 0.804 ± 0.003 | 0.000 | 0.000 | 0.000 |
| Cora | GraphSAGE | 0.784 ± 0.006 | 0.789 ± 0.000 | 0.783 ± 0.003 | 0.000 | 0.000 | 0.000 |
| Cora | GAT | 0.795 ± 0.013 | 0.778 ± 0.038 | 0.775 ± 0.036 | 0.004 | 0.001 | 0.001 |
| CiteSeer | GCN | 0.674 ± 0.003 | 0.652 ± 0.002 | 0.661 ± 0.004 | 0.000 | 0.000 | 0.000 |
| CiteSeer | GraphSAGE | 0.665 ± 0.004 | 0.662 ± 0.004 | 0.656 ± 0.009 | 0.000 | 0.000 | 0.000 |
| CiteSeer | GAT | 0.653 ± 0.014 | 0.663 ± 0.022 | 0.651 ± 0.005 | 0.002 | 0.001 | 0.003 |
| PubMed | GCN | 0.765 ± 0.002 | 0.767 ± 0.002 | 0.768 ± 0.007 | 0.004 | 0.002 | 0.002 |
| PubMed | GraphSAGE | 0.749 ± 0.004 | 0.756 ± 0.007 | 0.749 ± 0.007 | 0.001 | 0.000 | 0.000 |
| PubMed | GAT | 0.738 ± 0.007 | 0.755 ± 0.008 | 0.756 ± 0.010 | 0.001 | 0.001 | 0.001 |
| Amazon Photo | GCN | 0.907 ± 0.008 | 0.906 ± 0.004 | 0.901 ± 0.011 | 0.078 | 0.113 | 0.114 |
| Amazon Photo | GraphSAGE | 0.722 ± 0.084 | 0.896 ± 0.015 | 0.884 ± 0.031 | 0.479 | 0.105 | 0.093 |
| Amazon Photo | GAT | 0.894 ± 0.017 | 0.899 ± 0.017 | 0.895 ± 0.013 | 0.018 | 0.020 | 0.013 |
| Amazon Computers | GCN | 0.818 ± 0.012 | 0.835 ± 0.006 | 0.813 ± 0.006 | 0.078 | 0.136 | 0.166 |
| Amazon Computers | GraphSAGE | 0.630 ± 0.026 | 0.787 ± 0.031 | 0.799 ± 0.010 | 0.450 | 0.184 | 0.098 |
| Amazon Computers | GAT | 0.844 ± 0.007 | 0.810 ± 0.031 | 0.823 ± 0.006 | 0.033 | 0.015 | 0.020 |
| Coauthor CS | GCN | 0.903 ± 0.006 | 0.902 ± 0.005 | 0.899 ± 0.002 | 0.000 | 0.000 | 0.000 |
| Coauthor CS | GraphSAGE | 0.893 ± 0.003 | 0.895 ± 0.006 | 0.900 ± 0.005 | 0.000 | 0.000 | 0.000 |
| Coauthor CS | GAT | 0.889 ± 0.003 | 0.890 ± 0.009 | 0.881 ± 0.006 | 0.002 | 0.006 | 0.001 |
| Coauthor Physics | GCN | 0.925 ± 0.004 | 0.931 ± 0.000 | 0.932 ± 0.001 | 0.000 | 0.000 | 0.000 |
| Coauthor Physics | GraphSAGE | 0.923 ± 0.005 | out of memory | out of memory | 0.000 | out of memory | out of memory |
| Coauthor Physics | GAT | 0.910 ± 0.004 | 0.896 ± 0.020 | 0.905 ± 0.011 | 0.002 | 0.000 | 0.002 |
| ogbn-arxiv | GCN | 0.625 ± 0.005 | 0.628 ± 0.005 | 0.630 ± 0.005 | 1.478 | 1.437 | 1.434 |
| ogbn-arxiv | GraphSAGE | 0.649 ± 0.003 | 0.641 ± 0.003 | 0.640 ± 0.004 | 1.317 | 1.355 | 1.361 |
| ogbn-arxiv | GAT | 0.648 ± 0.005 | 0.650 ± 0.001 | 0.650 ± 0.001 | 1.309 | 1.308 | 1.314 |

## Graph classification

Median time per training epoch over 3 seeds.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/graph_classification_dark.png">
  <img alt="Graph classification: time per training epoch per dataset and model, for mlx-graphs, PyG on MPS and PyG on CPU" src="figures/graph_classification_light.png">
</picture>

| dataset | model | mlx-graphs | PyG MPS | PyG CPU | speedup vs MPS | speedup vs CPU |
|---|---|---:|---:|---:|---:|---:|
| MUTAG | GCN | 6.59 | 28.40 | 13.96 | 4.3× | 2.1× |
| MUTAG | GraphSAGE | 5.79 | 24.61 | 10.27 | 4.2× | 1.8× |
| MUTAG | GIN | 6.22 | 25.99 | 11.53 | 4.2× | 1.9× |
| PROTEINS | GCN | 33.90 | 217.69 | 89.66 | 6.4× | 2.6× |
| PROTEINS | GraphSAGE | 30.70 | 172.08 | 70.67 | 5.6× | 2.3× |
| PROTEINS | GIN | 33.05 | 185.22 | 73.07 | 5.6× | 2.2× |
| NCI1 | GCN | 118.20 | 308.04 | 279.26 | 2.6× | 2.4× |
| NCI1 | GraphSAGE | 108.89 | 245.79 | 244.98 | 2.3× | 2.2× |
| NCI1 | GIN | 120.30 | 257.03 | 260.75 | 2.1× | 2.2× |
| DD | GCN | 71.31 | 596.72 | 317.76 | 8.4× | 4.5× |
| DD | GraphSAGE | 89.75 | 512.00 | 275.99 | 5.7× | 3.1× |
| DD | GIN | 113.81 | 449.53 | 278.80 | 3.9× | 2.4× |

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
| MUTAG | GCN | 0.633 ± 0.143 | 0.617 ± 0.143 | 0.650 ± 0.122 | 0.499 | 0.492 | 0.491 |
| MUTAG | GraphSAGE | 0.617 ± 0.143 | 0.650 ± 0.041 | 0.633 ± 0.125 | 0.465 | 0.465 | 0.470 |
| MUTAG | GIN | 0.700 ± 0.108 | 0.633 ± 0.143 | 0.617 ± 0.143 | 0.401 | 0.418 | 0.408 |
| PROTEINS | GCN | 0.720 ± 0.021 | 0.738 ± 0.011 | 0.729 ± 0.028 | 0.610 | 0.607 | 0.607 |
| PROTEINS | GraphSAGE | 0.729 ± 0.017 | 0.720 ± 0.004 | 0.723 ± 0.019 | 0.596 | 0.596 | 0.596 |
| PROTEINS | GIN | 0.732 ± 0.032 | 0.729 ± 0.011 | 0.738 ± 0.015 | 0.612 | 0.608 | 0.626 |
| NCI1 | GCN | 0.678 ± 0.004 | 0.666 ± 0.015 | 0.666 ± 0.008 | 0.620 | 0.616 | 0.615 |
| NCI1 | GraphSAGE | 0.699 ± 0.013 | 0.693 ± 0.006 | 0.689 ± 0.005 | 0.596 | 0.605 | 0.609 |
| NCI1 | GIN | 0.667 ± 0.011 | 0.643 ± 0.027 | 0.637 ± 0.025 | 0.636 | 0.642 | 0.634 |
| DD | GCN | 0.695 ± 0.024 | 0.739 ± 0.049 | 0.751 ± 0.045 | 0.459 | 0.367 | 0.358 |
| DD | GraphSAGE | 0.734 ± 0.034 | 0.717 ± 0.038 | 0.720 ± 0.073 | 0.301 | 0.301 | 0.307 |
| DD | GIN | 0.723 ± 0.079 | 0.714 ± 0.031 | 0.717 ± 0.040 | 0.191 | 0.189 | 0.167 |
