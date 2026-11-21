Apple M1 Pro
------------

*Apple M1 Pro, 32 GB, macOS 27.0 · mlx-graphs 0.1.0 (c5f1246) · mlx 0.32.4 · PyTorch 2.15.0 · PyG 2.8.1*

mlx-graphs trains GNNs **6.0× faster than PyG on the Mac's GPU (MPS)** and **3.5× faster than PyG on the CPU** (median over 34 training benchmarks, up to 125×), and reaches the same accuracy.

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/headline_light.png
   :alt: Time of PyG on MPS and on CPU relative to mlx-graphs for one layer, node classification and graph classification, and the test accuracy of each framework
   :class: only-light
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/headline_dark.png
   :alt: Time of PyG on MPS and on CPU relative to mlx-graphs for one layer, node classification and graph classification, and the test accuracy of each framework
   :class: only-dark
   :width: 100%

One layer
~~~~~~~~~

One convolution layer, forward + backward, on random graphs with 128 input and output features (4 heads for GAT and GATv2).

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/layers_speedup_light.png
   :alt: Forward and backward time per layer and graph size, relative to mlx-graphs, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-light
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/layers_speedup_dark.png
   :alt: Forward and backward time per layer and graph size, relative to mlx-graphs, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-dark
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/layers_memory_light.png
   :alt: Peak memory per layer on the largest graph, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-light
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/layers_memory_dark.png
   :alt: Peak memory per layer on the largest graph, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-dark
   :width: 100%

Node classification
~~~~~~~~~~~~~~~~~~~

Full-batch training of 2-layer models (64 hidden units, Adam, lr 0.01), the whole graph at every step.

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/node_classification_light.png
   :alt: Node classification: time per training epoch per dataset and model, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-light
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/node_classification_dark.png
   :alt: Node classification: time per training epoch per dataset and model, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-dark
   :width: 100%

**Same training, same results.** The training loss of every framework at every epoch, overlaid, and the test accuracy reached at the epoch with the best validation accuracy.

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/node_loss_light.png
   :alt: Node classification: training loss per epoch for each dataset and model, mlx-graphs, PyG on MPS and PyG on CPU overlaid
   :class: only-light
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/node_loss_dark.png
   :alt: Node classification: training loss per epoch for each dataset and model, mlx-graphs, PyG on MPS and PyG on CPU overlaid
   :class: only-dark
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/node_accuracy_light.png
   :alt: Node classification: test accuracy per dataset and model, mean and range over seeds, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-light
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/node_accuracy_dark.png
   :alt: Node classification: test accuracy per dataset and model, mean and range over seeds, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-dark
   :width: 100%

Graph classification
~~~~~~~~~~~~~~~~~~~~

3 layers + mean pooling, batches of 64 graphs, random 80/10/10 split. The time per epoch includes batching the graphs.

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/graph_classification_light.png
   :alt: Graph classification: time per training epoch per dataset and model, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-light
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/graph_classification_dark.png
   :alt: Graph classification: time per training epoch per dataset and model, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-dark
   :width: 100%

**Same training, same results.** The training loss of every framework at every epoch, overlaid, and the test accuracy reached at the epoch with the best validation accuracy.

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/graph_loss_light.png
   :alt: Graph classification: training loss per epoch for each dataset and model, mlx-graphs, PyG on MPS and PyG on CPU overlaid
   :class: only-light
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/graph_loss_dark.png
   :alt: Graph classification: training loss per epoch for each dataset and model, mlx-graphs, PyG on MPS and PyG on CPU overlaid
   :class: only-dark
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/graph_accuracy_light.png
   :alt: Graph classification: test accuracy per dataset and model, mean and range over seeds, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-light
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M1_Pro/0.1.0-c5f1246/figures/graph_accuracy_dark.png
   :alt: Graph classification: test accuracy per dataset and model, mean and range over seeds, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-dark
   :width: 100%
