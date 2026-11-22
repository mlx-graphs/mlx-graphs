Apple M5 Pro
------------

*Apple M5 Pro, 64 GB, macOS 27.2.1 · mlx-graphs 0.2.0 (d93847e) · mlx 0.32.4 · PyTorch 2.15.0 · PyG 2.8.1*

One layer
~~~~~~~~~

One convolution layer, forward + backward, on random graphs with 128 input and output features (4 heads for the attention layers).

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M5_Pro/0.2.0-d93847e/figures/layers_speedup_light.png
   :alt: Forward and backward time per layer and graph size, relative to mlx-graphs, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-light
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M5_Pro/0.2.0-d93847e/figures/layers_speedup_dark.png
   :alt: Forward and backward time per layer and graph size, relative to mlx-graphs, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-dark
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M5_Pro/0.2.0-d93847e/figures/layers_memory_light.png
   :alt: Peak memory per layer on the largest graph, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-light
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M5_Pro/0.2.0-d93847e/figures/layers_memory_dark.png
   :alt: Peak memory per layer on the largest graph, for mlx-graphs, PyG on MPS and PyG on CPU
   :class: only-dark
   :width: 100%
