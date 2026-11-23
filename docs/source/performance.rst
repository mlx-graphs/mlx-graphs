.. _performance:

Benchmarks & validation
=======================

How fast mlx-graphs is compared with PyTorch Geometric on the same Mac, and
how its layers, samplers and models were checked against PyG.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`graph;1.2em;feature-icon` Benchmarks
      :link: benchmarks
      :link-type: doc

      Training time, memory and accuracy against PyG on MPS and on the CPU,
      machine by machine.

   .. grid-item-card:: :octicon:`check-circle;1.2em;feature-icon` Validation against PyG
      :link: validation
      :link-type: doc

      How every layer, sampler and model added for PyG parity was compared
      with PyG, the results, and how to rerun the checks.

.. toctree::
   :hidden:
   :maxdepth: 1

   benchmarks
   validation
