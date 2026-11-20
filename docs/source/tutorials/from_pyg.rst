.. _from_pyg:

Coming from PyTorch Geometric
=============================

mlx-graphs follows the design of `PyTorch Geometric <https://pyg.org>`_ (PyG):
graphs in COO format, a ``MessagePassing`` base class, batching by
concatenating graphs, and layers with the same names. Porting a PyG model is
mostly renaming. This page lists the differences and walks through porting a
model and its training loop.

.. contents::
    :local:


Names at a glance
-----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - PyG
     - mlx-graphs
   * - ``Data(x, edge_index, edge_attr, y)``
     - ``GraphData(edge_index, node_features, edge_features, node_labels / graph_labels)``
   * - ``DataLoader(dataset, batch_size)``
     - :class:`~mlx_graphs.loaders.Dataloader` ``(dataset, batch_size)``
   * - ``batch.batch``
     - ``batch.batch_indices``
   * - ``GCNConv(in_channels, out_channels)``
     - ``GCNConv(node_features_dim, out_features_dim)``
   * - ``conv(x, edge_index)``
     - ``conv(edge_index, node_features)``: **edges first**
   * - ``global_mean_pool(x, batch)``
     - ``global_mean_pool(node_features, batch_indices)``
   * - ``torch_geometric.nn.models.GCN``
     - :class:`~mlx_graphs.nn.GCN` (also ``GraphSAGE``, ``GIN``, ``GAT``, ``MLP``)
   * - ``Planetoid(root, "Cora")``
     - ``PlanetoidDataset("cora")``
   * - ``message(self, x_j, x_i)``
     - ``message(self, src_features, dst_features, **kwargs)``
   * - ``.to(device)``
     - nothing: memory is shared by the CPU and the GPU

The tensors are ``mx.array`` instead of ``torch.Tensor``.


Converting data
---------------

:func:`~mlx_graphs.utils.from_pyg` converts a PyG ``Data`` object to a
:class:`~mlx_graphs.data.GraphData`, and :func:`~mlx_graphs.utils.to_pyg`
converts back. ``x``, ``edge_attr`` and ``y`` are renamed, and every other
attribute (masks, positions...) keeps its name:

.. code-block:: python

    from torch_geometric.datasets import Planetoid

    from mlx_graphs.utils import from_pyg

    data = Planetoid(root="data/Planetoid", name="Cora")[0]
    graph = from_pyg(data)
    >>> GraphData(
            edge_index(shape=(2, 10556), int64)
            node_features(shape=(2708, 1433), float32)
            node_labels(shape=(2708,), int64)
            train_mask(shape=(2708,), bool)
            val_mask(shape=(2708,), bool)
            test_mask(shape=(2708,), bool))

A whole PyG dataset converts graph by graph, and the list goes straight into
a :class:`~mlx_graphs.loaders.Dataloader`:

.. code-block:: python

    from torch_geometric.datasets import TUDataset

    from mlx_graphs.loaders import Dataloader

    graphs = [from_pyg(data) for data in TUDataset(root="data/TU", name="MUTAG")]
    loader = Dataloader(graphs, batch_size=64, shuffle=True)

Many datasets are also built in (:ref:`datasets`), with no PyG needed.


Porting a model
---------------

A 2-layer GCN for node classification, in PyG:

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv


    class GCN(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)

        def forward(self, x, edge_index):
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            return self.conv2(x, edge_index)

And in mlx-graphs:

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_graphs.nn import GCNConv


    class GCN(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            # PyG's GCNConv adds self-loops by default, mlx-graphs' doesn't
            self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True)
            self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True)
            self.dropout = nn.Dropout(p=0.5)

        def __call__(self, edge_index, node_features):
            h = self.dropout(node_features)
            h = nn.relu(self.conv1(edge_index, h))
            h = self.dropout(h)
            return self.conv2(edge_index, h)

The differences: ``mlx.nn.Module`` and ``__call__`` instead of ``forward``,
``edge_index`` first in the calls, dropout as a module (it follows
``model.train()`` / ``model.eval()`` as in PyTorch), and explicit self-loops.

The ready-made :class:`~mlx_graphs.nn.GCN` model does the same in one line,
with self-loops on by default:

.. code-block:: python

    from mlx_graphs.nn import GCN

    model = GCN(1433, 64, num_layers=2, out_features_dim=7, dropout=0.5)


Porting the training loop
-------------------------

PyG trains with autograd and an optimizer that holds the parameters:

.. code-block:: python

    model = GCN(dataset.num_features, 64, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

MLX computes gradients with a function transformation,
``nn.value_and_grad``, and evaluates lazily: ``mx.eval`` runs the step. Here
with the ready-made model on the Cora graph converted above:

.. code-block:: python

    import mlx.optimizers as optim
    import numpy as np

    from mlx_graphs.nn import GCN

    model = GCN(graph.num_node_features, 64, num_layers=2, out_features_dim=7, dropout=0.5)
    # bias_correction=True: the update rule of torch.optim.Adam
    optimizer = optim.Adam(learning_rate=0.01, bias_correction=True)
    train = mx.array(np.nonzero(np.array(graph.train_mask))[0])


    def loss_fn(model):
        out = model(graph.edge_index, graph.node_features)
        return nn.losses.cross_entropy(
            out[train], graph.node_labels[train], reduction="mean"
        )


    loss_and_grad = nn.value_and_grad(model, loss_fn)
    model.train()
    for epoch in range(200):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    model.eval()
    pred = model(graph.edge_index, graph.node_features).argmax(axis=1)

Boolean masks are turned into index arrays once (``train``), since MLX
indexes with integer arrays.

.. warning::

    Create MLX's ``Adam`` / ``AdamW`` with ``bias_correction=True`` when
    porting from PyTorch. MLX's default has no bias correction and takes
    much larger first steps, which changes how training goes (about 4 points
    of accuracy for GAT on Cora in our runs).


Porting a custom layer
----------------------

A PyG layer that weights neighbour features by an edge weight:

.. code-block:: python

    from torch_geometric.nn import MessagePassing


    class WeightedConv(MessagePassing):
        def __init__(self, in_dim, out_dim):
            super().__init__(aggr="add")
            self.lin = torch.nn.Linear(in_dim, out_dim)

        def forward(self, x, edge_index, edge_weight):
            return self.propagate(edge_index, x=self.lin(x), edge_weight=edge_weight)

        def message(self, x_j, edge_weight):
            return edge_weight.view(-1, 1) * x_j

In mlx-graphs, ``propagate`` takes the node features and passes extra
arguments to ``message`` through ``message_kwargs``. ``x_j`` and ``x_i``
become ``src_features`` and ``dst_features``:

.. code-block:: python

    from mlx_graphs.nn import Linear, MessagePassing


    class WeightedConv(MessagePassing):
        def __init__(self, in_dim, out_dim):
            super().__init__(aggr="add")
            self.lin = Linear(in_dim, out_dim)

        def __call__(self, edge_index, node_features, edge_weights):
            return self.propagate(
                edge_index,
                self.lin(node_features),
                message_kwargs={"edge_weights": edge_weights},
            )

        def message(self, src_features, dst_features, edge_weights=None, **kwargs):
            return edge_weights.reshape(-1, 1) * src_features

This message is a weighted sum of neighbour features, which is what the
default ``message`` already computes. Dropping the ``message`` override and
decorating the class with
:func:`~mlx_graphs.nn.message_passing.fused_message_passing` aggregates with
one sparse x dense matmul on the GPU instead of building the
``[num_edges, features]`` array of messages, which is how the built-in
layers get their speed (see the :doc:`../benchmarks`):

.. code-block:: python

    from mlx_graphs.nn import fused_message_passing


    @fused_message_passing
    class WeightedConv(MessagePassing):
        def __init__(self, in_dim, out_dim):
            super().__init__(aggr="add")
            self.lin = Linear(in_dim, out_dim)

        def __call__(self, edge_index, node_features, edge_weights):
            return self.propagate(
                edge_index,
                self.lin(node_features),
                message_kwargs={"edge_weights": edge_weights},
            )


Things to watch for
-------------------

- **Self-loops.** ``GCNConv`` doesn't add self-loops unless
  ``add_self_loops=True`` (the ready-made ``GCN`` model does), and
  ``GATConv`` / ``GATv2Conv`` never add them: call
  :func:`~mlx_graphs.utils.add_self_loops` on the graph once beforehand.
- **Lazy evaluation.** MLX builds a graph of operations and runs it when a
  value is needed. Call ``mx.eval`` on the parameters and optimizer state at
  the end of each training step, and ``.item()`` to read a loss.
- **Optimizer.** ``bias_correction=True`` for ``Adam`` / ``AdamW``, as above.
- **Initialization.** Layers are initialized as in PyG (Glorot for GCN and
  GAT, PyTorch's default ``Linear`` initialization for GraphSAGE and
  GraphConv), so a ported model starts from the same distribution.
