import mlx.core as mx
import pytest
import torch

from mlx_graphs.data.batch import GraphDataBatch
from mlx_graphs.data.data import GraphData, HeteroGraphData
from mlx_graphs.nn import HeteroLayerNormalization, LayerNormalization


@pytest.mark.parametrize("mode", ["node", "graph"])
def test_layer_norm(mode):
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import LayerNorm as torch_LayerNormalization

    # Graph 1
    x = torch.tensor(
        [[0.8, 0.1], [0.1, 0.9], [0.5, 0.5], [0.7, 0.3]], dtype=torch.float
    )
    edge_index = torch.tensor([[0, 1, 2, 2, 3], [1, 0, 2, 3, 0]], dtype=torch.long)

    # Graph 2
    x2 = torch.tensor(
        [[0.9, 0.4], [0.7, 1.0], [0.8, 0.9], [0.7, 0.3]], dtype=torch.float
    )
    edge_index2 = torch.tensor([[0, 1, 2, 2, 3], [1, 0, 2, 3, 0]], dtype=torch.long)

    graph_batch = Batch.from_data_list([Data(x, edge_index), Data(x2, edge_index2)])

    torch_layer_norm = torch_LayerNormalization(2, affine=False, mode=mode)

    graphs = [
        GraphData(
            edge_index=mx.array([[0, 1, 2, 2, 3], [1, 0, 2, 3, 0]]),
            node_features=mx.array([[0.8, 0.1], [0.1, 0.9], [0.5, 0.5], [0.7, 0.3]]),
        ),
        GraphData(
            edge_index=mx.array([[0, 1, 2, 2, 3], [1, 0, 2, 3, 0]]),
            node_features=mx.array([[0.9, 0.4], [0.7, 1.0], [0.8, 0.9], [0.7, 0.3]]),
        ),
    ]
    batch = GraphDataBatch(graphs)
    layer_norm = LayerNormalization(2, affine=False, mode=mode)
    assert mx.allclose(
        mx.array(torch_layer_norm(graph_batch.x, graph_batch.batch).numpy()),
        layer_norm(batch.node_features, batch.batch_indices),
    )


@pytest.mark.parametrize("mode", ["graph", "node"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("affine", [True, False])
def test_layer_norm_matches_pyg_with_gradients(mode, batched, affine):
    # "graph" mode broadcast [B, 1] statistics against [B] counts (only right
    # when num_features == num_graphs) and failed without batch; "node" mode
    # used separate affine parameters.
    import numpy as np
    from torch_geometric.nn import LayerNorm as PyGLayerNorm

    rng = np.random.default_rng(0)
    x = rng.normal(size=(9, 5)).astype(np.float32)
    # graphs of 4 nodes, 1 node (a single-node graph), none (graph 2) and 4
    batch = np.array([0, 0, 0, 0, 1, 3, 3, 3, 3])
    weight = rng.normal(size=5).astype(np.float32)
    bias = rng.normal(size=5).astype(np.float32)
    cot = rng.normal(size=(9, 5)).astype(np.float32)

    ref = PyGLayerNorm(5, affine=affine, mode=mode)
    norm = LayerNormalization(5, affine=affine, mode=mode)
    if affine:
        with torch.no_grad():
            ref.weight.copy_(torch.tensor(weight))
            ref.bias.copy_(torch.tensor(bias))
        norm.weight, norm.bias = mx.array(weight), mx.array(bias)
    batch_args = (mx.array(batch),) if batched else ()
    torch_batch_args = (torch.tensor(batch),) if batched else ()

    x_torch = torch.tensor(x, requires_grad=True)
    expected = ref(x_torch, *torch_batch_args)
    (expected * torch.tensor(cot)).sum().backward()

    def loss(params, x):
        norm.update(params)
        return (norm(x, *batch_args) * mx.array(cot)).sum()

    out = norm(mx.array(x), *batch_args)
    grads, x_grad = mx.grad(loss, argnums=(0, 1))(
        norm.trainable_parameters(), mx.array(x)
    )
    tol = dict(rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.array(out), expected.detach().numpy(), **tol)
    np.testing.assert_allclose(np.array(x_grad), x_torch.grad.numpy(), **tol)
    if affine:
        np.testing.assert_allclose(np.array(grads["weight"]), ref.weight.grad, **tol)
        np.testing.assert_allclose(np.array(grads["bias"]), ref.bias.grad, **tol)
    assert sorted(dict(norm.trainable_parameters())) == (
        ["bias", "weight"] if affine else []
    )


def _flatten_by_type(node_features_dict: dict, type_order: list):
    """Mirrors PyG's HeteroData.to_homogeneous(): concatenates per-type node
    features into a single (x, type_vec) pair, with type_vec[i] holding the
    integer type index of row i, following `type_order`.
    """
    xs = []
    type_vecs = []
    for type_index, node_type in enumerate(type_order):
        feats = node_features_dict[node_type]
        xs.append(feats)
        type_vecs.append(mx.full((feats.shape[0],), type_index, dtype=mx.int32))
    return mx.concatenate(xs, axis=0), mx.concatenate(type_vecs, axis=0)


def _build_graphs(user_features, movie_features, edge_index):
    """Builds matching torch_geometric.data.HeteroData and mlx_graphs
    HeteroGraphData objects from the same underlying numpy arrays.
    """
    from torch_geometric.data import HeteroData

    torch_data = HeteroData()
    torch_data["user"].x = torch.tensor(user_features)
    torch_data["movie"].x = torch.tensor(movie_features)
    torch_data["user", "rates", "movie"].edge_index = torch.tensor(edge_index)

    mlx_data = HeteroGraphData(
        edge_index_dict={
            ("user", "rates", "movie"): mx.array(edge_index),
        },
        node_features_dict={
            "user": mx.array(user_features),
            "movie": mx.array(movie_features),
        },
    )
    return torch_data, mlx_data


def test_hetero_layer_norm():
    import numpy as np
    from torch_geometric.nn import HeteroLayerNorm as torch_HeteroLayerNorm

    user_features = np.random.uniform(0, 1, (5, 4)).astype(np.float32)
    movie_features = np.random.uniform(0, 1, (7, 4)).astype(np.float32)
    edge_index = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)

    torch_data, mlx_data = _build_graphs(user_features, movie_features, edge_index)

    # NOTE: assumes to_homogeneous() assigns type indices in the same order
    # node types were first inserted ("user"=0, "movie"=1). Worth confirming
    # this against `torch_data.node_types` when actually run.
    homo = torch_data.to_homogeneous()
    torch_norm = torch_HeteroLayerNorm(4, num_types=2, affine=False)
    torch_out = torch_norm(homo.x, homo.node_type)

    x, type_vec = _flatten_by_type(mlx_data.node_features_dict, ["user", "movie"])
    mlx_norm = HeteroLayerNormalization(4, num_types=2, affine=False)
    mlx_out = mlx_norm(x, type_vec)

    assert mx.allclose(mx.array(torch_out.detach().numpy()), mlx_out, atol=1e-5)


def test_hetero_layer_norm_affine():
    import numpy as np
    from torch_geometric.nn import HeteroLayerNorm as torch_HeteroLayerNorm

    user_features = np.random.uniform(0, 1, (5, 4)).astype(np.float32)
    movie_features = np.random.uniform(0, 1, (7, 4)).astype(np.float32)
    edge_index = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    weight_np = np.random.uniform(0.5, 2.0, (2, 4)).astype(np.float32)
    bias_np = np.random.uniform(-1.0, 1.0, (2, 4)).astype(np.float32)

    torch_data, mlx_data = _build_graphs(user_features, movie_features, edge_index)
    homo = torch_data.to_homogeneous()

    torch_norm = torch_HeteroLayerNorm(4, num_types=2, affine=True)
    torch_norm.weight.data = torch.tensor(weight_np)
    torch_norm.bias.data = torch.tensor(bias_np)
    torch_out = torch_norm(homo.x, homo.node_type)

    x, type_vec = _flatten_by_type(mlx_data.node_features_dict, ["user", "movie"])
    mlx_norm = HeteroLayerNormalization(4, num_types=2, affine=True)
    mlx_norm.weight = mx.array(weight_np)
    mlx_norm.bias = mx.array(bias_np)
    mlx_out = mlx_norm(x, type_vec)

    assert mx.allclose(mx.array(torch_out.detach().numpy()), mlx_out, atol=1e-5)
