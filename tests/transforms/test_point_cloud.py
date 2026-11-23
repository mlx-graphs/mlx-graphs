"""Point cloud and mesh transforms compared with PyG's on fixed inputs (random
transforms with a degenerate range, or checked through their invariants)."""

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

from mlx_graphs.data import GraphData
from mlx_graphs.data.batch import GraphDataBatch
from mlx_graphs.transforms import (
    Cartesian,
    Center,
    Compose,
    Distance,
    FaceToEdge,
    FixedPoints,
    LinearTransformation,
    NormalizeScale,
    PointPairFeatures,
    Polar,
    RandomJitter,
    RandomRotate,
    RandomScale,
    SamplePoints,
)


def assert_close(actual, expected, tol=1e-5):
    np.testing.assert_allclose(
        np.array(actual), np.asarray(expected), rtol=tol, atol=tol
    )


def cloud(seed=0, num_nodes=30, dim=3, edges=True, edge_attr=True):
    rng = np.random.default_rng(seed)
    pos = rng.normal(loc=0.5, size=(num_nodes, dim)).astype(np.float32)
    e = rng.integers(0, num_nodes, (2, 60))
    attr = rng.normal(size=(60, 2)).astype(np.float32)
    kwargs, pyg = {}, {}
    if edges:
        kwargs["edge_index"] = mx.array(e)
        pyg["edge_index"] = torch.tensor(e)
    else:
        kwargs["edge_index"] = mx.zeros((2, 0), dtype=mx.int32)
    if edge_attr:
        kwargs["edge_features"] = mx.array(attr)
        pyg["edge_attr"] = torch.tensor(attr)
    graph = GraphData(pos=mx.array(pos), **kwargs)
    return graph, Data(pos=torch.tensor(pos), **pyg)


def mesh():
    """A unit cube (8 vertices, 12 triangles) and a pyramid with a quad."""
    pos = np.array(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float32
    )
    face = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
            [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ]
    ).T  # fmt: skip
    return pos * np.array([2.0, 1.0, 0.5], np.float32) - 0.3, face


def test_center_and_normalize_scale_match_pyg():
    graph, data = cloud()
    assert_close(Center()(graph).pos, T.Center()(data.clone()).pos)
    out = NormalizeScale()(graph)
    assert_close(out.pos, T.NormalizeScale()(data.clone()).pos)
    assert float(mx.abs(out.pos).max()) < 1
    # the input graph is unchanged
    assert_close(graph.pos, data.pos)


def test_linear_transformation_matches_pyg():
    graph, data = cloud()
    matrix = np.random.default_rng(1).normal(size=(3, 3)).astype(np.float32)
    out = LinearTransformation(mx.array(matrix))(graph)
    assert_close(out.pos, T.LinearTransformation(torch.tensor(matrix))(data).pos)
    with pytest.raises(ValueError, match="square"):
        LinearTransformation(mx.zeros((2, 3)))


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("dim", [2, 3])
def test_random_rotate_matches_pyg(axis, dim):
    graph, data = cloud(dim=dim)
    # a degenerate range gives a fixed angle
    out = RandomRotate((37.0, 37.0), axis=axis)(graph)
    expected = T.RandomRotate((37.0, 37.0), axis=axis)(data).pos
    assert_close(out.pos, expected)

    # random angles are uniform in the range, and rotations keep norms
    mx.random.seed(0)
    norms = np.linalg.norm(np.array(graph.pos), axis=1)
    for _ in range(3):
        rotated = RandomRotate(180, axis=axis)(graph).pos
        assert_close(np.linalg.norm(np.array(rotated), axis=1), norms, 1e-4)


def test_random_scale_matches_pyg():
    graph, data = cloud()
    assert_close(
        RandomScale((1.7, 1.7))(graph).pos, T.RandomScale((1.7, 1.7))(data).pos
    )
    mx.random.seed(0)
    ratios = [
        float(RandomScale((0.5, 2.0))(graph).pos[0, 0] / graph.pos[0, 0])
        for _ in range(20)
    ]
    assert min(ratios) >= 0.5 and max(ratios) <= 2.0 and len(set(ratios)) > 1


@pytest.mark.parametrize("translate", [0.1, (0.0, 0.2, 0.05)])
def test_random_jitter(translate):
    graph, data = cloud()
    mx.random.seed(0)
    delta = np.array(RandomJitter(translate)(graph).pos) - np.array(graph.pos)
    bound = np.broadcast_to(np.asarray(translate), (3,))
    assert (np.abs(delta) <= bound + 1e-6).all()
    assert np.abs(delta).max(0)[bound > 0].min() > 0.5 * bound[bound > 0].min()
    assert_close(RandomJitter(0.0)(graph).pos, T.RandomJitter(0.0)(data).pos)


@pytest.mark.parametrize(
    "transform,pyg",
    [
        (Cartesian(), T.Cartesian()),
        (Cartesian(norm=False), T.Cartesian(norm=False)),
        (
            Cartesian(max_value=4.0, interval=(-1, 1)),
            T.Cartesian(max_value=4.0, interval=(-1, 1)),
        ),
        (Cartesian(cat=False), T.Cartesian(cat=False)),
        (Distance(), T.Distance()),
        (Distance(norm=False, cat=False), T.Distance(norm=False, cat=False)),
        (
            Distance(max_value=3.0, interval=(1, 2)),
            T.Distance(max_value=3.0, interval=(1, 2)),
        ),
    ],
)
@pytest.mark.parametrize("edge_attr", [True, False])
def test_edge_pseudo_coordinates_match_pyg(transform, pyg, edge_attr):
    graph, data = cloud(edge_attr=edge_attr)
    out = transform(graph)
    expected = pyg(data)
    assert_close(out.edge_features, expected.edge_attr)
    assert_close(out.pos, graph.pos)


@pytest.mark.parametrize(
    "kwargs", [{}, dict(norm=False), dict(max_value=2.0, cat=False)]
)
def test_polar_matches_pyg(kwargs):
    graph, data = cloud(dim=2)
    assert_close(
        Polar(**kwargs)(graph).edge_features, T.Polar(**kwargs)(data).edge_attr
    )
    with pytest.raises(ValueError, match="2D"):
        Polar()(cloud(dim=3)[0])


@pytest.mark.parametrize("cat", [True, False])
def test_point_pair_features_match_pyg(cat):
    graph, data = cloud()
    normal = np.random.default_rng(2).normal(size=(30, 3)).astype(np.float32)
    graph.norm, data.norm = mx.array(normal), torch.tensor(normal)
    out = PointPairFeatures(cat=cat)(graph)
    assert_close(out.edge_features, T.PointPairFeatures(cat=cat)(data).edge_attr, 1e-4)
    # `normal` (as set by SamplePoints) is used without `norm`
    graph2, _ = cloud()
    graph2.normal = mx.array(normal)
    assert_close(PointPairFeatures(cat=cat)(graph2).edge_features, out.edge_features)


@pytest.mark.parametrize("quads", [False, True])
def test_face_to_edge_matches_pyg(quads):
    pos, face = mesh()
    if quads:
        face = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6]]).T
    graph = GraphData(
        mx.zeros((2, 0), dtype=mx.int32), pos=mx.array(pos), face=mx.array(face)
    )
    data = Data(pos=torch.tensor(pos), face=torch.tensor(face))
    out = FaceToEdge()(graph)
    expected = T.FaceToEdge()(data)
    np.testing.assert_array_equal(np.array(out.edge_index), expected.edge_index.numpy())
    assert "face" not in out.to_dict() and "face" in graph.to_dict()
    assert "face" in FaceToEdge(remove_faces=False)(graph).to_dict()


def test_sample_points_on_the_mesh():
    pos, face = mesh()
    graph = GraphData(
        mx.zeros((2, 0), dtype=mx.int32), pos=mx.array(pos), face=mx.array(face)
    )
    mx.random.seed(0)
    out = SamplePoints(2000, include_normals=True)(graph)
    assert out.pos.shape == (2000, 3) and out.num_nodes == 2000
    assert "face" not in out.to_dict()
    p = np.array(out.pos)
    lo, hi = pos.min(0), pos.max(0)
    # every point is on a face of the box
    on_face = (np.abs(p - lo) < 1e-5) | (np.abs(p - hi) < 1e-5)
    assert on_face.any(1).all() and (p >= lo - 1e-5).all() and (p <= hi + 1e-5).all()
    # normals are unit vectors orthogonal to the face of the point
    n = np.array(out.normal)
    assert_close(np.linalg.norm(n, axis=1), np.ones(2000), 1e-5)
    axis = np.abs(n).argmax(1)
    assert on_face[np.arange(2000), axis].all()
    # faces are sampled proportionally to their area (as in PyG): the box
    # is 2 x 1 x 0.5, so the faces orthogonal to x hold 0.5 / 3.5 of the area
    frac = (axis == 0).mean()
    assert abs(frac - 0.5 / 3.5) < 0.03

    # PyG gives the same distribution
    torch.manual_seed(0)
    ref = T.SamplePoints(2000, include_normals=True)(
        Data(pos=torch.tensor(pos), face=torch.tensor(face))
    )
    ref_frac = (ref.normal.abs().argmax(1) == 0).float().mean().item()
    assert abs(frac - ref_frac) < 0.04
    assert "face" in SamplePoints(10, remove_faces=False)(graph).to_dict()


@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(replace=False), dict(replace=False, allow_duplicates=True)],
)
@pytest.mark.parametrize("num", [10, 50])
def test_fixed_points(kwargs, num):
    graph, _ = cloud()
    graph.node_features = mx.arange(30).reshape(30, 1).astype(mx.float32)
    graph.graph_labels = mx.array([3])
    mx.random.seed(0)
    out = FixedPoints(num, **kwargs)(graph)
    expected_num = 30 if (num == 50 and kwargs == dict(replace=False)) else num
    assert out.num_nodes == expected_num
    idx = np.array(out.node_features).reshape(-1).astype(int)
    assert idx.shape == (expected_num,)
    assert_close(out.pos, np.array(graph.pos)[idx])
    # edges and graph-level attributes are kept
    assert out.edge_index.shape == graph.edge_index.shape
    assert out.edge_features.shape == graph.edge_features.shape
    assert out.graph_labels.tolist() == [3]
    if kwargs.get("replace") is False:
        if kwargs.get("allow_duplicates"):
            assert len(set(idx[:30].tolist())) == min(30, expected_num)
        else:
            assert len(set(idx.tolist())) == expected_num


def test_point_cloud_batching_and_compose():
    """Point clouds without node features or edges get their number of nodes
    from ``pos``, and their ``face`` is offset in batches."""
    pos, face = mesh()
    graphs = [
        GraphData(
            mx.zeros((2, 0), dtype=mx.int32), pos=mx.array(pos), face=mx.array(face)
        ),
        GraphData(
            mx.zeros((2, 0), dtype=mx.int32),
            pos=mx.array(pos[:5]),
            face=mx.array(face[:, :2]),
        ),
    ]
    assert graphs[0].num_nodes == 8
    batch = GraphDataBatch(graphs)
    assert batch.num_nodes == 13
    assert batch.face.shape == (3, 14)
    np.testing.assert_array_equal(np.array(batch.face[:, 12:]), face[:, :2] + 8)
    assert batch.batch_indices.tolist() == [0] * 8 + [1] * 5
    assert not batch.is_node_attr("face")
    np.testing.assert_array_equal(np.array(batch[1].face), face[:, :2])

    mx.random.seed(0)
    sampled = [Compose([NormalizeScale(), SamplePoints(64)])(g) for g in graphs]
    batch = GraphDataBatch(sampled)
    assert batch.pos.shape == (128, 3)
    assert batch.batch_indices.tolist() == [0] * 64 + [1] * 64
    # Center works on batches too (over all the points, as in PyG)
    assert_close(Center()(batch).pos.mean(0), np.zeros(3), 1e-5)
