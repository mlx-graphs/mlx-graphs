import json
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.datasets import ModelNet, ShapeNet
from mlx_graphs.datasets.modelnet import parse_off
from mlx_graphs.loaders import Dataloader
from mlx_graphs.transforms import NormalizeScale, SamplePoints

OFF = """OFF
5 3 0
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
0.5 0.5 1.5
3 0 1 4
4 0 1 2 3
3 2 3 4
"""


def test_parse_off_matches_pyg():
    from torch_geometric.io.off import parse_off as pyg_parse_off

    for text in (OFF, OFF.replace("OFF\n", "OFF")):  # missing line break
        pos, face = parse_off(text.split("\n"))
        ref = pyg_parse_off(text.split("\n")[:-1])
        np.testing.assert_allclose(pos, ref.pos.numpy())
        np.testing.assert_array_equal(face, ref.face.numpy())
        assert face.shape == (3, 4)


def write_modelnet(root, name="10"):
    """A tiny raw ModelNet folder: 2 categories, 3 + 2 train / 1 + 1 test."""
    folder = os.path.join(root, f"ModelNet{name}", "raw", f"ModelNet{name}")
    for category, counts in (("chair", (3, 1)), ("bed", (2, 1))):
        for split, count in zip(("train", "test"), counts):
            os.makedirs(os.path.join(folder, category, split))
            for i in range(count):
                path = os.path.join(folder, category, split, f"{category}_{i:04d}.off")
                with open(path, "w") as f:
                    f.write(OFF)


def test_modelnet_from_raw_files(tmp_path):
    write_modelnet(str(tmp_path))
    train = ModelNet("10", train=True, base_dir=str(tmp_path))
    test = ModelNet("10", train=False, base_dir=str(tmp_path))
    assert len(train) == 5 and len(test) == 2
    assert train.categories == ["bed", "chair"]
    assert [g.graph_labels.item() for g in train] == [0, 0, 1, 1, 1]
    assert train.num_classes == 2
    graph = train[0]
    assert graph.pos.shape == (5, 3) and graph.face.shape == (3, 4)
    assert graph.num_nodes == 5 and graph.edge_index.shape == (2, 0)
    assert repr(train) == "ModelNet10(num_graphs=5)"

    # transforms, and batches of sampled point clouds
    mx.random.seed(0)
    sampled = ModelNet(
        "10", base_dir=str(tmp_path), transform=SamplePoints(16)
    )  # reloads the processed data
    batch = next(iter(Dataloader(sampled, batch_size=4)))
    assert batch.pos.shape == (64, 3)
    assert batch.batch_indices.tolist() == sum([[i] * 16 for i in range(4)], [])
    assert batch.graph_labels.shape == (4,)

    with pytest.raises(ValueError, match="'10' or '40'"):
        ModelNet("20", base_dir=str(tmp_path))


def write_shapenet(root):
    folder = os.path.join(
        root,
        "ShapeNet",
        "raw",
        "shapenetcore_partanno_segmentation_benchmark_v0_normal",
    )
    rng = np.random.default_rng(0)
    files = {"train": [], "val": [], "test": []}
    for i, (cat, split) in enumerate(
        [
            ("02691156", "train"),
            ("02691156", "val"),
            ("03797390", "train"),
            ("03797390", "test"),
        ]
    ):
        os.makedirs(os.path.join(folder, cat), exist_ok=True)
        values = rng.normal(size=(10 + i, 7))
        values[:, -1] = rng.integers(0, 4, 10 + i) + (36 if cat == "03797390" else 0)
        np.savetxt(os.path.join(folder, cat, f"shape{i}.txt"), values)
        files[split].append(f"shape_data/{cat}/shape{i}")
    os.makedirs(os.path.join(folder, "train_test_split"))
    for split, names in files.items():
        path = os.path.join(
            folder, "train_test_split", f"shuffled_{split}_file_list.json"
        )
        with open(path, "w") as f:
            json.dump(names, f)
    return folder


def test_shapenet_from_raw_files(tmp_path):
    folder = write_shapenet(str(tmp_path))
    trainval = ShapeNet(base_dir=str(tmp_path))
    assert len(trainval) == 3
    assert [g.category.item() for g in trainval] == [0, 11, 0]  # Airplane, Mug
    graph = trainval[0]
    values = np.loadtxt(os.path.join(folder, "02691156", "shape0.txt"))
    np.testing.assert_allclose(np.array(graph.pos), values[:, :3], rtol=1e-6)
    np.testing.assert_allclose(np.array(graph.node_features), values[:, 3:6], rtol=1e-6)
    np.testing.assert_array_equal(np.array(graph.node_labels), values[:, -1])
    assert trainval.num_classes == 50
    assert trainval.y_mask.shape == (16, 50)
    assert trainval.y_mask[11].tolist() == [i in (36, 37) for i in range(50)]

    mugs = ShapeNet(
        categories="Mug", split="test", include_normals=False, base_dir=str(tmp_path)
    )
    assert len(mugs) == 1 and mugs[0].node_features is None
    assert mugs[0].category.item() == 0
    assert len(ShapeNet(split="train", base_dir=str(tmp_path))) == 2

    with pytest.raises(ValueError, match="Unknown"):
        ShapeNet(categories="Plane", base_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Split"):
        ShapeNet(split="validation", base_dir=str(tmp_path))


@pytest.mark.slow
def test_modelnet10_matches_pyg(tmp_path):
    from torch_geometric.datasets import ModelNet as PyGModelNet

    for train in (True, False):
        dataset = ModelNet("10", train, base_dir=str(tmp_path / "mxg"))
        ref = PyGModelNet(str(tmp_path / "pyg"), "10", train)
        assert len(dataset) == len(ref) == (3991 if train else 908)
        # PyG lists the files of each category in directory order: match
        # the meshes by size and label, then compare them
        ours = {}
        for g in dataset:
            key = (g.pos.shape[0], g.face.shape[1], g.graph_labels.item())
            ours.setdefault(key, []).append(g)
        for data in ref:
            key = (data.pos.shape[0], data.face.shape[1], data.y.item())
            assert any(
                np.array_equal(np.array(g.pos), data.pos.numpy())
                and np.array_equal(np.array(g.face), data.face.numpy())
                for g in ours[key]
            )


@pytest.mark.slow
def test_shapenet_matches_pyg(tmp_path):
    from torch_geometric.datasets import ShapeNet as PyGShapeNet

    dataset = ShapeNet(categories=["Mug", "Cap"], split="test", base_dir=str(tmp_path))
    # PyG's download URL is offline: reuse the raw files
    os.makedirs(tmp_path / "pyg")
    os.symlink(
        os.path.join(dataset.raw_path, ShapeNet._folder), tmp_path / "pyg" / "raw"
    )
    ref = PyGShapeNet(str(tmp_path / "pyg"), categories=["Mug", "Cap"], split="test")
    assert len(dataset) == len(ref)
    for graph, data in zip(dataset, ref):
        np.testing.assert_allclose(np.array(graph.pos), data.pos.numpy())
        np.testing.assert_allclose(np.array(graph.node_features), data.x.numpy())
        np.testing.assert_array_equal(np.array(graph.node_labels), data.y.numpy())
        assert graph.category.item() == int(data.category)


def test_normalize_scale_pre_transform(tmp_path):
    write_modelnet(str(tmp_path))
    dataset = ModelNet("10", base_dir=str(tmp_path), pre_transform=NormalizeScale())
    pos = np.array(dataset[0].pos)
    assert np.abs(pos).max() < 1 and np.abs(pos.mean(0)).max() < 1e-6
