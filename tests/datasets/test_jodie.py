import os

import numpy as np
import pytest

from mlx_graphs.data import TemporalData
from mlx_graphs.datasets import JODIEDataset


def test_jodie_unknown_name(tmp_path):
    with pytest.raises(ValueError):
        JODIEDataset("reddit", base_dir=tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["mooc", "lastfm", "wikipedia"])
def test_jodie_dataset(tmp_path, name):
    from torch_geometric.datasets import JODIEDataset as JODIEDataset_torch

    dataset = JODIEDataset(name, base_dir=tmp_path)
    assert len(dataset) == 1 and repr(dataset) == name.capitalize() + "()"
    data = dataset[0]
    assert isinstance(data, TemporalData)

    # PyG processes the same raw file
    pyg_raw = os.path.join(tmp_path, "pyg", name, "raw")
    os.makedirs(pyg_raw)
    os.symlink(
        os.path.join(dataset.raw_path, f"{name}.csv"),
        os.path.join(pyg_raw, f"{name}.csv"),
    )
    data_torch = JODIEDataset_torch(os.path.join(tmp_path, "pyg"), name)[0]

    assert data.num_events == data_torch.num_events
    assert data.num_nodes == data_torch.num_nodes
    for key in ("src", "dst", "t", "y"):
        assert np.array_equal(np.array(data[key]), data_torch[key].numpy()), key
    assert np.array_equal(np.array(data.msg), data_torch.msg.numpy())

    splits = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    splits_torch = data_torch.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    assert [s.num_events for s in splits] == [s.num_events for s in splits_torch]

    # the processed dataset is loaded back from disk
    again = JODIEDataset(name, base_dir=tmp_path)[0]
    assert np.array_equal(np.array(again.msg), np.array(data.msg))
