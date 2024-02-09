import pytest

from mlx_graphs.datasets import Dataset


def test_dataset():
    # childred dataset with no implemented download and process methods
    # can't be instantiated
    class FakeDataset(Dataset):
        pass

    with pytest.raises(TypeError):
        _ = FakeDataset("fake_dataset")
