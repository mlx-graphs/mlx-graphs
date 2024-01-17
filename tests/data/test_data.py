from mlx_graphs.data.data import Data


def test_data_kwargs():
    data = Data(a=2)
    assert data.a == 2, "extra kwarg not assigned correctly"  # type: ignore
    assert "a" in data.to_dict(), "extra kwarg not in dict"
