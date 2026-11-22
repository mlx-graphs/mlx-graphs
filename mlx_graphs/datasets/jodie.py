import os
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data.temporal import TemporalData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download


class JODIEDataset(Dataset):
    """The temporal interaction graphs of the `"JODIE: Predicting Dynamic
    Embedding Trajectory in Temporal Interaction Networks"
    <https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf>`_ paper, as PyG's
    ``torch_geometric.datasets.JODIEDataset``.

    Every dataset is a single :class:`~mlx_graphs.data.TemporalData` stream of
    user-item interactions sorted by time, with ``src`` the users, ``dst`` the
    items (numbered after the users), ``t`` the timestamps (int64), ``msg``
    the interaction features and ``y`` the state change labels.

    ======================  ========  ===========  ==========  ========
    Name                    #nodes    #events      #features   #classes
    ======================  ========  ===========  ==========  ========
    ``"wikipedia"``         9,227     157,474      172         2
    ``"mooc"``              7,144     411,749      4           2
    ``"lastfm"``            1,980     1,293,103    2           1
    ======================  ========  ===========  ==========  ========

    The ``"reddit"`` dataset of PyG is not available (its raw file is 2.4 GB).

    Args:
        name: Name of the dataset, ``"wikipedia"``, ``"mooc"`` or ``"lastfm"``
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the ``TemporalData``
            before saving it to disk
        transform: A function/transform applied to the ``TemporalData`` at
            every access

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import JODIEDataset

        dataset = JODIEDataset("wikipedia")
        data = dataset[0]
        >>> TemporalData(src=[157474], dst=[157474], t=[157474],
                         msg=[157474, 172], y=[157474])
        train_data, val_data, test_data = data.train_val_test_split(
            val_ratio=0.15, test_ratio=0.15
        )

    References:
        - Kumar et al., `JODIE: Predicting Dynamic Embedding Trajectory in
          Temporal Interaction Networks
          <https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf>`_
        - `PyG JODIEDataset <https://pytorch-geometric.readthedocs.io/en/latest/
          generated/torch_geometric.datasets.JODIEDataset.html>`_
    """

    url = "https://snap.stanford.edu/jodie/{}.csv"
    names = ["wikipedia", "mooc", "lastfm"]

    def __init__(
        self,
        name: str,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        name = name.lower()
        if name not in self.names:
            raise ValueError(f"Unknown JODIE dataset '{name}', available: {self.names}")
        self.jodie_name = name
        super().__init__(
            name=f"jodie_{name}",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def raw_file_name(self) -> str:
        return f"{self.jodie_name}.csv"

    def download(self):
        download(self.url.format(self.jodie_name), path=self.raw_path)

    def process(self):
        path = os.path.join(self.raw_path, self.raw_file_name)
        # Columns: user, item, timestamp, state label, features...
        table = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
        src = table[:, 0].astype(np.int64)
        dst = table[:, 1].astype(np.int64)
        dst += int(src.max()) + 1
        t = table[:, 2].astype(np.int64)
        y = table[:, 3].astype(np.int64)
        msg = table[:, 4:].astype(np.float32)
        del table

        data = TemporalData(
            src=mx.array(src.astype(np.int32)),
            dst=mx.array(dst.astype(np.int32)),
            t=mx.array(t),
            msg=mx.array(msg),
            y=mx.array(y.astype(np.int32)),
        )
        self.graphs = [data]

    def __repr__(self) -> str:
        return f"{self.jodie_name.capitalize()}()"
