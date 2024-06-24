import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import HeteroGraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download, extract_archive


class IMDB(Dataset):
    """
    A subset of the Internet Movie Database (IMDB), as collected in the
    `"MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph
    Embedding" <https://arxiv.org/abs/2002.01680>`_ paper.
    IMDB is a heterogeneous graph containing three types of entities - movies
    (4,278 nodes), actors (5,257 nodes), and directors (2,081 nodes).
    The movies are divided into three classes (action, comedy, drama) according
    to their genre.
    Movie features correspond to elements of a bag-of-words representation of
    its plot keywords.

    Args:
        base_dir (str): directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`HeteroGraphData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an `HeteroGraphData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(
        self,
        base_dir: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(
            name="IMDB",
            base_dir=base_dir,
            transform=transform,
            pre_transform=pre_transform,
        )

    @property
    def raw_path(self) -> str:
        return f"{super(self.__class__, self).raw_path}"

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "adjM.npz",
            "features_0.npz",
            "features_1.npz",
            "features_2.npz",
            "labels.npy",
            "train_val_test_idx.npz",
        ]

    def download(self):
        url = "https://www.dropbox.com/s/g0btk9ctr1es39x/IMDB_processed.zip?dl=1"
        path = download(url=url, path=self.raw_path)
        new_path = path.split("?")[-2]
        os.rename(path, new_path)
        print("path is ", path)
        extract_archive(new_path, self.raw_path)
        os.remove(new_path)

    def process(self):
        try:
            import scipy.sparse as sp
        except ImportError:
            raise ImportError("scipy is required to download and process the raw data")
        data = HeteroGraphData(
            edge_index_dict={},
            node_features_dict={},
            edge_features_dict={},
            node_labels_dict={},
        )
        node_types = ["movie", "director", "actor"]
        for i, node_type in enumerate(node_types):
            nodes = sp.load_npz(osp.join(self.raw_path, f"features_{i}.npz"))
            data.node_features_dict[node_type] = mx.array(nodes.todense())

        y = np.load(osp.join(self.raw_path, "labels.npy"))
        data.node_labels_dict["movies"] = mx.array(y)

        split = np.load(osp.join(self.raw_path, "train_val_test_idx.npz"))
        for name in ["train", "val", "test"]:
            idx = split[f"{name}_idx"]
            idx = mx.array(idx, dtype=mx.int64)
            mask = mx.zeros(data.num_nodes["movie"], dtype=mx.bool_)
            mask[idx] = True
            setattr(data, f"movie_{name}_mask", mask)
        s = {}
        N_m = data.num_nodes["movie"]
        N_d = data.num_nodes["director"]
        N_a = data.num_nodes["actor"]
        s["movie"] = (0, N_m)
        s["director"] = (N_m, N_m + N_d)
        s["actor"] = (N_m + N_d, N_m + N_d + N_a)

        A = sp.load_npz(osp.join(self.raw_path, "adjM.npz"))
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0] : s[src][1], s[dst][0] : s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = mx.array(A_sub.row, dtype=mx.int64)
                col = mx.array(A_sub.col, dtype=mx.int64)
                data.edge_index_dict[(src, "to", dst)] = mx.stack([row, col], axis=0)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.graphs = [data]
