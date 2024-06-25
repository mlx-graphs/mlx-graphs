import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import HeteroGraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download, extract_archive


class DBLP(Dataset):
    """
    A subset of the DBLP computer science bibliography website, as
    collected in the `"MAGNN: Metapath Aggregated Graph Neural Network for
    Heterogeneous Graph Embedding" <https://arxiv.org/abs/2002.01680>`_ paper.
    DBLP is a heterogeneous graph containing four types of entities - authors
    (4,057 nodes), papers (14,328 nodes), terms (7,723 nodes), and conferences
    (20 nodes).
    The authors are divided into four research areas (database, data mining,
    artificial intelligence, information retrieval).
    Each author is described by a bag-of-words representation of their paper
    keywords.

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
            name="DBLP",
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
        url = "https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=1"
        path = download(url=url, path=self.raw_path)
        new_path = path.split("?")[-2]
        os.rename(path, new_path)
        extract_archive(new_path, self.raw_path)
        os.remove(new_path)

    def process(self):
        try:
            import scipy.sparse as sp
        except ImportError:
            raise ImportError("scipy is required to download and process the raw data")
        node_types = ["author", "paper", "term", "conference"]
        node_features_dict = {}
        for i, node_type in enumerate(node_types[:2]):
            nodes = sp.load_npz(osp.join(self.raw_path, f"features_{i}.npz"))
            node_features_dict[node_type] = mx.array(nodes.todense())

        term = np.load(osp.join(self.raw_path, "features_2.npy"))
        node_features_dict["term"] = mx.array(term).astype(mx.float32)

        node_type_idx = np.load(osp.join(self.raw_path, "node_types.npy"))
        node_type_idx = mx.array(node_type_idx).astype(mx.int32)

        """
        Conference nodes don't have features and hence adding
        it explicitly to a dictionary will not make sense.
        Either override the property in the class or set attribute separately
        for conference
        """
        conference_nodes = int((node_type_idx == 3).sum().item())
        print(conference_nodes)

        node_labels_dict = {}
        y = np.load(osp.join(self.raw_path, "labels.npy"))
        node_labels_dict["author"] = mx.array(y)

        data = HeteroGraphData(
            edge_index_dict={},
            node_features_dict=node_features_dict,
            edge_features_dict={},
            node_labels_dict=node_labels_dict,
        )

        split = np.load(osp.join(self.raw_path, "train_val_test_idx.npz"))
        for name in ["train", "val", "test"]:
            idx = split[f"{name}_idx"]
            idx = mx.array(idx, dtype=mx.int64)
            mask = mx.zeros(data.num_nodes["author"], dtype=mx.bool_)
            mask[idx] = True
            setattr(data, f"author_{name}_mask", mask)

        s = {}
        N_a = data.num_nodes["author"]
        N_p = data.num_nodes["paper"]
        N_t = data.num_nodes["term"]
        N_c = conference_nodes
        s["author"] = (0, N_a)
        s["paper"] = (N_a, N_a + N_p)
        s["term"] = (N_a + N_p, N_a + N_p + N_t)
        s["conference"] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)

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
