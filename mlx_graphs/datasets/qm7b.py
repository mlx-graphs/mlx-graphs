import os

import mlx.core as mx
from dataset import DEFAULT_BASE_DIR, Dataset
from utils import check_sha1, download

from mlx_graphs.data import GraphData
from mlx_graphs.utils.transformations import to_sparse_adjacency_matrix

try:
    import scipy as sp
except ImportError:
    raise ImportError("scipy is required to download and process the raw data")


class QM7bDataset(Dataset):
    _url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7b.mat"
    _sha1_str = "4102c744bb9d6fd7b40ac67a300e49cd87e28392"

    def __init__(self, base_dir: str = DEFAULT_BASE_DIR):
        super().__init__(name="qm7b", base_dir=base_dir)

    def download(self):
        assert self.raw_path is not None, "Unable to access/create the self.raw_path"
        file_path = os.path.join(self.raw_path, self.name + ".mat")
        download(self._url, path=file_path)
        if not check_sha1(file_path, self._sha1_str):
            raise UserWarning(
                "File {} is downloaded but the content hash does not match."
                "The repo may be outdated or download may be incomplete. "
                "Otherwise you can create an issue for it.".format(self.name)
            )

    def process(self):
        assert self.raw_path is not None, "Unable to access/create the self.raw_path"
        mat_path = os.path.join(self.raw_path, self.name + ".mat")
        data = sp.io.loadmat(mat_path)
        labels = mx.array(data["T"].tolist())
        features = data["X"]
        num_graphs = labels.shape[0]
        graphs = []
        for i in range(num_graphs):
            edge_index, edge_features = to_sparse_adjacency_matrix(features[i])
            graphs.append(
                GraphData(
                    edge_index=edge_index,
                    edge_features=edge_features,
                    graph_labels=labels[i],
                )
            )
        self.graphs = graphs

        # # currently skipping saving as mlx arrays don't work with pickle
        # assert (
        #     self.processed_path is not None
        # ), "Unable to access/create the self.processed_path"
        # save_graphs(self.processed_path, self.graphs)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


a = QM7bDataset()
