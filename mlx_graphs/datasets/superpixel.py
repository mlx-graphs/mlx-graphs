import os
import pickle
from typing import Literal, Optional, get_args

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_graphs.data.data import GraphData
from mlx_graphs.datasets import Dataset
from mlx_graphs.datasets.utils import download, extract_archive
from mlx_graphs.utils import pairwise_distances

SUPERPIXEL_NAMES = Literal["MNIST", "CIFAR10"]
SUPERPIXEL_SPLITS = Literal["train", "test"]
SUPERPIXEL_PKL_FILES = {
    "MNIST": "mnist_75sp",
    "CIFAR10": "cifar10_150sp",
}


def sigma(distances: mx.array, k: int = 8) -> mx.array:
    """
    Computes the scale parameter sigma defined as the averaged distance xk of the k
    nearest neighbors for each node in the distances matrix.

    See Equation (47) in `<http://arxiv.org/abs/2003.00982>`_ for more details.

    Args:
        distances: array of pairwise distances between points
        k: nearest neighbors to consider

    Returns:
        The value of sigma for each node.
    """
    num_nodes = distances.shape[0]

    if k > num_nodes:
        # handle graphs with num_nodes less than k
        sigma = mx.array([1] * num_nodes).reshape(num_nodes, 1)
    else:
        # get knns for each node
        knns = mx.partition(distances, k, axis=-1)[:, : k + 1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1)) / k

    return sigma + 1e-8


def image_to_adjacency_matrix(
    coordinates: mx.array, features: mx.array, use_features: bool = True
) -> mx.array:
    """
    Computes a k-NN adjacency matrix as in Equation (47) in
    `<http://arxiv.org/abs/2003.00982>`_.

    Args:
        coordinates: coordinates of each pixel/node
        features: features of each pixel/node
        use_features: whether to use features in computing the adjacency matrix.
            Defaults to True

    Returns:
        Adjacency matrix
    """
    coord = coordinates.reshape(-1, 2)
    coord_dist = pairwise_distances(coord, coord)

    if use_features:
        features_dist = pairwise_distances(features, features)
        adjacency_matrix = mx.exp(
            -((coord_dist / sigma(coord_dist)) ** 2)
            - (features_dist / sigma(features_dist)) ** 2
        )
    else:
        adjacency_matrix = mx.exp(-((coord_dist / sigma(coord_dist)) ** 2))

    # convert to symmetric matrix without self-loops
    adjacency_matrix = 0.5 * (adjacency_matrix + adjacency_matrix.T)
    for i in range(adjacency_matrix.shape[0]):
        adjacency_matrix[i, i] = 0
    return adjacency_matrix


def adjacency_matrix_to_knn_edges(
    adjacency_matrix: mx.array, k: int = 9
) -> tuple[mx.array, mx.array]:
    """
    Compute list of knn nodes per node (and features)

    Args:
        adjacency_matrix: the adjacency matrix
        k: the number of nearest neighbors

    Returns:
        list of knns for each node and list of their features
    """
    num_nodes = adjacency_matrix.shape[0]
    new_kth = num_nodes - k

    np_adj_mat = np.array(adjacency_matrix, copy=False)
    if num_nodes > k:
        # we need to use numpy's argpartition and partition because when there are
        # equal elements in a partition, their order is random with mlx, while when
        # using `numpy` they're ordered based on their index in the original array.
        # This turns out to be a significant problem with highly connected graphs.
        knns = mx.array(
            np.argpartition(np_adj_mat, new_kth - 1, axis=-1)[:, new_kth:-1].tolist()
        )
        knn_values = mx.array(
            np.partition(np_adj_mat, new_kth - 1, axis=-1)[:, new_kth:-1].tolist()
        )
    else:
        # for graphs with less than k nodes the resulting graph will be fully connected.
        knns = mx.repeat(mx.arange(num_nodes), num_nodes).reshape(num_nodes, num_nodes)
        knn_values = adjacency_matrix

        # remove self loops
        if num_nodes != 1:
            knn_values = mx.array(
                np_adj_mat[knns != np.arange(num_nodes)[:, None]]
                .reshape(num_nodes, -1)
                .tolist()
            )
            knns = mx.array(
                np.array(knns, copy=False)[knns != np.arange(num_nodes)[:, None]]
                .reshape(num_nodes, -1)
                .tolist()
            )
    return knns, knn_values


class SuperPixelDataset(Dataset):
    """
    MNIST and CIFAR10 superpixel datasets for graph classification tasks
    converted fromt the original MINST and CIFAR10 images.

    The datasets were introduced in `<http://arxiv.org/abs/2003.00982>`_.

    Args:
        name: name of the selected dataset
        split: split of the dataset to load
        use_features: if True, the adjacency matrix is computed from superpixels
            locations and features. If False, only from superpixels locations.
            Defaults to False.
        base_dir: directory where to store the datasets

    """

    _url = "https://www.dropbox.com/s/y2qwa77a0fxem47/superpixels.zip?dl=1"

    def __init__(
        self,
        name: SUPERPIXEL_NAMES,
        split: SUPERPIXEL_SPLITS,
        use_features: bool = False,
        base_dir: Optional[str] = None,
    ):
        assert name in get_args(SUPERPIXEL_NAMES), "Invalid dataset name"
        assert split in get_args(SUPERPIXEL_SPLITS), "Invalid split specified"
        self.split = split
        self.use_features = use_features
        super().__init__(name=name, base_dir=base_dir)

    @property
    def _img_size(self):
        """Size of dataset image."""
        if self.name == "MNIST":
            return 28
        return 32

    @property
    def processed_path(self) -> str:
        # processed path includes split and use_features
        return os.path.join(
            f"{super(self.__class__, self).processed_path}",
            self.split,
            "use_features_" + str(self.use_features),
        )

    def download(self):
        file_path = os.path.join(self.raw_path, "superpixels.zip")
        path = download(self._url, path=file_path)
        extract_archive(path, self.raw_path, overwrite=True)

    def process(self):
        with open(
            os.path.join(
                self.raw_path,
                "superpixels",
                f"{SUPERPIXEL_PKL_FILES[self.name]}_{self.split}.pkl",
            ),
            "rb",
        ) as f:
            labels, data = pickle.load(f)
            labels = mx.array([labels.tolist()]).T

        for idx, sample in enumerate(
            tqdm(data, desc=f"Processing {self.name} {self.split} dataset")
        ):
            mean_px, coord = sample[:2]
            mean_px = mx.array(mean_px.tolist())
            coord = mx.array(coord.tolist()) / self._img_size

            if self.use_features:
                adjacency_matrix = image_to_adjacency_matrix(
                    coord, mean_px
                )  # using super-pixel locations + features
            else:
                adjacency_matrix = image_to_adjacency_matrix(
                    coord, mean_px, False
                )  # using only super-pixel locations

            edges_list, edges_values = adjacency_matrix_to_knn_edges(adjacency_matrix)

            num_nodes = adjacency_matrix.shape[0]
            mean_px = mean_px.reshape(num_nodes, -1)
            coord = coord.reshape(num_nodes, 2)
            node_features = mx.concatenate((mean_px, coord), axis=1)

            src_nodes = []
            dst_nodes = []
            # TODO: use mlx once indexing by bool is supported
            for src, dsts in enumerate(np.array(edges_list, copy=False)):
                if num_nodes == 1:
                    src_nodes.append(src)
                    dst_nodes.append(dsts)
                else:
                    dsts = dsts[dsts != src].tolist()
                    srcs = [src] * len(dsts)
                    src_nodes.extend(srcs)
                    dst_nodes.extend(dsts)

            edge_index = mx.stack([mx.array(src_nodes), mx.array(dst_nodes)])
            edge_features = mx.expand_dims(edges_values.reshape(-1), 1)

            self.graphs.append(
                GraphData(
                    edge_index=edge_index,
                    node_features=node_features,
                    edge_features=edge_features,
                    graph_labels=labels[idx],
                )
            )
