import os
import pickle
from typing import Literal, Optional, get_args

import mlx.core as mx
from tqdm import tqdm

from mlx_graphs.data.data import GraphData
from mlx_graphs.datasets import Dataset
from mlx_graphs.datasets.utils import download, extract_archive
from mlx_graphs.utils import (
    pairwise_distances,
    remove_self_loops,
    to_sparse_adjacency_matrix,
)

SUPERPIXEL_NAMES = Literal["MNIST", "CIFAR10"]
SUPERPIXEL_SPLITS = Literal["train", "test"]
SUPERPIXEL_PKL_FILES = {
    "MNIST": "mnist_75sp",
    "CIFAR10": "cifar10_150sp",
}


def sigma(distances, k: int = 8) -> mx.array:
    """
    Computes the scale parameter sigma defined as the averaged distance xk of the k
    nearest neighbors for each node in the distances matrix.

    See Equation (47) in `<http://arxiv.org/abs/2003.00982>`_ for more details.

    Args:
        distances:
        k: nearest neighbors to consider

    Returns:
        The value of sigma for each node.
    """
    num_nodes = distances.shape[0]

    # Compute sigma and reshape.
    if k > num_nodes:
        # Handling for graphs with num_nodes less than kth.
        sigma = mx.array([1] * num_nodes).reshape(num_nodes, 1)
    else:
        # Get k-nearest neighbors for each node.
        knns = mx.partition(distances, k, axis=-1)[:, : k + 1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1)) / k

    return sigma + 1e-8


def compute_adjacency_matrix_images(
    coordinates: mx.array, features: mx.array, use_feat: bool = True
) -> mx.array:
    """
    Computes a k-NN adjacency matrix as in Equation (47) in
    `<http://arxiv.org/abs/2003.00982>`_.
    """
    coord = coordinates.reshape(-1, 2)
    coord_dist = pairwise_distances(coord, coord)

    if use_feat:
        features_dist = pairwise_distances(features, features)
        adjacency_matrix = mx.exp(
            -((coord_dist / sigma(coord_dist)) ** 2)
            - (features_dist / sigma(features_dist)) ** 2
        )
    else:
        adjacency_matrix = mx.exp(-((coord_dist / sigma(coord_dist)) ** 2))

    # Convert to symmetric matrix.
    adjacency_matrix = 0.5 * (adjacency_matrix + adjacency_matrix.T)
    return adjacency_matrix


class SuperPixelDataset(Dataset):
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
            labels = mx.array(labels.tolist())

        for idx, sample in enumerate(
            tqdm(data, desc=f"Processing {self.name} {self.split} dataset")
        ):
            mean_px, coord = sample[:2]
            mean_px = mx.array(mean_px.tolist())
            coord = mx.array(coord.tolist()) / self._img_size

            if self.use_features:
                adjacency_matrix = compute_adjacency_matrix_images(
                    coord, mean_px
                )  # using super-pixel locations + features
            else:
                adjacency_matrix = compute_adjacency_matrix_images(
                    coord, mean_px, False
                )  # using only super-pixel locations
            edge_index, edge_features = to_sparse_adjacency_matrix(adjacency_matrix)
            num_nodes = adjacency_matrix.shape[0]
            if num_nodes > 1:
                edge_index, edge_features = remove_self_loops(edge_index, edge_features)

            mean_px = mean_px.reshape(num_nodes, -1)
            coord = coord.reshape(num_nodes, 2)
            node_features = mx.concatenate((mean_px, coord), axis=1)
            self.graphs.append(
                GraphData(
                    edge_index=edge_index,
                    node_features=node_features,
                    edge_features=edge_features,
                    graph_labels=labels[idx],
                )
            )
