import os
import os.path as osp
from typing import Callable, List, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import HeteroGraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download, extract_archive


class MovieLens100K(Dataset):
    file_id = "1ggYlYf2_kTyi7oF9g07oTNn3VDhjl7so"

    def __init__(
        self,
        base_dir: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(
            name="MovieLens100K",
            base_dir=base_dir,
            transform=transform,
            pre_transform=pre_transform,
        )

    @property
    def raw_path(self) -> str:
        return f"{super(self.__class__, self).raw_path}"

    @property
    def raw_file_names(self) -> List[str]:
        return ["u.item", "u.user", "u1.base", "u1.test"]

    def download(self):
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        self.data_path = "ml-100k"
        path = download(url, self.raw_path)
        extract_archive(path, self.raw_path)
        os.remove(path)

    def process(self):
        data = HeteroGraphData(edge_index_dict={})

        self.raw_paths = [
            osp.join(f"{self.raw_path}/{self.data_path}", self.raw_file_names[i])
            for i in range(len(self.raw_file_names))
        ]

        df = np.loadtxt(
            self.raw_paths[0], delimiter="|", dtype=str, encoding="ISO-8859-1"
        )
        movie_mapping = {idx: i for i, idx in enumerate(df[:, 0].astype(int))}
        x = df[:, 6:].astype(float)
        data.node_features_dict = {"movie": mx.array(x, dtype=mx.float32)}

        df = np.loadtxt(
            self.raw_paths[1], delimiter="|", dtype=str, encoding="ISO-8859-1"
        )

        user_mapping = {idx: i for i, idx in enumerate(df[:, 0].astype(int))}

        age = df[:, 1].astype(float) / df[:, 1].astype(float).max()
        age = mx.array(age, mx.float32).reshape(-1, 1)

        gender = np.eye(np.unique(df[:, 2]).size)[
            np.unique(df[:, 2], return_inverse=True)[1]
        ]
        gender = mx.array(gender, mx.float32)

        occupation = np.eye(np.unique(df[:, 3]).size)[
            np.unique(df[:, 3], return_inverse=True)[1]
        ]
        occupation = mx.array(occupation, mx.float32)

        data.node_features_dict["user"] = mx.concatenate(
            [age, gender, occupation], axis=-1
        )

        df = np.loadtxt(self.raw_paths[2], delimiter="\t")

        src = [user_mapping[idx] for idx in df[:, 0]]
        dst = [movie_mapping[idx] for idx in df[:, 1]]
        edge_index = mx.array([src, dst], mx.int64)
        data.edge_index_dict[("user", "rates", "movie")] = edge_index

        rating = mx.array(df[:, 2], dtype=mx.int64)
        data.user_rates_movie_rating = rating

        time = mx.array(df[:, 3], dtype=mx.int64)
        data.user_rates_movie_time = time

        data.edge_index_dict[("movie", "rated_by", "user")] = mx.array(
            [dst, src], mx.int64
        )
        data.movie_rated_by_user_time = time

        df = np.loadtxt(self.raw_paths[3], delimiter="\t")

        src = [user_mapping[idx] for idx in df[:, 0]]
        dst = [movie_mapping[idx] for idx in df[:, 1]]
        edge_label_index = mx.array([src, dst], mx.int64)
        data.edge_labels_index_dict = {}
        data.edge_labels_index_dict[("user", "rates", "movie")] = edge_label_index

        edge_label = mx.array(df[:, 2], mx.float32)
        data.edge_labels_dict = {}
        data.edge_labels_dict[("user", "rates", "movie")] = edge_label

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.graphs = [data]
