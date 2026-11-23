import hashlib
import json
import os
from typing import Callable, List, Literal, Optional, Union

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_graphs.data.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download, extract_archive

SHAPENET_SPLITS = Literal["train", "val", "test", "trainval"]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ShapeNet(Dataset):
    r"""The ShapeNet part level segmentation dataset from the `"A Scalable
    Active Framework for Region Annotation in 3D Shape Collections"
    <http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf>`_
    paper, containing about 17,000 3D shape point clouds from 16 shape
    categories. Each category is annotated with 2 to 6 parts (50 part labels
    in total).

    Each graph is a point cloud with positions ``pos`` of shape ``[N, 3]``, the
    normals in ``node_features`` (if ``include_normals``), the part label of
    each point in ``node_labels``, its category index in ``category`` and no
    edges.

    The original download (``shapenet.cs.stanford.edu``) is offline: the
    archive is downloaded from a mirror on the Hugging Face Hub and checked
    against its SHA-256 hash.

    Args:
        categories: The category names (see :attr:`category_ids`) to include.
            Default: ``None`` (all of them).
        include_normals: Whether to store the normals as node features.
            Default: ``True``
        split: ``"train"``, ``"val"``, ``"test"`` or ``"trainval"``.
            Default: ``"trainval"``
        base_dir: Directory where to store the dataset. Default: ``None``
            (``.mlx_graphs_data/`` in the current directory).
        pre_transform: A transform applied to each graph once, before saving
            the processed dataset. Default: ``None``
        transform: A transform applied to each graph when accessed. Default:
            ``None``

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import ShapeNet

        dataset = ShapeNet(categories=["Airplane"], split="train")
        >>> dataset[0]
        GraphData(
            edge_index(shape=[2, 0], int32)
            node_features(shape=[2518, 3], float32)
            node_labels(shape=[2518], int32)
            pos(shape=[2518, 3], float32)
            category(shape=[1], int32))
    """

    urls = [
        "https://huggingface.co/datasets/cminst/ShapeNet/resolve/main/"
        "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip",
        "https://shapenet.cs.stanford.edu/media/"
        "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip",
    ]
    sha256 = "0e26411700bae2da38ee8ecc719ba4db2e6e0133486e258665952ad5dfced0fe"
    _folder = "shapenetcore_partanno_segmentation_benchmark_v0_normal"

    category_ids = {
        "Airplane": "02691156",
        "Bag": "02773838",
        "Cap": "02954340",
        "Car": "02958343",
        "Chair": "03001627",
        "Earphone": "03261776",
        "Guitar": "03467517",
        "Knife": "03624134",
        "Lamp": "03636649",
        "Laptop": "03642806",
        "Motorbike": "03790512",
        "Mug": "03797390",
        "Pistol": "03948459",
        "Rocket": "04099429",
        "Skateboard": "04225987",
        "Table": "04379243",
    }

    seg_classes = {
        "Airplane": [0, 1, 2, 3],
        "Bag": [4, 5],
        "Cap": [6, 7],
        "Car": [8, 9, 10, 11],
        "Chair": [12, 13, 14, 15],
        "Earphone": [16, 17, 18],
        "Guitar": [19, 20, 21],
        "Knife": [22, 23],
        "Lamp": [24, 25, 26, 27],
        "Laptop": [28, 29],
        "Motorbike": [30, 31, 32, 33, 34, 35],
        "Mug": [36, 37],
        "Pistol": [38, 39, 40],
        "Rocket": [41, 42, 43],
        "Skateboard": [44, 45, 46],
        "Table": [47, 48, 49],
    }

    def __init__(
        self,
        categories: Optional[Union[str, List[str]]] = None,
        include_normals: bool = True,
        split: SHAPENET_SPLITS = "trainval",
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        if categories is None:
            categories = list(self.category_ids)
        if isinstance(categories, str):
            categories = [categories]
        unknown = [c for c in categories if c not in self.category_ids]
        if unknown:
            raise ValueError(f"Unknown categories {unknown}")
        if split not in ("train", "val", "test", "trainval"):
            raise ValueError(
                f"Split {split} found, but expected either train, val, trainval "
                "or test"
            )
        self.categories = list(categories)
        self.include_normals = include_normals
        self.split = split
        super().__init__(
            name="ShapeNet",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )
        if not include_normals:
            for graph in self.graphs:
                graph.node_features = None
        mask = np.zeros((len(self.seg_classes), 50), dtype=bool)
        for i, labels in enumerate(self.seg_classes.values()):
            mask[i, labels] = True
        self.y_mask = mx.array(mask)

    @property
    def num_classes(self) -> int:
        """The number of part labels (50)."""
        return self.y_mask.shape[-1]

    @property
    def processed_path(self) -> str:
        cats = "_".join(c[:3].lower() for c in self.categories)
        return os.path.join(super().processed_path, cats, self.split)

    def download(self):
        archive = os.path.join(self.raw_path, f"{self._folder}.zip")
        error: Optional[Exception] = None
        for url in self.urls:
            try:
                download(url, path=archive, retries=2)
                break
            except Exception as e:  # try the next mirror
                error = e
        else:
            raise RuntimeError("Could not download ShapeNet") from error
        if _sha256(archive) != self.sha256:
            os.remove(archive)
            raise RuntimeError("The downloaded ShapeNet archive is corrupted")
        extract_archive(archive, self.raw_path, overwrite=True)
        os.remove(archive)

    def _file_names(self, split: str) -> List[str]:
        path = os.path.join(
            self.raw_path,
            self._folder,
            "train_test_split",
            f"shuffled_{split}_file_list.json",
        )
        with open(path) as f:
            # "shape_data/<category id>/<name>" -> "<category id>/<name>.txt"
            return [
                os.path.join(*name.split("/")[1:]) + ".txt" for name in json.load(f)
            ]

    def process(self):
        splits = ["train", "val"] if self.split == "trainval" else [self.split]
        names = [name for split in splits for name in self._file_names(split)]
        category_index = {
            self.category_ids[c]: i for i, c in enumerate(self.categories)
        }
        names = [n for n in names if n.split(os.sep)[0] in category_index]
        for name in tqdm(names, desc=f"Processing ShapeNet {self.split}"):
            with open(os.path.join(self.raw_path, self._folder, name)) as f:
                values = np.array(f.read().split(), dtype=np.float32).reshape(-1, 7)
            self.graphs.append(
                GraphData(
                    edge_index=mx.zeros((2, 0), dtype=mx.int32),
                    pos=mx.array(values[:, :3]),
                    node_features=mx.array(values[:, 3:6]),
                    node_labels=mx.array(values[:, -1].astype(np.int32)),
                    category=mx.array(
                        [category_index[name.split(os.sep)[0]]], dtype=mx.int32
                    ),
                )
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_graphs={len(self)}, "
            f"categories={self.categories})"
        )
