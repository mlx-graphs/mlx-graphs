import glob
import os
from typing import Callable, List, Literal, Optional, Tuple

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_graphs.data.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download, extract_archive

MODELNET_NAMES = Literal["10", "40"]


def parse_off(lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Parses the lines of an OFF (Object File Format) mesh file.

    Quadrilaterals are split into two triangles, as in PyG.

    Args:
        lines: The lines of the file.

    Returns:
        A tuple ``(pos, face)`` with the vertex positions of shape
        ``[num_nodes, 3]`` (float32) and the triangles of shape ``[3,
        num_faces]`` (int64).
    """
    lines = [line for line in lines if line.strip()]
    # Some ModelNet files have no line break after "OFF".
    header = lines[0].strip()
    if header == "OFF":
        lines = lines[1:]
    else:
        lines = [header[3:]] + lines[1:]
    num_nodes, num_faces = (int(v) for v in lines[0].split()[:2])
    pos = np.array(
        " ".join(lines[1 : 1 + num_nodes]).split(), dtype=np.float32
    ).reshape(num_nodes, -1)[:, :3]
    triangles, quads = [], []
    for line in lines[1 + num_nodes : 1 + num_nodes + num_faces]:
        values = [int(v) for v in line.split()]
        if values[0] == 3:
            triangles.append(values[1:4])
        elif values[0] == 4:
            quads.append(values[1:5])
    face = np.array(triangles, dtype=np.int64).reshape(-1, 3)
    if quads:
        rect = np.array(quads, dtype=np.int64)
        face = np.concatenate([face, rect[:, [0, 1, 2]], rect[:, [0, 2, 3]]])
    return pos, np.ascontiguousarray(face.T)


def read_off(path: str) -> GraphData:
    """Reads an OFF mesh file into a ``GraphData`` with ``pos`` and ``face``
    (and no edges).

    Args:
        path: Path of the file.

    Returns:
        The mesh.
    """
    with open(path) as f:
        pos, face = parse_off(f.read().split("\n"))
    return GraphData(
        edge_index=mx.zeros((2, 0), dtype=mx.int32),
        pos=mx.array(pos),
        face=mx.array(face.astype(np.int32)),
    )


class ModelNet(Dataset):
    r"""The ModelNet10/40 datasets from the `"3D ShapeNets: A Deep
    Representation for Volumetric Shapes"
    <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
    containing CAD models of 10 and 40 categories, respectively.

    Each graph is a mesh with vertex positions ``pos`` of shape ``[N, 3]``,
    triangles ``face`` of shape ``[3, num_faces]``, no edges, and its
    category in ``graph_labels``. Point clouds are usually sampled from the
    meshes with :class:`~mlx_graphs.transforms.SamplePoints`.

    .. note::

        The raw archives are large (ModelNet10: 0.5 GB, ModelNet40: 2 GB), and
        processing them takes a few minutes.

    Args:
        name: ``"10"`` for ModelNet10 or ``"40"`` for ModelNet40. Default:
            ``"10"``
        train: Whether to load the training or the test split. Default:
            ``True``
        base_dir: Directory where to store the dataset. Default: ``None``
            (``.mlx_graphs_data/`` in the current directory).
        pre_transform: A transform applied to each graph once, before saving
            the processed dataset. Default: ``None``
        transform: A transform applied to each graph when accessed. Default:
            ``None``

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import ModelNet
        from mlx_graphs.transforms import NormalizeScale, SamplePoints

        dataset = ModelNet(
            "10",
            train=True,
            pre_transform=NormalizeScale(),
            transform=SamplePoints(1024),
        )
        >>> dataset
        ModelNet10(num_graphs=3991)
        >>> dataset[0]
        GraphData(
            edge_index(shape=[2, 0], int32)
            graph_labels(shape=[1], int32)
            pos(shape=[1024, 3], float32))
    """

    urls = {
        "10": "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        "40": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
    }

    def __init__(
        self,
        name: MODELNET_NAMES = "10",
        train: bool = True,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        name = str(name)
        if name not in ("10", "40"):
            raise ValueError(f"name must be '10' or '40', got {name!r}")
        self.modelnet_name = name
        self.train = train
        super().__init__(
            name=f"ModelNet{name}",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def processed_path(self) -> str:
        split = "train" if self.train else "test"
        return os.path.join(super().processed_path, split)

    @property
    def _raw_folder(self) -> str:
        return os.path.join(self.raw_path, f"ModelNet{self.modelnet_name}")

    @property
    def categories(self) -> List[str]:
        """The category names, sorted (the label of a shape is the index of
        its category)."""
        folders = glob.glob(os.path.join(self._raw_folder, "*", ""))
        return sorted(os.path.basename(os.path.normpath(f)) for f in folders)

    def download(self):
        url = self.urls[self.modelnet_name]
        path = download(url, path=os.path.join(self.raw_path, url.split("/")[-1]))
        extract_archive(path, self.raw_path, overwrite=True)
        os.remove(path)

    def process(self):
        split = "train" if self.train else "test"
        paths = []
        for target, category in enumerate(self.categories):
            folder = os.path.join(self._raw_folder, category, split)
            for path in sorted(glob.glob(os.path.join(folder, f"{category}_*.off"))):
                paths.append((target, path))
        for target, path in tqdm(paths, desc=f"Processing {self.name} {split}"):
            graph = read_off(path)
            graph.graph_labels = mx.array([target], dtype=mx.int32)
            self.graphs.append(graph)

    def __repr__(self) -> str:
        return f"{self.name}(num_graphs={len(self)})"
