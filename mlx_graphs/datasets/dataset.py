import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from mlx_graphs.data import GraphData

# Default path for downloaded datasets is the root of mlx-graphs package
file_dir = os.path.dirname(os.path.realpath(__file__))
DEFAULT_BASE_DIR = os.path.join(
    Path(file_dir).parents[1].absolute(), ".mlx_graphs_data/"
)


class Dataset(ABC):
    """
    Base dataset class. ``download``, ``process``, ``__get_item__``,
    ``__len__`` methods must be implemented by children classes

    Args:
        name: name of the dataset
        base_dir: directory where to store raw and processed data. Default is
            ``~/.mlx_graphs_data/``. If `None`, data are not stored.
    """

    def __init__(
        self,
        name: str,
        base_dir: Optional[str] = DEFAULT_BASE_DIR,
    ):
        self._name = name
        self._base_dir = base_dir

        self._load()

    @property
    def name(self) -> str:
        """
        Name of the dataset
        """
        return self._name

    @property
    def raw_path(self) -> Optional[str]:
        """
        The path where raw files are stored. Defaults at `<base_dir>/<name>/raw`

        """
        if self._base_dir is not None:
            return os.path.expanduser(os.path.join(self._base_dir, self.name, "raw"))
        return None

    @property
    def processed_path(self) -> Optional[str]:
        """
        The path where raw files are stored. Defaults at `<base_dir>/<name>/processed`
        """
        if self._base_dir is not None:
            return os.path.expanduser(
                os.path.join(self._base_dir, self.name, "processed")
            )
        return None

    @abstractmethod
    def download(self):
        """Download the dataset at `self.raw_path`."""
        pass

    @abstractmethod
    def process(self):
        """Process the dataset and possibly save it at `self.processed_path`"""
        pass

    def _download(self):
        if self._base_dir is not None and self.raw_path is not None:
            if os.path.exists(self.raw_path):
                return
            os.makedirs(self._base_dir)
            self.download()

    def _load(self):
        self._download()
        self.process()

    @abstractmethod
    def __getitem__(self, idx) -> GraphData:
        """Returns the `GraphData` at index idx."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Number of examples in the dataset"""
        pass
