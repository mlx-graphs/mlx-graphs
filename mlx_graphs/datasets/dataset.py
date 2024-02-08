import os
from abc import ABC, abstractmethod
from typing import Optional

DEFAULT_BASE_DIR = "~/.mlx_graphs_data/"


class Dataset(ABC):
    """
    Base dataset class. ``download``, ``process``, ``__get_item__``, ``__len__``
    methods must be implemented by children classes

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
            return os.path.join(self._base_dir, self.name, "raw")
        return None

    @property
    def processed_path(self) -> Optional[str]:
        """
        The path where raw files are stored. Defaults at `<base_dir>/<name>/processed`
        """
        if self._base_dir is not None:
            return os.path.join(self._base_dir, self.name, "processed")
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
        if self._base_dir is not None and self.raw_path:
            if os.path.exists(self.raw_path):
                return
            os.makedirs(self._base_dir)
            self.download()

    def _load(self):
        self._download()
        self.process()

    @abstractmethod
    def __getitem__(self, idx):
        """Returns the `GraphData` at index idx."""
        pass

    @abstractmethod
    def __len__(self):
        """Number of examples in the dataset"""
        pass
