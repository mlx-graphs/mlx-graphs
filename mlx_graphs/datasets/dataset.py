import os
from abc import ABC, abstractmethod
from typing import Optional

DEFAULT_BASE_DIR = "~/.mlx_graphs_data/"


class Dataset(ABC):
    """
    Base dataset class. ``download`` and ``process`` methods must
    be implemented by children classes

    Args:
        name: name of the dataset
        base_dir: directory where to store raw and processed data. Default is
            ``~/.mlx_graphs_data/``
    """

    def __init__(
        self,
        name: str,
        base_dir: Optional[str] = None,
    ):
        self._name = name
        self._base_dir = base_dir if base_dir is not None else DEFAULT_BASE_DIR

        self._load()

    @property
    def name(self):
        """
        Name of the dataset
        """
        return self._name

    @property
    def raw_path(self):
        """
        The path where raw files are stored. Defaults at `<base_dir>/<name>/raw`

        """
        return os.path.join(self._base_dir, self.name, "raw")

    @property
    def processed_path(self):
        """
        The path where raw files are stored. Defaults at `<base_dir>/<name>/processed`
        """
        return os.path.join(self._base_dir, self.name, "processed")

    @abstractmethod
    def download(self):
        """Download the dataset at `self.raw_path`."""
        pass

    @abstractmethod
    def process(self):
        """Process the dataset and possibly save it at `self.processed_path`"""
        pass

    def _download(self):
        if os.path.exists(self.raw_path):
            return
        os.makedirs(self._base_dir)
        self.download()

    def _load(self):
        self._download()
        self.process()
