import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, Union

DEFAULT_BASE_DIR = os.path.join(os.getcwd(), ".mlx_graphs_data/")


class BaseDataset(ABC):
    """
    Abstract base class for datasets.

    Args:
        name: The name of the dataset.
        base_dir: The base directory where the dataset is stored.
            Default is in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function to apply as a pre-transform to each graph in the
            dataset. Defaults to None.
        transform : A function to apply as a transform to each graph in the dataset.
            Defaults to None.
    """

    def __init__(
        self,
        name: str,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        self._name = name
        self._base_dir = base_dir if base_dir else DEFAULT_BASE_DIR
        self.transform = transform
        self.pre_transform = pre_transform
        self.graphs = []
        self._load()

    @property
    def name(self) -> str:
        """Name of the dataset"""
        return self._name

    @property
    def raw_path(self) -> str:
        """The path where raw files are stored."""
        return os.path.expanduser(os.path.join(self._base_dir, self.name, "raw"))

    @property
    def processed_path(self) -> str:
        """The path where processed files are stored."""
        return os.path.expanduser(os.path.join(self._base_dir, self.name, "processed"))

    @property
    def num_items(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self)

    @abstractmethod
    def download(self):
        """Download the dataset at `self.raw_path`."""
        pass

    @abstractmethod
    def process(self):
        """Process the dataset and store data in `self.data`"""
        pass

    def save(self):
        """Save the processed dataset"""
        with open(os.path.join(self.processed_path, "graphs.pkl"), "wb") as f:
            pickle.dump(self.graphs, f)

    def load(self):
        """Load the processed dataset"""
        with open(os.path.join(self.processed_path, "graphs.pkl"), "rb") as f:
            obj = pickle.load(f)
            self.graphs = obj

    def _download(self):
        if self._base_dir is not None and self.raw_path is not None:
            if os.path.exists(self.raw_path):
                return
            os.makedirs(self.raw_path, exist_ok=True)
            print(f"Downloading {self.name} raw data ...", end=" ")
            self.download()
            print("Done")

    def _process(self):
        self.process()

        if self.pre_transform:
            print(f"Applying pre-transform to {self.name} data ...", end=" ")
            self.graphs = [self.pre_transform(graph) for graph in self.graphs]
            print("Done")

    def _save(self):
        if self._base_dir is not None and self.processed_path is not None:
            if not os.path.exists(self.processed_path):
                os.makedirs(self.processed_path, exist_ok=True)
        print(f"Saving processed {self.name} data ...", end=" ")
        self.save()
        print("Done")

    def _load(self):
        # try to load the already processed dataset, if unavailable download
        # and process the raw data and save the processed one
        try:
            print(f"Loading {self.name} data ...", end=" ")
            self.load()
            print("Done")
        except FileNotFoundError:
            self._download()
            print(f"Processing {self.name} raw data ...", end=" ")
            self._process()
            print("Done")
            self._save()

    def __len__(self):
        return len(self.graphs)

    @abstractmethod
    def __getitem__(self, idx: Union[int, slice, Sequence]) -> Any:
        pass
