import os
import pickle
from typing import Literal, Optional, get_args

from mlx_graphs.datasets import Dataset
from mlx_graphs.datasets.utils import download, extract_archive

SUPERPIXEL_NAMES = Literal["MNIST", "CIFAR10"]
SUPERPIXEL_SPLITS = Literal["train", "test"]
SUPERPIXEL_PKL_FILES = {
    "MNIST": "mnist_75sp",
    "CIFAR10": "cifar10_150sp",
}


class SuperPixelDataset(Dataset):
    _url = "https://www.dropbox.com/s/y2qwa77a0fxem47/superpixels.zip?dl=1"

    def __init__(
        self,
        name: SUPERPIXEL_NAMES,
        split: SUPERPIXEL_SPLITS,
        base_dir: Optional[str] = None,
    ):
        assert name in get_args(SUPERPIXEL_NAMES), "Invalid dataset name"
        assert split in get_args(SUPERPIXEL_SPLITS), "Invalid split specified"
        self.split = split
        super().__init__(name=name, base_dir=base_dir)

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
            print(labels)
