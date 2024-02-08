import os
from typing import Optional


class Dataset:
    """
    Base dataset class. ``download``, ``process``, and ``save`` methods must
    be implemented by children classes

    Args:
        name: name of the dataset
        raw_dir: directory where to store raw data
        save_dir: directory where to store processed data
    """

    def __init__(
        self,
        name: str,
        download_url: Optional[str] = None,
        raw_dir: str = ".local_data/raw",
        save_dir: str = ".local_data/processed",
    ):
        self.name = name
        self.dowload_url = download_url
        self.raw_dir = raw_dir
        self.save_dir = save_dir
        self.raw_path = os.path.join(self.raw_dir, self.name)
        self.save_path = os.path.join(self.save_dir, self.name)

        self.load()

    def download(self):
        raise NotImplementedError

    def _download(self):
        if os.path.exists(self.raw_path):
            return
        os.makedirs(self.raw_dir)
        self.download()

    def process(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        self._download()
        self.process()
        self.save()
