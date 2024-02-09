import os

from dataset import DEFAULT_BASE_DIR, Dataset
from utils import check_sha1, download


class QM7bDataset(Dataset):
    _url = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7b.mat"
    _sha1_str = "4102c744bb9d6fd7b40ac67a300e49cd87e28392"

    def __init__(self, base_dir: str = DEFAULT_BASE_DIR):
        super().__init__(name="qm7b", base_dir=base_dir)

    def download(self):
        assert self.raw_path is not None, "Unable to access/create the self.raw_path"
        file_path = os.path.join(self.raw_path, self.name + ".mat")
        download(self._url, path=file_path)
        if not check_sha1(file_path, self._sha1_str):
            raise UserWarning(
                "File {} is downloaded but the content hash does not match."
                "The repo may be outdated or download may be incomplete. "
                "Otherwise you can create an issue for it.".format(self.name)
            )

    def process(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


a = QM7bDataset()
