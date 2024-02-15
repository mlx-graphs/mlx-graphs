import hashlib
import os
import pickle
import warnings
import zipfile
from typing import Optional

import requests

from mlx_graphs.data.data import GraphData


def save_graphs(path: str, data: list[GraphData], file_name: Optional[str] = None):
    if file_name is None:
        file_name = "data.pkl"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, file_name), "wb") as f:
        pickle.dump(data, f)


def download(
    url: str,
    path: Optional[str] = None,
    overwrite: bool = True,
    sha1_hash: Optional[str] = None,
    retries: int = 5,
    verify_ssl: bool = True,
    log: bool = True,
) -> str:
    """Download a given URL.

    Code borrowed from dgl

    Args:
        url: URL to download.
        path: Destination path to store downloaded file. By default stores to the
            current directory with the same name as in url.
        overwrite: Whether to overwrite the destination file if it already exists.
            By default always overwrites the downloaded file.
        sha1_hash: Expected sha1 hash in hexadecimal digits. Will ignore existing file
            when hash is specified but doesn't match.
        retries: The number of times to attempt downloading in case of failure or non
            200 return codes.
        verify_ssl: Verify SSL certificates.
        log: Whether to print the progress for download

    Returns:
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, (
            "Can't construct file-name from this URL. "
            "Please set the `path` option manually."
        )
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split("/")[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised."
        )

    if (
        overwrite
        or not os.path.exists(fname)
        or (sha1_hash and not check_sha1(fname, sha1_hash))
    ):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                if log:
                    print("Downloading %s from %s..." % (fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s" % url)
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not check_sha1(fname, sha1_hash):
                    raise UserWarning(
                        "File {} is downloaded but the content hash does not match."
                        " The repo may be outdated or download may be incomplete. "
                        'If the "repo_url" is overridden, consider switching to '
                        "the default repo.".format(fname)
                    )
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    if log:
                        print(
                            "download failed, retrying, {} attempt{} left".format(
                                retries, "s" if retries > 1 else ""
                            )
                        )

    return fname


def check_sha1(filename: str, sha1_hash: str) -> bool:
    """Check whether the sha1 hash of the file content matches the expected hash.

    Code borrowed from dgl

    Args:
        filename: Path to the file.
        sha1_hash: Expected sha1 hash in hexadecimal digits.

    Returns:
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def extract_zip(path: str, folder: str, log: bool = True):
    r"""Extracts a zip archive to a specific folder.

    Code borrowed from PyG

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
    """
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)
