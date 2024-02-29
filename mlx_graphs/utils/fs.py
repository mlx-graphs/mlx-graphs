"""Borrowed from PyG (https://github.com/pyg-team/pytorch_geometric/blob/b812cffc9dc4cd2b901377305f5cb716fa31e5fd/torch_geometric/io/fs.py)"""


import os.path as osp
import sys
from typing import Any, Dict

import fsspec


def get_fs(path: str) -> fsspec.AbstractFileSystem:
    r"""Get filesystem backend given a path URI to the resource.

    Here are some common example paths and dispatch result:

    * :obj:`"/home/file"` ->
      :class:`fsspec.implementations.local.LocalFileSystem`
    * :obj:`"memory://home/file"` ->
      :class:`fsspec.implementations.memory.MemoryFileSystem`
    * :obj:`"https://home/file"` ->
      :class:`fsspec.implementations.http.HTTPFileSystem`
    * :obj:`"gs://home/file"` -> :class:`gcsfs.GCSFileSystem`
    * :obj:`"s3://home/file"` -> :class:`s3fs.S3FileSystem`

    A full list of supported backend implementations of :class:`fsspec` can be
    found `here <https://github.com/fsspec/filesystem_spec/blob/master/fsspec/
    registry.py#L62>`_.

    The backend dispatch logic can be updated with custom backends following
    `this tutorial <https://filesystem-spec.readthedocs.io/en/latest/
    developer.html#implementing-a-backend>`_.

    Args:
        path (str): The URI to the filesystem location, *e.g.*,
            :obj:`"gs://home/me/file"`, :obj:`"s3://..."`.
    """
    return fsspec.core.url_to_fs(path)[0]


def exists(path: str) -> bool:
    return get_fs(path).exists(path)


def isdir(path: str) -> bool:
    return get_fs(path).isdir(path)


def isdisk(path: str) -> bool:
    return "file" in get_fs(path).protocol


def islocal(path: str) -> bool:
    return isdisk(path) or "memory" in get_fs(path).protocol


def cp(
    path1: str,
    path2: str,
    extract: bool = False,
    log: bool = True,
) -> None:
    kwargs: Dict[str, Any] = {}

    is_path1_dir = isdir(path1)
    is_path2_dir = isdir(path2)

    if not islocal(path1):
        if log and "pytest" not in sys.modules:
            print(f"Downloading {path1}", file=sys.stderr)

    # Handle automatic extraction:
    multiple_files = False
    if extract and path1.endswith(".tar.gz"):
        kwargs.setdefault("tar", dict(compression="gzip"))
        path1 = f"tar://**::{path1}"
        multiple_files = True
    elif extract and path1.endswith(".zip"):
        path1 = f"zip://**::{path1}"
        multiple_files = True
    elif extract and path1.endswith(".gz"):
        kwargs.setdefault("compression", "infer")
    elif extract:
        raise NotImplementedError(
            f"Automatic extraction of '{path1}' not yet supported"
        )

    # If the source path points to a directory, we need to make sure to
    # recursively copy all files within this directory. Additionally, if the
    # destination folder does not yet exist, we inherit the basename from the
    # source folder.
    if is_path1_dir:
        if exists(path2):
            path2 = osp.join(path2, osp.basename(path1))
        path1 = osp.join(path1, "**")
        multiple_files = True

    # Perform the copy:
    for open_file in fsspec.open_files(path1, **kwargs):
        with open_file as f_from:
            if not multiple_files:
                if is_path2_dir:
                    basename = osp.basename(path1)
                    if extract and path1.endswith(".gz"):
                        basename = ".".join(basename.split(".")[:-1])
                    to_path = osp.join(path2, basename)
                else:
                    to_path = path2
            else:
                # Open file has protocol stripped.
                common_path = osp.commonprefix(
                    [fsspec.core.strip_protocol(path1), open_file.path]
                )
                to_path = osp.join(path2, open_file.path[len(common_path) :])
            with fsspec.open(to_path, "wb") as f_to:
                while True:
                    chunk = f_from.read(10 * 1024 * 1024)
                    if not chunk:
                        break
                    f_to.write(chunk)
