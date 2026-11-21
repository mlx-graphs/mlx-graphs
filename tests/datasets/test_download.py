import io
import os
import tarfile
import zipfile

import pytest

from mlx_graphs.datasets.utils import extract_archive


def _add_file(tar: tarfile.TarFile, name: str, data: bytes = b"x"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _make_tar(path, add):
    with tarfile.open(path, "w") as tar:
        add(tar)
    return str(path)


def test_extract_tar(tmp_path):
    archive = _make_tar(
        tmp_path / "ok.tar", lambda tar: _add_file(tar, "dir/file.txt", b"hello")
    )
    target = tmp_path / "out"
    extract_archive(archive, str(target))
    assert (target / "dir" / "file.txt").read_bytes() == b"hello"


def test_extract_tar_rejects_sibling_with_common_prefix(tmp_path):
    # "../out_evil" shares the string prefix ".../out" with the target, which a
    # `commonprefix` check accepted.
    archive = _make_tar(
        tmp_path / "evil.tar", lambda tar: _add_file(tar, "../out_evil/file.txt")
    )
    with pytest.raises(Exception, match="Path Traversal"):
        extract_archive(archive, str(tmp_path / "out"))
    assert not os.path.exists(tmp_path / "out_evil")


def test_extract_tar_rejects_links_outside(tmp_path):
    def add(tar):
        link = tarfile.TarInfo("escape")
        link.type = tarfile.SYMTYPE
        link.linkname = str(tmp_path)
        tar.addfile(link)
        _add_file(tar, "escape/pwned.txt")

    archive = _make_tar(tmp_path / "evil.tar", add)
    with pytest.raises(Exception, match="Path Traversal"):
        extract_archive(archive, str(tmp_path / "out"))
    assert not os.path.exists(tmp_path / "pwned.txt")


def test_extract_zip_rejects_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../out_evil/file.txt", b"x")
    with pytest.raises(Exception, match="Path Traversal"):
        extract_archive(str(archive), str(tmp_path / "out"))
