"""Checks the large TGB datasets against py-tgb on the first events of their
real raw files, without downloading the archives (1.3 to 1.7 GB each).

    python benchmarks/tgn_comparison/check_tgb_slices.py --out <dir>

For every dataset, the zip archive is read with HTTP range requests: its list
of files is checked against the files ``TGBLinkPropPredDataset`` /
``TGBNodePropPredDataset`` expect, and the first ``--rows`` lines of the edge
list (plus the airport file of ``tgbl-flight`` and the label rows of the
sliced nodes for ``tgbn-*``) are written to ``<out>/<dataset>/raw``. Both
libraries then process the slice and their events, features, splits and label
streams must be identical. The negative samples (pickles of the whole
dataset) are not checked.
"""

import io
import os
import sys
import zipfile
from argparse import ArgumentParser

import numpy as np
import requests

from mlx_graphs.datasets import TGBLinkPropPredDataset, TGBNodePropPredDataset

LINK = ["tgbl-coin", "tgbl-comment", "tgbl-flight"]
NODE = ["tgbn-reddit", "tgbn-token"]


class HTTPFile:
    """A read-only, seekable file over HTTP range requests, fetched in blocks
    of 4 MB."""

    block = 1 << 22

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        self.size = int(requests.head(url, timeout=60).headers["content-length"])
        self.cache_start, self.cache = 0, b""

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _fetch(self, start: int):
        end = min(self.size, start + self.block) - 1
        r = requests.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                         timeout=300)  # fmt: skip
        r.raise_for_status()
        self.cache_start, self.cache = start, r.content

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        out = []
        while self.pos < end:
            offset = self.pos - self.cache_start
            if not 0 <= offset < len(self.cache):
                self._fetch(self.pos)
                offset = 0
            chunk = self.cache[offset : offset + end - self.pos]
            out.append(chunk)
            self.pos += len(chunk)
        return b"".join(out)


def head_lines(archive: zipfile.ZipFile, member: str, num_lines: int) -> list[bytes]:
    lines = []
    with archive.open(member) as f:
        reader = io.BufferedReader(f, buffer_size=1 << 22)
        for line in reader:
            lines.append(line)
            if len(lines) > num_lines:  # header + rows
                break
    return lines


def slice_dataset(cls, name: str, out: str, rows: int):
    files = object.__new__(cls)  # not loaded: only for the file names and URL
    files.tgb_name = name
    url = files.url
    raw = os.path.join(out, name.replace("-", "_"), "raw")
    os.makedirs(raw, exist_ok=True)
    archive = zipfile.ZipFile(HTTPFile(url))
    members = {os.path.basename(m): m for m in archive.namelist()}
    expected = files.raw_file_names
    missing = [f for f in expected if f not in members]
    print(f"{name}: {url}\n  files: {sorted(members)}")
    assert not missing, f"missing files {missing}"

    edges = [f for f in expected if "_edgelist" in f][0]
    lines = head_lines(archive, members[edges], rows)
    with open(os.path.join(raw, edges), "wb") as f:
        f.writelines(lines)
    for small in [f for f in expected if f.startswith("airport_node_feat")]:
        with archive.open(members[small]) as src, open(
            os.path.join(raw, small), "wb"
        ) as dst:
            dst.write(src.read())
    if cls is TGBNodePropPredDataset:
        # label rows whose node and class appear in the sliced events
        rows_ = [line.decode().rstrip("\n").split(",") for line in lines[1:]]
        users, items = {r[1] for r in rows_}, {r[2] for r in rows_}
        labels = [f for f in expected if f.endswith("_node_labels.csv")][0]
        kept, last_t = [], int(rows_[-1][0])
        with archive.open(members[labels]) as f:
            reader = io.BufferedReader(f, buffer_size=1 << 22)
            header = next(reader)
            for line in reader:
                row = line.decode().split(",")
                if int(row[0]) > last_t + 7 * 86400:
                    break
                if row[1] in users and row[2] in items:
                    kept.append(line)
        with open(os.path.join(raw, labels), "wb") as f:
            f.write(header)
            f.writelines(kept)
        print(f"  {len(kept)} label rows")
    # Empty placeholders for the negative samples (not checked), so that the
    # dataset does not download the archive.
    for f in expected:
        if f.endswith(".pkl"):
            open(os.path.join(raw, f), "wb").close()
    return raw


def py_tgb_root(out: str, name: str, raw: str) -> str:
    from tgb.utils.info import PROJ_DIR

    target = os.path.join(out, "py_tgb", name.replace("-", "_"))
    os.makedirs(target, exist_ok=True)
    for f in os.listdir(raw):
        if not os.path.exists(os.path.join(target, f)):
            os.symlink(os.path.join(raw, f), os.path.join(target, f))
    return os.path.relpath(os.path.join(out, "py_tgb"), PROJ_DIR)


def same(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b))


def check_link(name, out, raw):
    from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

    dataset = TGBLinkPropPredDataset(name, base_dir=out)
    ref = PyGLinkPropPredDataset(name=name, root=py_tgb_root(out, name, raw))
    data, data_ref = dataset[0], ref.get_TemporalData()
    checks = {key: same(data[key], data_ref[key].numpy())
              for key in ("src", "dst", "t")}  # fmt: skip
    checks.update({key: same(data[key], data_ref[key].numpy().astype(np.float32))
                   for key in ("msg", "w")})  # fmt: skip
    for split in ("train_mask", "val_mask", "test_mask"):
        checks[split] = same(getattr(dataset, split), getattr(ref, split).numpy())
    if ref.node_feat is not None:
        checks["node_feat"] = same(dataset.node_feat, ref.node_feat.numpy())
    return data.num_events, checks


def check_node(name, out, raw):
    from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset

    dataset = TGBNodePropPredDataset(name, base_dir=out)
    root = py_tgb_root(out, name, raw)
    ref = PyGNodePropPredDataset(name=name, root=root)
    if name == "tgbn-token":  # py-tgb takes the log of the weights once cached
        ref = PyGNodePropPredDataset(name=name, root=root)
    data, data_ref = dataset[0], ref.get_TemporalData()
    checks = {key: same(data[key], data_ref[key].numpy())
              for key in ("src", "dst", "t", "msg")}  # fmt: skip
    for split in ("train_mask", "val_mask", "test_mask"):
        checks[split] = same(getattr(dataset, split), getattr(ref, split).numpy())
    label_ts = ref.dataset.label_ts
    checks["label_ts"] = same(dataset.label_timestamps, label_ts)
    ok = True
    for ts in label_ts:
        mine, theirs = dataset.get_node_label(int(ts)), ref.get_node_label(int(ts))
        ok &= all(same(a, b.numpy()) for a, b in zip(mine, theirs))
    checks["labels"] = ok
    return data.num_events, checks


def main(args):
    failed = False
    for name in args.datasets:
        if name in LINK:
            raw = slice_dataset(TGBLinkPropPredDataset, name, args.out, args.rows)
            num, checks = check_link(name, args.out, raw)
        else:
            raw = slice_dataset(TGBNodePropPredDataset, name, args.out, args.rows)
            num, checks = check_node(name, args.out, raw)
        failed |= not all(checks.values())
        print(f"  {num} events, identical to py-tgb: {checks}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--datasets", nargs="+", default=LINK + NODE)
    main(parser.parse_args())
