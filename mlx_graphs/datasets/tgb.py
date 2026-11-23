import csv
import datetime
import os
import pickle
import zipfile
from typing import Any, Callable, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data.temporal import TemporalData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download
from mlx_graphs.metrics.tgb import TGB_EVAL_METRICS, TGBEvaluator

TGB_URL = (
    "https://object-arbutus.alliancecan.ca/swift/v1/"
    "14c95234f6cd4a21a47deafe20cce2a7/tgb/{}.zip"
)

# Version of the files of every dataset (`tgb.utils.info.DATA_VERSION_DICT`):
# version ``v > 1`` is downloaded from ``<name>-v<v>.zip`` and its files are
# suffixed with ``_v<v>``.
_LINK_VERSIONS = {
    "tgbl-wiki": 2,
    "tgbl-review": 2,
    "tgbl-coin": 2,
    "tgbl-comment": 1,
    "tgbl-flight": 2,
}
# Number of classes of the node property prediction datasets
# (`tgb.utils.info.DATA_NUM_CLASSES`). The item nodes (genres, subreddits,
# tokens) are numbered from 0 and the user nodes from this number.
_NODE_NUM_CLASSES = {
    "tgbn-trade": 255,
    "tgbn-genre": 513,
    "tgbn-reddit": 698,
    "tgbn-token": 1001,
}
_SPLITS = ("val", "test")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, mx.array):
        return np.array(value)
    if hasattr(value, "detach"):  # torch tensors
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _suffix(version: int) -> str:
    return f"_v{version}" if version > 1 else ""


def _read_csv(path: str) -> list[list[str]]:
    """Rows of a (small) CSV file without its header."""
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader, None)
        return list(reader)


def _num_rows(path: str) -> int:
    """Number of lines of a file minus its header, as TGB counts the rows."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def _parse_csv(path: str, columns: list) -> list[np.ndarray]:
    """Parses a CSV file (without its header) row by row into preallocated
    arrays, as TGB does, without holding the rows in memory.

    Args:
        path: The CSV file
        columns: ``(column, convert, dtype)`` triples: the array of every
            triple gets ``convert(row[column])`` for every row (``column`` can
            be a tuple of columns, passed as several arguments, and ``dtype``
            a ``(dtype, width)`` pair for a vector per row). The converters
            are called in this order for every row.

    Returns:
        The arrays.
    """
    num_rows = max(_num_rows(path), 0)
    outs = []
    for _, _, dtype in columns:
        if isinstance(dtype, tuple):
            outs.append(np.empty((num_rows, dtype[1]), dtype=dtype[0]))
        else:
            outs.append(np.empty(num_rows, dtype=dtype))
    getters = [
        (out, convert, col if isinstance(col, tuple) else None, col)
        for out, (col, convert, _) in zip(outs, columns)
    ]
    count = 0
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader, None)
        for i, row in enumerate(reader):
            for out, convert, cols, col in getters:
                if cols is None:
                    out[i] = convert(row[col])
                else:
                    out[i] = convert(*(row[c] for c in cols))
            count = i + 1
    return [out[:count] for out in outs]


class _IdMap(dict):
    """Numbers keys in order of first appearance from ``start``."""

    def __init__(self, start: int = 0):
        super().__init__()
        self.start = start

    def __call__(self, key: str) -> int:
        idx = self.get(key)
        if idx is None:
            idx = self[key] = self.start + len(self)
        return idx


def _str_to_codes(string: str) -> np.ndarray:
    """TGB's ``convert_str2int``: digits are kept, ``"!"`` (padding) is -1 and
    letters are ``ord(upper) - 35``."""
    out = []
    for c in string:
        if c.isnumeric():
            out.append(float(c))
        elif c == "!":
            out.append(-1.0)
        else:
            out.append(float(ord(c.upper()) - 44 + 9))
    return np.array(out, dtype=np.float32)


def _split_masks(t: np.ndarray, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """TGB's chronological split: events up to the ``1 - val - test`` quantile
    of the timestamps are for training, then up to the ``1 - test`` quantile for
    validation, and the rest for test."""
    val_time, test_time = np.quantile(t, [1 - val_ratio - test_ratio, 1 - test_ratio])
    train_mask = t <= val_time
    val_mask = np.logical_and(t <= test_time, t > val_time)
    test_mask = t > test_time
    return train_mask, val_mask, test_mask


# ----------------------------------------------------------------------
# Negative samples
# ----------------------------------------------------------------------

_NS_ALLOWED_GLOBALS = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
}


class _NumpyUnpickler(pickle.Unpickler):
    """Unpickler restricted to plain containers and NumPy arrays / scalars:
    the negative samples of TGB are pickled dictionaries, and this refuses
    any other object a tampered file could contain."""

    def find_class(self, module: str, name: str):
        if (module, name) in _NS_ALLOWED_GLOBALS:
            import warnings

            with warnings.catch_warnings():
                # `numpy.core` is a deprecated alias of `numpy._core`
                warnings.simplefilter("ignore", DeprecationWarning)
                return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to load '{module}.{name}' from a TGB negative sample file"
        )


def _key_dtype(num_fields: int) -> np.dtype:
    return np.dtype([(f"f{i}", "<i8") for i in range(num_fields)])


def _make_keys(columns: list[np.ndarray]) -> np.ndarray:
    keys = np.empty(len(columns[0]), dtype=_key_dtype(len(columns)))
    for i, col in enumerate(columns):
        keys[f"f{i}"] = col
    return keys


def convert_tgb_negatives(pkl_path: str, npz_path: str):
    """Converts a TGB negative sample file (a pickled dictionary mapping every
    positive edge ``(src, dst, t)`` to its array of negative destinations) to
    a ``.npz`` file of arrays: the sorted positive edges ``keys`` and the
    negative destinations ``neg`` (padded with -1) and their number
    ``num_neg``.

    The pickle is read with an unpickler that only accepts NumPy arrays and
    scalars.

    Args:
        pkl_path: The TGB ``*_ns*.pkl`` file
        npz_path: The ``.npz`` file to write
    """
    with open(pkl_path, "rb") as f:
        eval_set = _NumpyUnpickler(f).load()  # nosec B301 - restricted unpickler
    if not isinstance(eval_set, dict):
        raise ValueError(f"{pkl_path} does not contain a dictionary")
    num_keys = len(eval_set)
    key_list = list(eval_set.keys())
    num_fields = len(key_list[0]) if num_keys else 3
    keys = np.array(key_list, dtype=np.int64).reshape(num_keys, num_fields)
    num_neg = np.fromiter((len(v) for v in eval_set.values()), np.int64, num_keys)
    width = int(num_neg.max()) if num_keys else 0
    neg = np.full((num_keys, width), -1, dtype=np.int64)
    for i, value in enumerate(eval_set.values()):
        neg[i, : num_neg[i]] = np.asarray(value, dtype=np.int64)
    del eval_set
    dtype = np.int32 if neg.size == 0 or neg.max() < 2**31 else np.int64
    np.savez(npz_path, keys=keys, neg=neg.astype(dtype), num_neg=num_neg)


class TGBNegativeEdgeSampler:
    """The precomputed negative edges of the validation and test events of a
    TGB link property prediction dataset, as ``py-tgb``'s
    ``tgb.linkproppred.negative_sampler.NegativeEdgeSampler``.

    Each positive event ``(src, dst, t)`` of the validation and test sets is
    evaluated against a fixed list of negative destinations (one-vs-many),
    which mixes historical and random destinations (TGB's ``"hist_rnd"``
    strategy). The sets are loaded with :meth:`load_eval_set` (or the
    ``load_val_ns`` / ``load_test_ns`` methods of the dataset) and queried
    with :meth:`query_batch`, or with :meth:`query_batch_array` to get a
    padded array.

    Args:
        dataset_name: Name of the TGB dataset
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.eval_set: dict[str, Optional[dict]] = {}

    def load_eval_set(self, fname: str, split_mode: str = "val"):
        """Loads the negative samples of a split.

        Args:
            fname: A ``.npz`` file written by :func:`convert_tgb_negatives`
                (or a TGB ``.pkl`` file, which is then converted next to it)
            split_mode: ``"val"`` or ``"test"``
        """
        if split_mode not in _SPLITS:
            raise ValueError("Invalid split-mode! It should be `val` or `test`")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File not found at {fname}")
        if fname.endswith(".pkl"):
            npz = fname[: -len(".pkl")] + ".npz"
            if not os.path.exists(npz):
                convert_tgb_negatives(fname, npz)
            fname = npz
        with np.load(fname, allow_pickle=False) as f:
            keys, neg, num_neg = f["keys"], f["neg"], f["num_neg"]
        keys = _make_keys([keys[:, i] for i in range(keys.shape[1])])
        order = np.argsort(keys, kind="stable")
        self.eval_set[split_mode] = {
            "keys": keys[order],
            "neg": neg[order],
            "num_neg": num_neg[order],
        }

    def reset_eval_set(self, split_mode: str = "test"):
        """Unloads the negative samples of a split."""
        if split_mode not in _SPLITS:
            raise ValueError("Invalid split-mode! It should be `val` or `test`")
        self.eval_set[split_mode] = None

    def _rows(self, pos_src, pos_dst, pos_timestamp, edge_type, split_mode):
        if split_mode not in _SPLITS:
            raise ValueError("Invalid split-mode! It should be `val` or `test`")
        eval_set = self.eval_set.get(split_mode)
        if eval_set is None:
            raise ValueError(
                f"Evaluation set is None! You should load the {split_mode} "
                "evaluation set first!"
            )
        columns = [_to_numpy(pos_src), _to_numpy(pos_dst), _to_numpy(pos_timestamp)]
        if edge_type is not None:
            columns.append(_to_numpy(edge_type))
        columns = [c.reshape(-1).astype(np.int64) for c in columns]
        keys = eval_set["keys"]
        if len(columns) != len(keys.dtype.names):
            raise ValueError(
                f"The {split_mode} negative samples are indexed by "
                f"{len(keys.dtype.names)} fields (got {len(columns)})"
            )
        query = _make_keys(columns)
        rows = np.searchsorted(keys, query)
        found = rows < len(keys)
        found[found] = keys[rows[found]] == query[found]
        if not found.all():
            missing = tuple(int(c[~found][0]) for c in columns)
            raise ValueError(
                f"The edge {missing} is not in the '{split_mode}' evaluation set! "
                "Please check the implementation."
            )
        return eval_set, rows

    def query_batch(
        self,
        pos_src: Any,
        pos_dst: Any,
        pos_timestamp: Any,
        edge_type: Any = None,
        split_mode: str = "test",
    ) -> list[list[int]]:
        """Negative destinations of each positive edge, as ``py-tgb``.

        Args:
            pos_src: Sources of the positive edges, shape `[batch_size]`
            pos_dst: Destinations of the positive edges
            pos_timestamp: Timestamps of the positive edges
            edge_type: Edge types (only for datasets with typed edges)
            split_mode: ``"val"`` or ``"test"``

        Returns:
            A list with the list of negative destinations of every positive
            edge.
        """
        eval_set, rows = self._rows(
            pos_src, pos_dst, pos_timestamp, edge_type, split_mode
        )
        neg, num_neg = eval_set["neg"][rows], eval_set["num_neg"][rows]
        return [row[:n].tolist() for row, n in zip(neg, num_neg)]

    def query_batch_array(
        self,
        pos_src: Any,
        pos_dst: Any,
        pos_timestamp: Any,
        edge_type: Any = None,
        split_mode: str = "test",
    ) -> tuple[mx.array, mx.array]:
        """Negative destinations of each positive edge as a padded array.

        Most positive edges of a split have the same number of negatives, a
        few have less: their rows are padded with ``-1``. Give the scores of
        the padding entries the value ``-inf`` before passing them to the
        :class:`~mlx_graphs.metrics.TGBEvaluator`, so that they never rank
        above the positive.

        Args:
            pos_src: Sources of the positive edges, shape `[batch_size]`
            pos_dst: Destinations of the positive edges
            pos_timestamp: Timestamps of the positive edges
            edge_type: Edge types (only for datasets with typed edges)
            split_mode: ``"val"`` or ``"test"``

        Returns:
            The negative destinations, shape `[batch_size, max_num_neg]`, and
            the boolean mask of the valid (non-padding) entries.
        """
        eval_set, rows = self._rows(
            pos_src, pos_dst, pos_timestamp, edge_type, split_mode
        )
        num_neg = eval_set["num_neg"][rows]
        width = int(num_neg.max()) if num_neg.size else 0
        neg = eval_set["neg"][rows, :width]
        mask = np.arange(width)[None, :] < num_neg[:, None]
        return mx.array(neg), mx.array(mask)

    def __repr__(self) -> str:
        loaded = [s for s in _SPLITS if self.eval_set.get(s) is not None]
        return f"TGBNegativeEdgeSampler({self.dataset_name}, loaded={loaded})"


class _FlightTime:
    """Timestamps of ``"tgbl-flight"``: unix times, or dates converted with
    the local time zone as TGB does (the format of the first row decides)."""

    def __init__(self):
        self.is_unix = None
        self.cache = {}

    def __call__(self, day: str) -> int:
        if self.is_unix is None:
            self.is_unix = day.isdigit()
        if self.is_unix:
            return int(float(int(day)))
        ts = self.cache.get(day)
        if ts is None:
            date = datetime.datetime.strptime(day, "%Y-%m-%d")
            ts = self.cache[day] = int(float(date.timestamp()))
        return ts


class _FlightFeatures:
    """Edge features of ``"tgbl-flight"`` (``csv_to_pd_data`` of TGB): the
    character codes of the call sign and aircraft type code, padded with
    ``"!"`` to 8 characters each."""

    def __init__(self):
        self.cache = {}

    def __call__(self, callsign: str, typecode: str) -> np.ndarray:
        key = (callsign, typecode)
        feat = self.cache.get(key)
        if feat is None:
            if len(callsign) == 0:
                callsign = "!!!!!!!!"
            callsign = callsign + "!" * max(0, 8 - len(callsign))
            if len(typecode) == 0:
                typecode = "!!!!!!!!"
            typecode = typecode + "!" * max(0, 8 - len(typecode))
            if len(typecode) > 8:
                typecode = "!!!!!!!!"
            feat = self.cache[key] = _str_to_codes(callsign + typecode)
        return feat


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------


class _TGBDataset(Dataset):
    names: dict

    def __init__(
        self,
        name: str,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        name = name.lower()
        if name not in self.names:
            raise ValueError(
                f"Unknown {type(self).__name__} dataset '{name}', available: "
                f"{list(self.names)}"
            )
        self.tgb_name = name
        self.arrays: dict[str, np.ndarray] = {}
        super().__init__(
            name=name.replace("-", "_"),
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def version(self) -> int:
        return 1

    @property
    def url(self) -> str:
        """URL of the zip archive of the dataset."""
        archive = self.tgb_name
        if self.version > 1:
            archive += f"-v{self.version}"
        return TGB_URL.format(archive)

    def raw_file(self, kind: str) -> str:
        return os.path.join(
            self.raw_path, f"{self.tgb_name}_{kind}{_suffix(self.version)}"
        )

    @property
    def raw_file_names(self) -> list[str]:
        raise NotImplementedError

    def _download(self):
        # Download again if a raw file is missing (e.g. interrupted download).
        if all(
            os.path.exists(os.path.join(self.raw_path, f)) for f in self.raw_file_names
        ):
            return
        os.makedirs(self.raw_path, exist_ok=True)
        print(f"Downloading {self.name} raw data ...", end=" ")
        self.download()
        print("Done")

    def download(self):
        zip_path = os.path.join(self.raw_path, os.path.basename(self.url))
        download(self.url, path=zip_path)
        wanted = set(self.raw_file_names)
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.namelist():
                # Only extract the expected files, by base name.
                if os.path.basename(member) in wanted and not member.endswith("/"):
                    target = os.path.join(self.raw_path, os.path.basename(member))
                    with archive.open(member) as src, open(target, "wb") as dst:
                        while chunk := src.read(1 << 24):
                            dst.write(chunk)
        os.remove(zip_path)
        missing = [
            f
            for f in self.raw_file_names
            if not os.path.exists(os.path.join(self.raw_path, f))
        ]
        if missing:
            raise FileNotFoundError(f"Missing files in {self.url}: {missing}")

    @property
    def eval_metric(self) -> str:
        """The official evaluation metric of the dataset (``"mrr"`` or
        ``"ndcg"``)."""
        return TGB_EVAL_METRICS[self.tgb_name]

    @property
    def evaluator(self) -> TGBEvaluator:
        """A :class:`~mlx_graphs.metrics.TGBEvaluator` for this dataset."""
        return TGBEvaluator(self.tgb_name)

    @property
    def train_mask(self) -> mx.array:
        """Boolean mask of the training events, shape `[num_events]`."""
        return mx.array(self.arrays["train_mask"])

    @property
    def val_mask(self) -> mx.array:
        """Boolean mask of the validation events, shape `[num_events]`."""
        return mx.array(self.arrays["val_mask"])

    @property
    def test_mask(self) -> mx.array:
        """Boolean mask of the test events, shape `[num_events]`."""
        return mx.array(self.arrays["test_mask"])

    @property
    def num_events(self) -> int:
        """The number of events (temporal edges)."""
        return int(self.arrays["src"].shape[0])

    def get_TemporalData(self) -> TemporalData:
        """The whole event stream (``dataset[0]`` without the transform), as
        ``py-tgb``'s ``get_TemporalData``."""
        return self.graphs[0]

    def _finalize(self, arrays: dict[str, np.ndarray]):
        train, val, test = _split_masks(arrays["t"])
        arrays.update(train_mask=train, val_mask=val, test_mask=test)
        self.arrays = arrays
        self._build_graph()

    def _temporal_data(self) -> TemporalData:
        a = self.arrays
        num_nodes = max(int(a["src"].max()), int(a["dst"].max())) + 1
        int_type = np.int32 if num_nodes <= 2**31 else np.int64
        return TemporalData(
            src=mx.array(a["src"].astype(int_type)),
            dst=mx.array(a["dst"].astype(int_type)),
            t=mx.array(a["t"]),
            msg=mx.array(a["msg"]),
            y=mx.ones(a["src"].shape[0], dtype=mx.float32),
        )

    def _build_graph(self):
        self.graphs = [self._temporal_data()]

    def save(self):
        np.savez(os.path.join(self.processed_path, "data.npz"), **self.arrays)

    def load(self):
        path = os.path.join(self.processed_path, "data.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as f:
            self.arrays = {key: f[key] for key in f.files}
        self._build_graph()
        if self.pre_transform is not None:
            self.graphs = [self.pre_transform(g) for g in self.graphs]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.tgb_name})"


class TGBLinkPropPredDataset(_TGBDataset):
    """The dynamic link property prediction datasets of the `Temporal Graph
    Benchmark (TGB) <https://tgb.complexdatalab.com/>`_, as ``py-tgb``'s
    ``tgb.linkproppred.dataset_pyg.PyGLinkPropPredDataset``.

    The dataset is a single :class:`~mlx_graphs.data.TemporalData` stream of
    events sorted by time, with ``src`` and ``dst`` the nodes, ``t`` the
    timestamps (int64), ``msg`` the edge features, ``w`` the edge weights and
    ``y`` all ones (the label of a true edge). The events are split
    chronologically at the 70% and 85% quantiles of the timestamps
    (``train_mask``, ``val_mask``, ``test_mask``) and every validation and
    test event comes with a
    fixed set of negative destinations (see
    :class:`TGBNegativeEdgeSampler`; 999 per event for ``"tgbl-wiki"``, 100
    for ``"tgbl-review"``), against which it is ranked to compute the mean
    reciprocal rank (MRR) with the :class:`~mlx_graphs.metrics.TGBEvaluator`.

    =================  ==========  ===========  ==========
    Name               #nodes      #events      #features
    =================  ==========  ===========  ==========
    ``"tgbl-wiki"``    9,227       157,474      172
    ``"tgbl-review"``  352,637     4,873,540    1
    ``"tgbl-coin"``    638,486     22,809,486   1
    ``"tgbl-comment"`` 994,790     44,314,507   3
    ``"tgbl-flight"``  18,143      67,169,570   16
    =================  ==========  ===========  ==========

    The node ids and features are those of ``py-tgb`` (the processing of each
    raw file is the same): ``"tgbl-wiki"`` numbers the items after the users
    and its features are the JODIE Wikipedia features; ``"tgbl-review"`` and
    ``"tgbl-coin"`` number the nodes in order of first appearance and their
    feature is ``log2`` of the edge weight (``w``, zero weights count as 1);
    ``"tgbl-comment"`` has the number of words (divided by 5,000, twice) and
    the score of the comments; ``"tgbl-flight"`` encodes the call sign and
    aircraft type code as characters, and has node features
    (:attr:`node_feat`, the type, continent, region and coordinates of the
    airports).

    Args:
        name: Name of the dataset, e.g. ``"tgbl-wiki"``
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the ``TemporalData``
            after processing (it is applied every time the dataset is
            loaded, as the processed arrays are stored without it)
        transform: A function/transform applied to the ``TemporalData`` at
            every access

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import TGBLinkPropPredDataset
        from mlx_graphs.loaders import TemporalDataLoader

        dataset = TGBLinkPropPredDataset("tgbl-wiki")
        data = dataset[0]
        >>> TemporalData(src=[157474], dst=[157474], t=[157474],
                         msg=[157474, 172], y=[157474], w=[157474])
        train_data = data[dataset.train_mask]
        val_data = data[dataset.val_mask]

        dataset.load_val_ns()
        sampler, evaluator = dataset.negative_sampler, dataset.evaluator
        for batch in TemporalDataLoader(val_data, batch_size=200):
            neg_dst, mask = sampler.query_batch_array(
                batch.src, batch.dst, batch.t, split_mode="val"
            )
            # pos_scores [B], neg_scores [B, num_neg] with -inf where ~mask
            ...
        evaluator.eval({"y_pred_pos": pos_scores, "y_pred_neg": neg_scores,
                        "eval_metric": ["mrr"]})

    References:
        - Huang et al., `Temporal Graph Benchmark for Machine Learning on
          Temporal Graphs <https://arxiv.org/abs/2307.01026>`_
        - `py-tgb <https://github.com/shenyangHuang/TGB>`_
    """

    names = _LINK_VERSIONS

    def __init__(
        self,
        name: str,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(name, base_dir, pre_transform, transform)
        self._ns_sampler = TGBNegativeEdgeSampler(self.tgb_name)

    @property
    def version(self) -> int:
        return _LINK_VERSIONS[self.tgb_name]

    @property
    def raw_file_names(self) -> list[str]:
        sfx = _suffix(self.version)
        files = [
            f"{self.tgb_name}_edgelist{sfx}.csv",
            f"{self.tgb_name}_val_ns{sfx}.pkl",
            f"{self.tgb_name}_test_ns{sfx}.pkl",
        ]
        if self.tgb_name == "tgbl-flight":
            files.append(f"airport_node_feat{sfx}.csv")
        return files

    @property
    def negative_sampler(self) -> TGBNegativeEdgeSampler:
        """The :class:`TGBNegativeEdgeSampler` of the dataset (load its
        negative samples with :meth:`load_val_ns` and :meth:`load_test_ns`)."""
        return self._ns_sampler

    def get_negative_sampler(self) -> TGBNegativeEdgeSampler:
        """Alias of :attr:`negative_sampler`."""
        return self._ns_sampler

    def _ns_file(self, split_mode: str) -> str:
        """The negative samples of a split, converted to ``.npz`` in the
        processed directory the first time."""
        npz = os.path.join(self.processed_path, f"{split_mode}_ns.npz")
        if not os.path.exists(npz):
            pkl = self.raw_file(f"{split_mode}_ns") + ".pkl"
            if not os.path.exists(pkl):
                raise FileNotFoundError(f"File not found at {pkl}")
            os.makedirs(self.processed_path, exist_ok=True)
            convert_tgb_negatives(pkl, npz)
        return npz

    def load_val_ns(self):
        """Loads the negative samples of the validation events."""
        self._ns_sampler.load_eval_set(self._ns_file("val"), split_mode="val")

    def load_test_ns(self):
        """Loads the negative samples of the test events."""
        self._ns_sampler.load_eval_set(self._ns_file("test"), split_mode="test")

    @property
    def node_feat(self) -> Optional[mx.array]:
        """Node features (``"tgbl-flight"`` only), shape `[num_nodes, 20]`."""
        feat = self.arrays.get("node_feat")
        return None if feat is None else mx.array(feat)

    @property
    def min_dst_idx(self) -> int:
        """Smallest destination node id (to sample negative destinations)."""
        return int(self.arrays["dst"].min())

    @property
    def max_dst_idx(self) -> int:
        """Largest destination node id (to sample negative destinations)."""
        return int(self.arrays["dst"].max())

    def _temporal_data(self) -> TemporalData:
        data = super()._temporal_data()
        data.w = mx.array(self.arrays["w"])
        return data

    def process(self):
        path = self.raw_file("edgelist") + ".csv"
        name = self.tgb_name
        node_feat = None
        if name == "tgbl-wiki":
            # Columns: user, item, timestamp, state label, features...
            table = np.loadtxt(
                path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2
            )
            src = table[:, 0].astype(np.int64)
            dst = table[:, 1].astype(np.int64)
            dst = dst + int(src.max()) + 1
            t = table[:, 2].astype(np.int64)
            msg = table[:, 4:]
            w = np.ones(t.shape[0])
            del table
        else:
            # Nodes numbered in order of first appearance (source first).
            node_ids = _IdMap()
            if name in ("tgbl-review", "tgbl-coin"):
                # time, src, dst, weight
                t, src, dst, w = _parse_csv(path, [
                    (0, int, np.int64), (1, node_ids, np.int64),
                    (2, node_ids, np.int64), (3, float, np.float64),
                ])  # fmt: skip
                w[w == 0] = 1
                w = np.log2(w)
                msg = w.reshape(-1, 1)
            elif name == "tgbl-comment":
                # time, src, dst, number of words, score
                t, src, dst, num_words, w = _parse_csv(path, [
                    (0, int, np.int64), (1, node_ids, np.int64),
                    (2, node_ids, np.int64), (3, int, np.int64),
                    (4, int, np.int64),
                ])  # fmt: skip
                num_words = num_words / 5000
                w = w.astype(np.float64)
                msg = np.stack([num_words, num_words, w], axis=1)
            else:  # tgbl-flight
                # day, src, dst, callsign, typecode
                t, src, dst, msg = _parse_csv(path, [
                    (0, _FlightTime(), np.int64), (1, node_ids, np.int64),
                    (2, node_ids, np.int64),
                    ((3, 4), _FlightFeatures(), (np.float32, 16)),
                ])  # fmt: skip
                w = np.ones(t.shape[0])
                node_feat = self._flight_node_feat(node_ids)

        arrays = dict(
            src=src,
            dst=dst,
            t=t,
            msg=np.asarray(msg, dtype=np.float32).reshape(t.shape[0], -1),
            w=w.astype(np.float32),
        )
        if node_feat is not None:
            arrays["node_feat"] = node_feat.astype(np.float32)
        self._finalize(arrays)

    def _flight_node_feat(self, node_ids: dict) -> np.ndarray:
        """Airport features of ``"tgbl-flight"`` (``process_node_feat`` of
        TGB): one-hot type, one-hot continent, region code and coordinates."""
        rows = _read_csv(self.raw_file_path("airport_node_feat"))
        types, continents = {}, {}
        for row in rows:
            if row[0] in node_ids:
                types.setdefault(row[1], len(types))
                continents.setdefault(row[2], len(continents))
        node_feat = np.zeros((len(node_ids), 20))
        for row in rows:
            if row[0] not in node_ids:
                continue
            type_vec = np.zeros(len(types))
            type_vec[types[row[1]]] = 1
            cont_vec = np.zeros(len(continents))
            cont_vec[continents[row[2]]] = 1
            region = row[3] + "!" * max(0, 7 - len(row[3]))
            coords = np.array([float(row[4]), float(row[5])])
            node_feat[node_ids[row[0]]] = np.concatenate(
                (type_vec, cont_vec, _str_to_codes(region), coords)
            )
        return node_feat

    def raw_file_path(self, stem: str) -> str:
        return os.path.join(self.raw_path, f"{stem}{_suffix(self.version)}.csv")


class TGBNodePropPredDataset(_TGBDataset):
    """The dynamic node property prediction datasets of the `Temporal Graph
    Benchmark (TGB) <https://tgb.complexdatalab.com/>`_, as ``py-tgb``'s
    ``tgb.nodeproppred.dataset_pyg.PyGNodePropPredDataset``.

    The dataset is a single :class:`~mlx_graphs.data.TemporalData` stream of
    weighted events sorted by time (``msg`` holds the weights), split
    chronologically (``train_mask``, ``val_mask``, ``test_mask``). The task is
    to predict, at given timestamps, the distribution of the future
    interactions of some nodes over ``num_classes`` items (e.g. the share of
    each trading partner of a nation the next year, or of each music genre
    of a user the next week), evaluated with NDCG@10
    (:class:`~mlx_graphs.metrics.TGBEvaluator`).

    The labels form a stream of label timestamps (:attr:`label_timestamps`).
    As in ``py-tgb``, :meth:`get_node_label` returns the labels of the next
    label timestamp once the events reach it, and :meth:`reset_label_time`
    rewinds the stream at the start of every pass over the events.
    :meth:`node_labels` gives the labels of any label timestamp directly.

    =================  ==========  ===========  =========  ==========
    Name               #nodes      #events      #classes   #features
    =================  ==========  ===========  =========  ==========
    ``"tgbn-trade"``   255         468,245      255        1
    ``"tgbn-genre"``   1,505       17,858,395   513        1
    ``"tgbn-reddit"``  11,766      27,174,118   698        1
    ``"tgbn-token"``   61,756      72,936,998   1,001      2
    =================  ==========  ===========  =========  ==========

    The node ids are those of ``py-tgb``: the classes (genres, subreddits,
    tokens) are the nodes ``0`` to ``num_classes - 1`` and the users are
    numbered after them; the nations of ``"tgbn-trade"`` are both the nodes
    and the classes. The features are the edge weights (and, for
    ``"tgbn-token"``, whether the user is the sender), with the natural
    logarithm of the weights for ``"tgbn-token"`` (as ``py-tgb`` once its
    processed files are cached).

    Args:
        name: Name of the dataset, e.g. ``"tgbn-trade"``
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the ``TemporalData``
            after processing (it is applied every time the dataset is
            loaded, as the processed arrays are stored without it)
        transform: A function/transform applied to the ``TemporalData`` at
            every access

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import TGBNodePropPredDataset
        from mlx_graphs.loaders import TemporalDataLoader

        dataset = TGBNodePropPredDataset("tgbn-trade")
        data = dataset[0]
        train_data = data[dataset.train_mask]
        dataset.reset_label_time()
        label_t = dataset.get_label_time()
        for batch in TemporalDataLoader(train_data, batch_size=200):
            query_t = batch.t[-1].item()
            if query_t > label_t:
                label_ts, label_nodes, labels = dataset.get_node_label(query_t)
                label_t = dataset.get_label_time()
                ...  # predict `labels` [num_label_nodes, num_classes]

    References:
        - Huang et al., `Temporal Graph Benchmark for Machine Learning on
          Temporal Graphs <https://arxiv.org/abs/2307.01026>`_
        - `py-tgb <https://github.com/shenyangHuang/TGB>`_
    """

    names = _NODE_NUM_CLASSES

    def __init__(
        self,
        name: str,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(name, base_dir, pre_transform, transform)
        self.label_ts_idx = 0

    @property
    def raw_file_names(self) -> list[str]:
        return [
            f"{self.tgb_name}_edgelist.csv",
            f"{self.tgb_name}_node_labels.csv",
        ]

    @property
    def num_classes(self) -> int:
        """The number of classes of the node labels (as ``py-tgb``)."""
        return _NODE_NUM_CLASSES[self.tgb_name]

    @property
    def label_dim(self) -> int:
        """The size of the label vectors (the number of classes seen in the
        events, at most :attr:`num_classes`)."""
        return int(self.arrays["label_dim"])

    @property
    def label_timestamps(self) -> np.ndarray:
        """The sorted timestamps at which there are node labels."""
        return self.arrays["label_ts"]

    @property
    def num_label_timestamps(self) -> int:
        """The number of label timestamps."""
        return int(self.arrays["label_ts"].shape[0])

    def node_labels(self, i: int) -> tuple[mx.array, mx.array, mx.array]:
        """The node labels of the ``i``-th label timestamp.

        Args:
            i: Index in :attr:`label_timestamps`

        Returns:
            The timestamp repeated for every labeled node, the labeled nodes,
            and their label vectors (float32, shape `[num_nodes, label_dim]`),
            in the order of ``py-tgb``.
        """
        a = self.arrays
        start, end = int(a["label_ptr"][i]), int(a["label_ptr"][i + 1])
        nodes = a["label_node"][start:end]
        labels = np.zeros((end - start, self.label_dim), dtype=np.float32)
        e_start, e_end = a["label_entry_ptr"][start], a["label_entry_ptr"][end]
        rows = np.repeat(
            np.arange(end - start), np.diff(a["label_entry_ptr"][start : end + 1])
        )
        labels[rows, a["label_class"][e_start:e_end]] = a["label_value"][e_start:e_end]
        ts = np.full(end - start, a["label_ts"][i], dtype=np.int64)
        return mx.array(ts), mx.array(nodes.astype(np.int32)), mx.array(labels)

    def get_node_label(
        self, cur_t: Union[int, mx.array]
    ) -> Optional[tuple[mx.array, mx.array, mx.array]]:
        """The labels of the next label timestamp if ``cur_t`` reached it, as
        ``py-tgb``'s ``get_node_label``: the label stream then moves to the
        following label timestamp. Returns ``None`` otherwise (or after the
        last label timestamp).

        Args:
            cur_t: The current time (e.g. the time of the last event of a
                mini-batch)

        Returns:
            ``(label_ts, label_nodes, labels)``, see :meth:`node_labels`.
        """
        if isinstance(cur_t, mx.array):
            cur_t = cur_t.item()
        if self.label_ts_idx >= self.num_label_timestamps:
            return None
        if cur_t >= self.arrays["label_ts"][self.label_ts_idx]:
            self.label_ts_idx += 1
            return self.node_labels(self.label_ts_idx - 1)
        return None

    def get_label_time(self) -> int:
        """The next label timestamp of the stream (the last one once the
        stream is exhausted)."""
        label_ts = self.arrays["label_ts"]
        return int(label_ts[min(self.label_ts_idx, len(label_ts) - 1)])

    def reset_label_time(self):
        """Rewinds the label stream to the first label timestamp."""
        self.label_ts_idx = 0

    def process(self):
        name = self.tgb_name
        path = self.raw_file("edgelist") + ".csv"
        if name == "tgbn-trade":
            # year, nation, trading nation, weight: the nations are both the
            # nodes and the classes
            node_ids = label_ids = _IdMap()
            t, src, dst, w = _parse_csv(path, [
                (0, int, np.int64), (1, node_ids, np.int64),
                (2, node_ids, np.int64), (3, float, np.float64),
            ])  # fmt: skip
            msg = w.reshape(-1, 1)
        else:
            # The classes are numbered from 0 and the users after them.
            node_ids = _IdMap(start=_NODE_NUM_CLASSES[name])
            label_ids = _IdMap()
            columns = [
                (0, int if name == "tgbn-genre" else float, np.float64),
                (1, node_ids, np.int64),
                (2, label_ids, np.int64),
            ]
            if name == "tgbn-genre":
                # time, user, genre, weight
                t, src, dst, w = _parse_csv(path, columns + [(3, float, np.float64)])
                msg = w.reshape(-1, 1)
            elif name == "tgbn-reddit":
                # time, user, subreddit, number of words, score
                t, src, dst, score = _parse_csv(path, columns + [(4, int, np.int64)])
                msg = score.astype(np.float64).reshape(-1, 1)
            else:  # tgbn-token: time, user, token, value, is sender
                t, src, dst, value, sender = _parse_csv(
                    path, columns + [(3, float, np.float64), (4, float, np.float64)]
                )
                msg = np.stack([np.log(value), sender], axis=1)
            t = t.astype(np.int64)

        arrays = dict(src=src, dst=dst, t=t, msg=msg.astype(np.float32))
        arrays.update(self._process_labels(node_ids, label_ids))
        self._finalize(arrays)

    def _process_labels(self, node_ids: dict, label_ids: dict) -> dict:
        """The label stream: for each label timestamp (sorted), the labeled
        nodes in order of first appearance and a sparse label vector for each
        of them (the last weight of every (time, node, class) of the file)."""
        ts, node, cls, value = _parse_csv(
            os.path.join(self.raw_path, self.raw_file_names[1]),
            [
                (0, int, np.int64),
                (1, node_ids.__getitem__, np.int64),
                (2, label_ids.__getitem__, np.int64),
                (3, float, np.float64),
            ],
        )

        # (time, node) pairs, in order of first appearance, then by time.
        pairs = _make_keys([ts, node])
        uniq, first, inv = np.unique(pairs, return_index=True, return_inverse=True)
        order = np.lexsort((first, uniq["f0"]))
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        pair = rank[inv.reshape(-1)]
        pair_ts = uniq["f0"][order]
        pair_node = uniq["f1"][order]

        # (pair, class) entries, keeping the last weight of duplicates.
        entries = _make_keys([pair[::-1], cls[::-1]])
        uniq_e, last = np.unique(entries, return_index=True)
        entry_value = value[::-1][last]
        entry_pair, entry_class = uniq_e["f0"], uniq_e["f1"]

        label_ts, label_ptr = np.unique(pair_ts, return_index=True)
        return dict(
            label_ts=label_ts,
            label_ptr=np.append(label_ptr, pair_ts.size).astype(np.int64),
            label_node=pair_node,
            label_entry_ptr=np.searchsorted(
                entry_pair, np.arange(pair_ts.size + 1)
            ).astype(np.int64),
            label_class=entry_class,
            label_value=entry_value,
            label_dim=np.array(len(label_ids)),
        )
