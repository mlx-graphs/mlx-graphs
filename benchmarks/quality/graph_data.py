"""Data of the graph-level / heterogeneous / link / KG quality checks.

Every dataset is loaded once through mlx-graphs, turned into numpy arrays and
cached in an ``.npz`` file; both frameworks then build their inputs from the
very same arrays (see ``graph_worker.py``). Loading through mlx-graphs also
exercises its dataset classes (``TUDataset``, ``IMDB``, ``DBLP``,
``FB15k_237``, ``MovieLens100K``, ``PlanetoidDataset``) and transforms
(``RandomLinkSplit``, ``train_test_split_edges``, ``to_homogeneous``).
"""

import contextlib
import io
import os

import numpy as np

# Datasets are shared by every checkout that sets MLX_GRAPHS_BENCH_DATA (e.g.
# worktrees), and downloaded only once.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.environ.get("MLX_GRAPHS_BENCH_DATA") or os.path.join(
    _REPO, ".mlx_graphs_data"
)
# mlx-graphs datasets of this harness live in their own folder, so that they
# are processed by the checked-out version of the library.
MLX_DIR = os.path.join(DATA_DIR, "quality_graph")
CACHE_DIR = os.path.join(MLX_DIR, "npz")


def _quiet():
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


def _cached(name, build):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.npz")
    if not os.path.exists(path):
        arrays = build()
        np.savez(path + ".tmp.npz", **arrays)
        os.replace(path + ".tmp.npz", path)
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


# ---------------------------------------------------------------------------
# TU graph classification datasets
# ---------------------------------------------------------------------------


def tu_dataset(name: str) -> dict:
    """Graphs of a TU dataset as concatenated arrays plus per-graph offsets.

    Returns a dict with ``x`` [N, F], ``edge_index`` [2, E] (local node ids),
    optional ``edge_attr`` [E, D], ``y`` [G], ``node_ptr`` / ``edge_ptr``
    [G + 1].
    """

    def build():
        from mlx_graphs.datasets import TUDataset

        with _quiet():
            dataset = TUDataset(name, base_dir=MLX_DIR)
        xs, eis, eas, ys = [], [], [], []
        for g in dataset:
            xs.append(np.array(g.node_features, dtype=np.float32))
            eis.append(np.array(g.edge_index, dtype=np.int64))
            if g.edge_features is not None:
                eas.append(np.array(g.edge_features, dtype=np.float32))
            ys.append(int(np.array(g.graph_labels).reshape(-1)[0]))
        out = {
            "x": np.concatenate(xs),
            "edge_index": np.concatenate(eis, axis=1),
            "y": np.array(ys, dtype=np.int64),
            "node_ptr": np.cumsum([0] + [len(x) for x in xs]),
            "edge_ptr": np.cumsum([0] + [e.shape[1] for e in eis]),
        }
        if eas:
            out["edge_attr"] = np.concatenate(eas)
        return out

    return _cached(f"tu_{name}", build)


def tu_graphs(data: dict) -> list[dict]:
    """Splits the arrays of :func:`tu_dataset` into one dict per graph."""
    graphs = []
    for i in range(len(data["y"])):
        n0, n1 = data["node_ptr"][i], data["node_ptr"][i + 1]
        e0, e1 = data["edge_ptr"][i], data["edge_ptr"][i + 1]
        g = {
            "x": data["x"][n0:n1],
            "edge_index": data["edge_index"][:, e0:e1],
            "y": data["y"][i : i + 1],
        }
        if "edge_attr" in data:
            g["edge_attr"] = data["edge_attr"][e0:e1]
        graphs.append(g)
    return graphs


def split_indices(seed: int, size: int, train=0.8, val=0.1):
    """Random train / validation / test split of ``range(size)``."""
    perm = np.random.default_rng(seed).permutation(size)
    n_train, n_val = int(train * size), int(val * size)
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def degree_histogram(graphs: list[dict]) -> np.ndarray:
    """In-degree histogram of a list of graphs (PNA's ``deg``)."""
    max_deg, hists = 0, []
    for g in graphs:
        deg = np.bincount(g["edge_index"][1], minlength=len(g["x"]))
        hists.append(np.bincount(deg))
        max_deg = max(max_deg, len(hists[-1]))
    out = np.zeros(max_deg, dtype=np.int64)
    for h in hists:
        out[: len(h)] += h
    return out


# ---------------------------------------------------------------------------
# Heterogeneous graphs
# ---------------------------------------------------------------------------


def imdb() -> dict:
    """IMDB with the metapath graphs of PyG's ``han_imdb.py`` example.

    ``AddMetaPaths([movie, actor, movie], [movie, director, movie])`` with
    ``drop_orig_edge_types=True``: the movies are linked when they share an
    actor or a director (self-loops included, as in PyG), and the other node
    types are dropped.
    """

    def build():
        import scipy.sparse as sp

        from mlx_graphs.datasets import IMDB

        with _quiet():
            g = IMDB(base_dir=MLX_DIR)[0]
        n_movie = g.node_features_dict["movie"].shape[0]
        labels = g.node_labels_dict.get("movie", g.node_labels_dict.get("movies"))
        out = {
            "x": np.array(g.node_features_dict["movie"], dtype=np.float32),
            "y": np.array(labels, dtype=np.int64),
        }
        for split in ("train", "val", "test"):
            out[f"{split}_mask"] = np.array(getattr(g, f"movie_{split}_mask"))
        for other in ("actor", "director"):
            ei = np.array(g.edge_index_dict[("movie", "to", other)])
            n_other = g.node_features_dict[other].shape[0]
            a = sp.coo_matrix(
                (np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n_movie, n_other)
            ).tocsr()
            mp = (a @ a.T).tocoo()
            order = np.lexsort((mp.col, mp.row))
            out[f"metapath_{other}"] = np.stack([mp.row[order], mp.col[order]]).astype(
                np.int64
            )
        return out

    return _cached("imdb_han", build)


def dblp() -> dict:
    """DBLP as per-type arrays: ``x__<type>``, ``ei__<src>__<dst>``, labels
    and masks of the authors. Conference nodes get a constant feature, like
    ``T.Constant(node_types='conference')`` in PyG's ``hgt_dblp.py``."""

    def build():
        from mlx_graphs.datasets import DBLP

        with _quiet():
            g = DBLP(base_dir=MLX_DIR)[0]
        out = {}
        for t, x in g.node_features_dict.items():
            out[f"x__{t}"] = np.array(x, dtype=np.float32)
        n_conf = 0
        for (s, _, d), ei in g.edge_index_dict.items():
            ei = np.array(ei, dtype=np.int64)
            out[f"ei__{s}__{d}"] = ei
            if s == "conference":
                n_conf = max(n_conf, int(ei[0].max()) + 1)
        out["x__conference"] = np.ones((n_conf, 1), dtype=np.float32)
        out["y"] = np.array(g.node_labels_dict["author"], dtype=np.int64)
        for split in ("train", "val", "test"):
            out[f"{split}_mask"] = np.array(getattr(g, f"author_{split}_mask"))
        return out

    return _cached("dblp", build)


def dblp_hetero(data: dict):
    """(x_dict, edge_index_dict) of :func:`dblp` as numpy arrays."""
    x = {k[3:]: v for k, v in data.items() if k.startswith("x__")}
    ei = {}
    for k, v in data.items():
        if k.startswith("ei__"):
            _, s, d = k.split("__")
            ei[(s, "to", d)] = v
    return x, ei


def dblp_homogeneous(feature_dim: int = 32) -> dict:
    """DBLP converted with ``HeteroGraphData.to_homogeneous()``.

    The bag-of-words features of authors and papers (334 and 4,231 wide)
    are first reduced to ``feature_dim`` by a fixed random projection, so
    that per-relation layers (whose PyG implementations materialize a weight
    per edge) fit in memory. Terms keep their 50 features and conferences
    their constant one; ``to_homogeneous`` zero-pads them to the widest.
    """

    def build():
        import mlx.core as mx

        from mlx_graphs.data import HeteroGraphData

        data = dblp()
        x, ei = dblp_hetero(data)
        rng = np.random.default_rng(0)
        for t in ("author", "paper"):
            proj = rng.normal(size=(x[t].shape[1], feature_dim)) / np.sqrt(feature_dim)
            x[t] = (x[t] @ proj).astype(np.float32)
        n_author = len(x["author"])
        hetero = HeteroGraphData(
            edge_index_dict={k: mx.array(v.astype(np.int32)) for k, v in ei.items()},
            node_features_dict={k: mx.array(v) for k, v in x.items()},
            node_labels_dict={"author": mx.array(data["y"].astype(np.int32))},
        )
        homo = hetero.to_homogeneous()
        types = list(hetero.node_types)
        start = sum(len(x[t]) for t in types[: types.index("author")])
        out = {
            "x": np.array(homo.node_features, dtype=np.float32),
            "edge_index": np.array(homo.edge_index, dtype=np.int64),
            "edge_type": np.array(homo.edge_type, dtype=np.int64),
            "node_type": np.array(homo.node_type, dtype=np.int64),
            "y": np.array(homo.node_labels, dtype=np.int64),
        }
        assert np.array_equal(out["y"][start : start + n_author], data["y"])
        for split in ("train", "val", "test"):
            idx = np.nonzero(data[f"{split}_mask"])[0] + start
            out[f"{split}_idx"] = idx.astype(np.int64)
        return out

    return _cached(f"dblp_homogeneous_{feature_dim}", build)


# ---------------------------------------------------------------------------
# Citation graphs and link splits
# ---------------------------------------------------------------------------


def planetoid(name: str) -> dict:
    """Cora / CiteSeer / PubMed with row-normalized features (as
    ``T.NormalizeFeatures()``) and the public split masks."""

    def build():
        from mlx_graphs.datasets import PlanetoidDataset

        with _quiet():
            g = PlanetoidDataset(name.lower(), base_dir=MLX_DIR)[0]
        x = np.array(g.node_features, dtype=np.float32)
        x = x / np.maximum(x.sum(1, keepdims=True), 1e-12)
        return {
            "x": x,
            "edge_index": np.array(g.edge_index, dtype=np.int64),
            "y": np.array(g.node_labels, dtype=np.int64).reshape(-1),
            "train_mask": np.array(g.train_mask),
            "val_mask": np.array(g.val_mask),
            "test_mask": np.array(g.test_mask),
        }

    return _cached(f"planetoid_{name.lower()}", build)


def link_split(name: str, seed: int, method: str = "random_link_split") -> dict:
    """Train / val / test edges of PyG's ``autoencoder.py`` example.

    ``random_link_split``: mlx-graphs' ``RandomLinkSplit(num_val=0.05,
    num_test=0.1, is_undirected=True, split_labels=True,
    add_negative_train_samples=False)``. ``train_test_split_edges``: the
    older function with the same ratios.
    """

    def build():
        import mlx.core as mx

        from mlx_graphs.data import GraphData

        mx.random.seed(seed)
        np.random.seed(seed)
        data = planetoid(name)
        graph = GraphData(
            edge_index=mx.array(data["edge_index"].astype(np.int32)),
            node_features=mx.array(data["x"]),
        )
        if method == "random_link_split":
            from mlx_graphs.transforms import RandomLinkSplit

            transform = RandomLinkSplit(
                num_val=0.05,
                num_test=0.1,
                is_undirected=True,
                add_negative_train_samples=False,
            )
            train, val, test = transform(graph)

            def labelled(split, label):
                index = np.array(split.edge_label_index)
                return index[:, np.array(split.edge_label) == label]

            out = {
                "train_edge_index": train.edge_index,
                "train_pos": labelled(train, 1),
                "val_pos": labelled(val, 1),
                "val_neg": labelled(val, 0),
                "test_pos": labelled(test, 1),
                "test_neg": labelled(test, 0),
            }
        else:
            from mlx_graphs.utils import train_test_split_edges

            g = train_test_split_edges(graph, val_ratio=0.05, test_ratio=0.1)
            out = {
                "train_edge_index": g.train_pos_edge_index,
                "train_pos": g.train_pos_edge_index,
                "val_pos": g.val_pos_edge_index,
                "val_neg": g.val_neg_edge_index,
                "test_pos": g.test_pos_edge_index,
                "test_neg": g.test_neg_edge_index,
            }
        return {k: np.array(v, dtype=np.int64) for k, v in out.items()}

    return _cached(f"link_{name.lower()}_{method}_{seed}", build)


# ---------------------------------------------------------------------------
# Recommendation and knowledge graphs
# ---------------------------------------------------------------------------


def movielens() -> dict:
    """MovieLens100K's ``u1.base`` (train) / ``u1.test`` ratings, kept as
    implicit feedback when the rating is at least 4."""

    def build():
        from mlx_graphs.datasets import MovieLens100K

        with _quiet():
            g = MovieLens100K(base_dir=MLX_DIR)[0]
        key = ("user", "rates", "movie")
        train = np.array(g.edge_index_dict[key], dtype=np.int64)
        rating = np.array(g.user_rates_movie_rating).reshape(-1)
        test = np.array(g.edge_labels_index_dict[key], dtype=np.int64)
        test_rating = np.array(g.edge_labels_dict[key]).reshape(-1)
        return {
            "train": train[:, rating >= 4],
            "test": test[:, test_rating >= 4],
            "num_users": np.array(g.node_features_dict["user"].shape[0]),
            "num_movies": np.array(g.node_features_dict["movie"].shape[0]),
        }

    return _cached("movielens", build)


def fb15k_237() -> dict:
    """FB15k-237 triplets: ``<split>_head``, ``<split>_rel``, ``<split>_tail``."""

    def build():
        from mlx_graphs.datasets import FB15k_237

        out = {}
        for split in ("train", "val", "test"):
            with _quiet():
                g = FB15k_237(split=split, base_dir=MLX_DIR)[0]
            ei = np.array(g.edge_index, dtype=np.int64)
            out[f"{split}_head"], out[f"{split}_tail"] = ei
            out[f"{split}_rel"] = np.array(g.edge_type, dtype=np.int64)
        out["num_nodes"] = np.array(g.num_nodes)
        out["num_relations"] = np.array(
            1 + max(int(out[f"{s}_rel"].max()) for s in ("train", "val", "test"))
        )
        return out

    return _cached("fb15k_237", build)


# ---------------------------------------------------------------------------
# Synthetic point clouds
# ---------------------------------------------------------------------------


def point_clouds(num_clouds: int = 1000, num_points: int = 64, k: int = 8) -> dict:
    """Randomly rotated, noisy point clouds of 4 shapes (sphere, cube
    surface, cylinder, cone), and their k-nearest-neighbor graphs built with
    mlx-graphs' ``knn_graph`` (checked against a brute-force numpy kNN)."""

    def build():
        import mlx.core as mx

        from mlx_graphs.utils import knn_graph

        rng = np.random.default_rng(0)
        rows = np.arange(num_points)

        def shape(c):
            u = rng.uniform(-1, 1, (num_points, 3))
            if c == 0:  # sphere
                p = u / np.linalg.norm(u, axis=1, keepdims=True)
            elif c == 1:  # cube surface
                axis = rng.integers(0, 3, num_points)
                p = u.copy()
                p[rows, axis] = np.sign(u[rows, axis])
            else:  # cylinder / cone
                theta = rng.uniform(0, 2 * np.pi, num_points)
                h = rng.uniform(-1, 1, num_points)
                r = np.ones(num_points) if c == 2 else (1 - h) / 2
                p = np.stack([r * np.cos(theta), r * np.sin(theta), h], axis=1)
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))  # random rotation
            noise = rng.normal(scale=0.05, size=p.shape)
            return (p @ q + noise).astype(np.float32)

        y = rng.integers(0, 4, num_clouds)
        pos = np.stack([shape(c) for c in y])  # [clouds, points, 3]
        batch = np.repeat(np.arange(num_clouds), num_points)
        ei = knn_graph(
            mx.array(pos.reshape(-1, 3)), k, mx.array(batch.astype(np.int32))
        )
        ei = np.array(ei, dtype=np.int64)
        # brute-force check of the first cloud: edges from the k nearest points
        d = ((pos[0][:, None] - pos[0][None]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        expected = {(int(j), i) for i in rows for j in np.argsort(d[i])[:k]}
        first = ei[:, ei[1] < num_points]
        assert {tuple(e) for e in first.T.tolist()} == expected, "knn_graph mismatch"
        # edges grouped by cloud, with local node indices
        ei = ei[:, np.argsort(batch[ei[1]], kind="stable")]
        local = ei - batch[ei[1]] * num_points
        return {"pos": pos, "y": y, "edge_index": local}

    return _cached(f"point_clouds_{num_clouds}_{num_points}_{k}", build)
