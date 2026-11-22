import os
from typing import Callable, Literal, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download

KG_SPLITS = Literal["train", "val", "test"]

_KG_URL = (
    "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master"
)
_RAW_FILE_NAMES = ["train.txt", "valid.txt", "test.txt"]


class _KnowledgeGraphDataset(Dataset):
    """A knowledge graph stored as ``train.txt``, ``valid.txt`` and
    ``test.txt`` triplet files."""

    _url: str = ""

    def download(self):
        for file_name in _RAW_FILE_NAMES:
            download(
                f"{self._url}/{file_name}", path=os.path.join(self.raw_path, file_name)
            )

    def _read_raw(self, file_name: str) -> str:
        with open(os.path.join(self.raw_path, file_name)) as f:
            return f.read()


def _merge_splits(
    srcs: list[np.ndarray], dsts: list[np.ndarray], edge_types: list[np.ndarray]
) -> GraphData:
    """Merges the train/val/test triplets into one graph sorted by
    (source, destination), with one boolean mask per split."""
    src, dst = np.concatenate(srcs), np.concatenate(dsts)
    edge_type = np.concatenate(edge_types)
    split = np.repeat(np.arange(3), [s.size for s in srcs])

    num_nodes = int(max(src.max(), dst.max())) + 1
    perm = np.argsort(num_nodes * src.astype(np.int64) + dst, kind="stable")

    return GraphData(
        edge_index=mx.array(np.stack([src[perm], dst[perm]]).astype(np.int32)),
        edge_type=mx.array(edge_type[perm].astype(np.int32)),
        train_mask=mx.array(split[perm] == 0),
        val_mask=mx.array(split[perm] == 1),
        test_mask=mx.array(split[perm] == 2),
        num_nodes=num_nodes,
    )


class FB15k_237(_KnowledgeGraphDataset):
    """The FB15K237 dataset from the `"Translating Embeddings for Modeling
    Multi-Relational Data"
    <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`_
    paper, containing 14,541 entities, 237 relations and 310,116 fact triples.

    Each split is a separate graph whose ``edge_index`` holds the (head, tail)
    entity pairs and ``edge_type`` the relation of each triplet. Entity and
    relation ids are shared across the three splits, and each graph has
    ``num_nodes`` set to the total number of entities.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.FB15k_237.html>`_.

    .. note::

        The original :class:`FB15k` dataset suffers from major test leakage
        through inverse relations, where a large number of test triples could
        be obtained by inverting triples in the training set. In order to
        create a dataset without this characteristic, :class:`FB15k_237`
        contains a subset of FB15k where inverse relations are removed.

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - Split
          - #nodes
          - #edges
          - #relations
        * - train
          - 14,541
          - 272,115
          - 237
        * - val
          - 14,541
          - 17,535
          - 237
        * - test
          - 14,541
          - 20,466
          - 237

    Args:
        split: The split to load (``"train"``, ``"val"``, ``"test"``).
            Default: ``"train"``.
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import FB15k_237

        dataset = FB15k_237(split="train")
        >>> FB15k_237(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 272115), int32)
                edge_type(shape=(272115,), int32)
                num_nodes=14541)

    References:
        - `Bordes et al., "Translating Embeddings for Modeling
          Multi-Relational Data", NeurIPS 2013
          <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`_
        - `Toutanova and Chen, "Observed versus latent features for knowledge
          base and text inference", CVSC 2015
          <https://aclanthology.org/W15-4007/>`_
    """

    _url = f"{_KG_URL}/FB15k-237"

    def __init__(
        self,
        split: KG_SPLITS = "train",
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        splits = ["train", "val", "test"]
        if split not in splits:
            raise ValueError(f"Invalid 'split' argument (got {split})")
        self.split = split
        super().__init__(
            name="FB15k_237",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )
        # the three splits are processed and saved together, keep only one
        self.graphs = [self.graphs[splits.index(split)]]

    def process(self):
        node_dict: dict[str, int] = {}
        rel_dict: dict[str, int] = {}
        graphs = []
        for file_name in _RAW_FILE_NAMES:
            lines = [x.split("\t") for x in self._read_raw(file_name).split("\n")[:-1]]
            edges = np.empty((2, len(lines)), dtype=np.int32)
            edge_type = np.empty(len(lines), dtype=np.int32)
            for i, (src, rel, dst) in enumerate(lines):
                edges[0, i] = node_dict.setdefault(src, len(node_dict))
                edges[1, i] = node_dict.setdefault(dst, len(node_dict))
                edge_type[i] = rel_dict.setdefault(rel, len(rel_dict))
            graphs.append((edges, edge_type))

        self.graphs = [
            GraphData(
                edge_index=mx.array(edges),
                edge_type=mx.array(edge_type),
                num_nodes=len(node_dict),
            )
            for edges, edge_type in graphs
        ]


class WordNet18(_KnowledgeGraphDataset):
    """The WordNet18 dataset from the `"Translating Embeddings for Modeling
    Multi-Relational Data"
    <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`_
    paper, containing 40,943 entities, 18 relations and 151,442 fact triplets,
    *e.g.*, furniture includes bed.

    The three splits are merged in a single graph sorted by
    (head, tail), with ``edge_type`` holding the relation of each triplet and
    the ``train_mask``, ``val_mask`` and ``test_mask`` edge masks telling the
    split of each triplet.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WordNet18.html>`_.

    .. note::

        The original :class:`WordNet18` dataset suffers from test leakage, *i.e.*
        more than 80% of test triplets can be found in the training set with
        another relation type. Therefore, it should not be used for research
        evaluation anymore. We recommend to use its cleaned version
        :class:`WordNet18RR` instead.

    Args:
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import WordNet18

        dataset = WordNet18()
        >>> WordNet18(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 151442), int32)
                edge_type(shape=(151442,), int32)
                train_mask(shape=(151442,), bool)
                val_mask(shape=(151442,), bool)
                test_mask(shape=(151442,), bool)
                num_nodes=40943)

    References:
        - `Bordes et al., "Translating Embeddings for Modeling
          Multi-Relational Data", NeurIPS 2013
          <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`_
    """

    _url = f"{_KG_URL}/WN18/original"

    def __init__(
        self,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            name="WordNet18",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    def process(self):
        srcs, dsts, edge_types = [], [], []
        for file_name in _RAW_FILE_NAMES:
            # first number is the number of triplets, then "head tail relation"
            edges = np.array(self._read_raw(file_name).split()[1:], dtype=np.int64)
            srcs.append(edges[::3])
            dsts.append(edges[1::3])
            edge_types.append(edges[2::3])
        self.graphs = [_merge_splits(srcs, dsts, edge_types)]


class WordNet18RR(_KnowledgeGraphDataset):
    """The WordNet18RR dataset from the `"Convolutional 2D Knowledge Graph
    Embeddings" <https://arxiv.org/abs/1707.01476>`_ paper, containing 40,943
    entities, 11 relations and 93,003 fact triplets.

    The three splits are merged in a single graph sorted by
    (head, tail), with ``edge_type`` holding the relation of each triplet and
    the ``train_mask``, ``val_mask`` and ``test_mask`` edge masks telling the
    split of each triplet. The relation ids are given by
    :attr:`WordNet18RR.edge2id`.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WordNet18RR.html>`_.

    Args:
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import WordNet18RR

        dataset = WordNet18RR()
        >>> WordNet18RR(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 93003), int32)
                edge_type(shape=(93003,), int32)
                train_mask(shape=(93003,), bool)
                val_mask(shape=(93003,), bool)
                test_mask(shape=(93003,), bool)
                num_nodes=40943)

    References:
        - `Dettmers et al., "Convolutional 2D Knowledge Graph Embeddings",
          AAAI 2018 <https://arxiv.org/abs/1707.01476>`_
    """

    _url = f"{_KG_URL}/WN18RR/original"

    edge2id = {
        "_also_see": 0,
        "_derivationally_related_form": 1,
        "_has_part": 2,
        "_hypernym": 3,
        "_instance_hypernym": 4,
        "_member_meronym": 5,
        "_member_of_domain_region": 6,
        "_member_of_domain_usage": 7,
        "_similar_to": 8,
        "_synset_domain_topic_of": 9,
        "_verb_group": 10,
    }

    def __init__(
        self,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            name="WordNet18RR",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    def process(self):
        node2id: dict[str, int] = {}
        srcs, dsts, edge_types = [], [], []
        for file_name in _RAW_FILE_NAMES:
            # "head relation tail" triplets
            tokens = self._read_raw(file_name).split()
            heads, rels, tails = tokens[::3], tokens[1::3], tokens[2::3]
            # node ids are given by order of first appearance, heads first
            for node in heads + tails:
                node2id.setdefault(node, len(node2id))
            srcs.append(np.array([node2id[n] for n in heads], dtype=np.int64))
            dsts.append(np.array([node2id[n] for n in tails], dtype=np.int64))
            edge_types.append(np.array([self.edge2id[r] for r in rels]))
        self.graphs = [_merge_splits(srcs, dsts, edge_types)]
