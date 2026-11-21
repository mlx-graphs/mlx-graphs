"""Exports the processed Peptides-func splits (with RWSE) to npz, so the PyG
twin does not need to import MLX."""

import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from examples.graph_gps_peptides import WALK_LENGTH  # noqa
from mlx_graphs.datasets import LRGBDataset  # noqa
from mlx_graphs.transforms import AddRandomWalkPE  # noqa

for split in ["train", "val", "test"]:
    graphs = LRGBDataset(
        "peptides-func",
        split,
        base_dir=sys.argv[1],
        pre_transform=AddRandomWalkPE(WALK_LENGTH),
    ).graphs
    out = {k: [] for k in ["x", "ei", "ea", "y", "pe"]}
    nn, ne = [], []
    for g in graphs:
        out["x"].append(np.array(g.node_features))
        out["ei"].append(np.array(g.edge_index).T)
        out["ea"].append(np.array(g.edge_features))
        out["y"].append(np.array(g.graph_labels))
        out["pe"].append(np.array(g.random_walk_pe))
        nn.append(g.node_features.shape[0])
        ne.append(g.edge_index.shape[1])
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(
        os.path.join(sys.argv[1], f"{split}.npz"),
        nn=np.array(nn),
        ne=np.array(ne),
        **arrays,
    )
    print(split, len(graphs), flush=True)
