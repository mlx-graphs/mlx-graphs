"""Helpers of the notebook ``graph_rag_g_reasoner.ipynb`` that are not specific
to G-reasoner: text embeddings with mlx-lm and a reader for the released
PyTorch checkpoint."""

import pickle
import zipfile
from collections import OrderedDict

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from mlx.utils import tree_unflatten


def embed(model, tokenizer, texts, batch_size=64):
    """Embeds texts with a Qwen3-Embedding model loaded by ``mlx_lm.load``.

    The embedding of a text is the hidden state of its last token, normalized.
    """
    tokens = [tokenizer.encode(text) for text in texts]
    order = np.argsort([len(t) for t in tokens])  # batches of similar length
    out = np.zeros((len(texts), 1024), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        ids = order[start : start + batch_size]
        batch = [tokens[i] for i in ids]
        width = max(len(t) for t in batch)
        # Causal attention: the padding on the right is never attended to.
        states = model.model(mx.array([t + [0] * (width - len(t)) for t in batch]))
        last = states[mx.arange(len(batch)), mx.array([len(t) - 1 for t in batch])]
        last = last.astype(mx.float32)
        out[ids] = np.array(last / mx.linalg.norm(last, axis=-1, keepdims=True))
    return out


def read_torch_checkpoint(path):
    """Reads the tensors of a PyTorch checkpoint with NumPy.

    The file is a zip archive with one raw buffer per tensor and a pickle that
    describes them. Any class that a dictionary of tensors does not need is
    refused.
    """

    def tensor(storage, offset, size, *unused):
        return storage[offset : offset + int(np.prod(size))].reshape(size)

    allowed = {
        ("torch._utils", "_rebuild_tensor_v2"): tensor,
        ("torch", "FloatStorage"): np.float32,
        ("collections", "OrderedDict"): OrderedDict,
    }
    with zipfile.ZipFile(path) as archive:
        pickle_name = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        root = pickle_name[: -len("data.pkl")]

        class Reader(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) not in allowed:
                    raise pickle.UnpicklingError(f"unexpected class {module}.{name}")
                return allowed[(module, name)]

            def persistent_load(self, pid):
                buffer = archive.read(f"{root}data/{pid[2]}")
                return np.frombuffer(buffer, dtype=pid[1])

        with archive.open(pickle_name) as f:
            return Reader(f).load()


def released_weights(repo="rmanluo/G-reasoner-34M"):
    """The released G-reasoner weights, named for the ``GReasoner`` class of
    the notebook: pass the result to ``model.update``."""
    checkpoint = read_torch_checkpoint(hf_hub_download(repo, "model.pth"))
    weights = []
    for name, value in checkpoint["model"].items():
        name = name.replace("entity_model.", "")
        if "relation_projection." in name:
            # layers.3.relation_projection.0.weight
            #   -> relation_projections.3.layers.0.weight
            _, index, _, rest = name.split(".", 3)
            name = f"relation_projections.{index}.layers.{rest}"
        for block in ("early_fuse_mlp.", "predict_mlp."):
            name = name.replace(block, block + "layers.")
        weights.append((name, mx.array(value)))
    return tree_unflatten(weights)
