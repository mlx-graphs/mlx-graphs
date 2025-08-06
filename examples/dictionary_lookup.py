import itertools
import math
import random
from multiprocessing import Process, Queue
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from mlx_graphs.nn import GATConv, GATv2Conv

"""
Dictionary Lookup problem "Synthetic Benchmark: DictionaryLookup"
    from "How Attentive Are Graph Attention Networks?"

Minimal version based off of https://github.com/tech-srl/how_attentive_are_gats
    but modified to showcase difference in accuracy from GATConv to GATv2Conv
"""


class Model(nn.Module):
    def __init__(
        self, k: int, hidden_states: int = 128, dynamic_attention: bool = True
    ):
        super().__init__()
        self.k = k
        self.hidden_states = hidden_states
        self.dynamic_attention = dynamic_attention

        # +1 corresponds to "empty" for keys
        self.keys0 = nn.Embedding(k + 1, self.hidden_states)
        self.values0 = nn.Embedding(k + 1, self.hidden_states)
        self.ff0 = nn.Sequential(nn.ReLU())

        if self.dynamic_attention:
            self.gnn = GATv2Conv(self.hidden_states, self.hidden_states, heads=1)
        else:
            self.gnn = GATConv(self.hidden_states, self.hidden_states, heads=1)

        self.ffn = nn.Linear(self.hidden_states, k)

    def __call__(self, edges: mx.array, nodes: mx.array):
        x_key, x_val = nodes[:, 0], nodes[:, 1]
        x_key_embed = self.keys0(x_key)
        x_val_embed = self.values0(x_val)

        x = x_key_embed + x_val_embed
        x = self.ff0(x)

        # We only need the attribute row in the keys.
        out = self.gnn(edges, x)[: self.k, :]
        return self.ffn(out)


class DictionaryLookupDataset(object):
    def __init__(self, k: int, max_examples: int = 32_000):
        super().__init__()

        self.k = k
        self.max_examples = max_examples
        # Empty id is for representing query nodes
        self.edges, self.empty_id = self.init_edges()

    def init_edges(self) -> tuple[list, int]:
        targets = range(0, self.k)
        sources = range(self.k, self.k * 2)

        # Complete Bipartite Graph
        next_unused_id = self.k
        all_pairs = itertools.product(sources, targets)
        edges = [list(i) for i in zip(*all_pairs)]

        return edges, next_unused_id

    def create_empty_graph(self) -> mx.array:
        edge_index = mx.array(self.edges, dtype=mx.int64)
        return edge_index

    def get_permutations(self) -> list[list]:
        limit = math.factorial(self.k)

        # Apply a hard limit of the factorial size is too big
        if limit <= self.max_examples:
            generator = itertools.permutations(range(self.k))
            return [perm for _, perm in zip(range(limit), generator)]
        else:
            return [
                np.random.permutation(range(self.k)).tolist()
                for _ in range(self.max_examples)
            ]

    def get_graphs(self) -> list[tuple[mx.array, mx.array]]:
        graphs = []
        for permutation in self.get_permutations():
            edge_index = self.create_empty_graph()
            nodes = mx.array(self.get_nodes_features(permutation), dtype=mx.int64)
            graphs.append((edge_index, nodes))
        return graphs

    def get_nodes_features(self, perm) -> list[tuple[int, int]]:
        # Query Nodes
        nodes = [(key, self.empty_id) for key in range(self.k)]
        # Key Nodes
        for key, val in zip(range(self.k), perm):
            nodes.append((key, val))

        return nodes


def loss_fn(model, X: list[tuple[mx.array, mx.array]]):
    logit_list = []
    value_list = []
    for e, n in X:
        logits = model(e, n)
        # Get the attribute row in keys
        values = n[model.k :, 1]

        logit_list.append(logits)
        value_list.append(values)

    logit_list = mx.concat(logit_list, axis=0)
    value_list = mx.concat(value_list, axis=0)

    loss = nn.losses.cross_entropy(logit_list, value_list).mean()
    return loss


def eval_fn(model, X: tuple[mx.array, mx.array]):
    logits = model(X[0], X[1])
    pred = logits.argmax(axis=1)

    true = X[1][model.k :, 1]

    return pred, true


def worker(queue, args):
    """
    A worker function that trains a model and puts the results in a queue.
    """
    for acc in handle_model(**args):
        queue.put({"name": args["name"], "acc": acc})


def handle_model(
    model,
    train_graphs: list,
    test_graphs: list,
    epochs: int = 1000,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    warmup_steps: int = 40,
    name: str = "",
    position: int = 0,
) -> Iterator[float]:
    mx.eval(model.parameters())

    step_size = int(len(train_graphs) / batch_size)

    warmup = optim.linear_schedule(0, learning_rate, steps=warmup_steps * step_size)
    decay = optim.step_decay(learning_rate, 0.95, step_size * 20)
    lr_schedule = optim.join_schedules([warmup, decay], [warmup_steps * step_size])

    optimizer = optim.Adam(learning_rate=lr_schedule)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    pbar = tqdm(range(epochs), desc=name, position=position * 2)
    for _ in pbar:
        # Train Step
        losses = []
        for i in tqdm(
            range(0, len(train_graphs), batch_size),
            desc=f"{name} training",
            leave=False,
            position=position * 2 + 1,
        ):
            loss, grads = loss_and_grad_fn(model, train_graphs[i : i + batch_size])
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(loss.item())

        # Eval Step
        preds = []
        trues = []
        for edge_index, nodes in test_graphs:
            pred, true = eval_fn(model, (edge_index, nodes))

            preds.append(pred)
            trues.append(true)

        preds = mx.concat(preds, axis=0)
        trues = mx.concat(trues, axis=0)

        acc = (preds == trues).sum() / preds.shape[0]
        pbar.set_postfix({"acc": f"{acc.item():.3f}", "loss": f"{np.mean(losses):.3f}"})
        yield acc


def run_example():
    np.random.seed(42)
    mx.random.seed(42)

    k = 12
    train_size = 0.8
    epochs = 1000
    batch_size = 256

    dataset = DictionaryLookupDataset(k)

    graphs = dataset.get_graphs()
    random.shuffle(graphs)

    train = graphs[: int(len(graphs) * train_size)]
    test = graphs[int(len(graphs) * train_size) :]

    process_queue = Queue()
    processes = []
    accuracies = {"GATv1": [], "GATv2": []}

    models_args = [
        {
            "name": "GATv1",
            "model": Model(k, dynamic_attention=False),
            "position": 0,
        },
        {
            "name": "GATv2",
            "model": Model(k, dynamic_attention=True),
            "position": 1,
        },
    ]

    for args in models_args:
        args.update(
            {
                "train_graphs": train,
                "test_graphs": test,
                "epochs": epochs,
                "batch_size": batch_size,
            }
        )
        p = Process(target=worker, args=(process_queue, args))
        p.start()
        processes.append(p)

    num_epochs = 0
    num_models = len(models_args)
    with tqdm(total=0, position=num_models * 2, bar_format="{desc}") as pbar:
        while any(p.is_alive() for p in processes):
            try:
                # block for 1 second
                res = process_queue.get(timeout=1)
                model_name = res["name"]
                acc = res["acc"]
                accuracies[model_name].append(acc)

                if (
                    len(accuracies["GATv1"]) > num_epochs
                    and len(accuracies["GATv2"]) > num_epochs
                ):
                    num_epochs += 1
                    pbar.set_description_str(
                        "Epoch {}: GATv1: {:0.2%} GATv2: {:0.2%}".format(
                            num_epochs, accuracies["GATv1"][-1], accuracies["GATv2"][-1]
                        )
                    )
                    # Early stopping if no improvement
                    if len(accuracies["GATv2"]) >= 10 and np.all(
                        np.diff(np.array(accuracies["GATv2"][-10:])) <= 0
                    ):
                        pbar.set_description_str(
                            "Stopping due to no accuracy improvements"
                        )
                        for p in processes:
                            p.terminate()
                        break

                if math.isclose(acc, 1.0):
                    pbar.set_description_str(
                        f"{model_name} reached maximum accuracy in {num_epochs} epochs"
                    )
                    for p in processes:
                        p.terminate()
                    break

            except Exception:
                # When queue is empty after timeout
                pass

    for p in processes:
        p.join()


if __name__ == "__main__":
    run_example()
