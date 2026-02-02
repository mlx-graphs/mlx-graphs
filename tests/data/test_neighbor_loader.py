import mlx.core as mx

from mlx_graphs.data import GraphData
from mlx_graphs.loaders import NeighborLoader


def _make_simple_graph():
    """
    Graph with 6 nodes (0-5), node 5 is isolated.
    Edges: 0->1, 0->2, 1->2, 1->3, 2->3, 3->4
    """
    edge_index = mx.array([[0, 0, 1, 1, 2, 3], [1, 2, 2, 3, 3, 4]], dtype=mx.int64)
    node_features = mx.ones((6, 3))
    node_labels = mx.array([0, 1, 0, 1, 0, 1])
    train_mask = mx.array([True, True, True, False, False, False])
    data = GraphData(
        edge_index=edge_index,
        node_features=node_features,
        node_labels=node_labels,
    )
    data.train_mask = train_mask
    return data


# ---------------------------------------------------------------------------
# Group 1: Basic correctness
# ---------------------------------------------------------------------------


class TestBasicCorrectness:
    def test_returns_graph_data(self):
        data = _make_simple_graph()
        loader = NeighborLoader(data, num_neighbors=[2], batch_size=2)
        batch = next(iter(loader))
        assert isinstance(batch, GraphData)

    def test_batch_size_attribute(self):
        data = _make_simple_graph()
        loader = NeighborLoader(data, num_neighbors=[2], batch_size=3)
        batch = next(iter(loader))
        assert batch.batch_size == 3

    def test_n_id_mapping(self):
        data = _make_simple_graph()
        input_nodes = mx.array([0, 1], dtype=mx.int64)
        loader = NeighborLoader(
            data, num_neighbors=[2], input_nodes=input_nodes, batch_size=2
        )
        batch = next(iter(loader))
        assert hasattr(batch, "n_id")
        assert batch.n_id.shape[0] >= 2
        # First batch_size entries should be the seed nodes
        seed_ids = set(batch.n_id[: batch.batch_size].tolist())
        assert seed_ids == {0, 1}

    def test_edge_index_locally_indexed(self):
        data = _make_simple_graph()
        loader = NeighborLoader(data, num_neighbors=[2], batch_size=2)
        batch = next(iter(loader))
        num_sampled = batch.n_id.shape[0]
        if batch.edge_index.shape[1] > 0:
            assert mx.max(batch.edge_index).item() < num_sampled
            assert mx.min(batch.edge_index).item() >= 0

    def test_node_features_sliced(self):
        data = _make_simple_graph()
        loader = NeighborLoader(data, num_neighbors=[2], batch_size=2)
        batch = next(iter(loader))
        assert batch.node_features.shape[0] == batch.n_id.shape[0]
        assert batch.node_features.shape[1] == 3

    def test_node_features_values_match(self):
        data = _make_simple_graph()
        # Give each node a unique feature so we can verify mapping
        data.node_features = mx.array([[i, i, i] for i in range(6)], dtype=mx.float32)
        input_nodes = mx.array([0, 1], dtype=mx.int64)
        loader = NeighborLoader(
            data, num_neighbors=[2], input_nodes=input_nodes, batch_size=2
        )
        batch = next(iter(loader))
        for i in range(batch.n_id.shape[0]):
            global_id = batch.n_id[i].item()
            expected = data.node_features[global_id]
            actual = batch.node_features[i]
            assert mx.array_equal(
                actual, expected
            ), f"node_features mismatch at local {i} (global {global_id})"

    def test_node_labels_sliced(self):
        data = _make_simple_graph()
        loader = NeighborLoader(data, num_neighbors=[2], batch_size=2)
        batch = next(iter(loader))
        assert batch.node_labels.shape[0] == batch.n_id.shape[0]

    def test_custom_mask_propagated(self):
        data = _make_simple_graph()
        loader = NeighborLoader(data, num_neighbors=[2], batch_size=3)
        batch = next(iter(loader))
        assert hasattr(batch, "train_mask")
        assert batch.train_mask.shape[0] == batch.n_id.shape[0]
        # Verify values match original
        for i in range(batch.n_id.shape[0]):
            global_id = batch.n_id[i].item()
            assert batch.train_mask[i].item() == data.train_mask[global_id].item()


# ---------------------------------------------------------------------------
# Group 2: Iteration behavior
# ---------------------------------------------------------------------------


class TestIteration:
    def test_full_epoch_covers_all_seeds(self):
        data = _make_simple_graph()
        loader = NeighborLoader(data, num_neighbors=[2], batch_size=2)
        batches = list(loader)
        total_seeds = sum(b.batch_size for b in batches)
        assert total_seeds == data.num_nodes

    def test_input_nodes_subset(self):
        data = _make_simple_graph()
        input_nodes = mx.array([0, 1, 2], dtype=mx.int64)
        loader = NeighborLoader(
            data, num_neighbors=[2], input_nodes=input_nodes, batch_size=2
        )
        batches = list(loader)
        total_seeds = sum(b.batch_size for b in batches)
        assert total_seeds == 3

    def test_last_batch_smaller(self):
        data = _make_simple_graph()
        input_nodes = mx.array([0, 1, 2], dtype=mx.int64)
        loader = NeighborLoader(
            data, num_neighbors=[2], input_nodes=input_nodes, batch_size=2
        )
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0].batch_size == 2
        assert batches[1].batch_size == 1

    def test_len(self):
        data = _make_simple_graph()
        input_nodes = mx.array([0, 1, 2, 3, 4], dtype=mx.int64)
        loader = NeighborLoader(
            data, num_neighbors=[2], input_nodes=input_nodes, batch_size=2
        )
        assert len(loader) == 3  # ceil(5/2)

    def test_multiple_epochs(self):
        data = _make_simple_graph()
        loader = NeighborLoader(data, num_neighbors=[2], batch_size=3)
        epoch1 = list(loader)
        epoch2 = list(loader)
        assert len(epoch1) == len(epoch2)

    def test_shuffle_changes_order(self):
        data = _make_simple_graph()
        loader = NeighborLoader(data, num_neighbors=[1], batch_size=1, shuffle=True)
        # Collect seed node order across multiple epochs
        orders = []
        for _ in range(10):
            seeds = [b.n_id[0].item() for b in loader]
            orders.append(tuple(seeds))
        # With 6 nodes, probability of all 10 being identical = (1/720)^9
        unique_orders = set(orders)
        assert len(unique_orders) > 1, "Shuffle should produce different orderings"

    def test_reiter_resets(self):
        data = _make_simple_graph()
        loader = NeighborLoader(data, num_neighbors=[2], batch_size=6)
        b1 = list(loader)
        assert len(b1) == 1
        b2 = list(loader)
        assert len(b2) == 1


# ---------------------------------------------------------------------------
# Group 3: Multi-hop sampling
# ---------------------------------------------------------------------------


class TestMultiHop:
    def test_two_hop_reaches_farther(self):
        # data = _make_simple_graph()
        # 1-hop from node 0 with sample-all: should reach {0} + incoming neighbors of 0
        # In CSC, "neighbors of 0" = nodes with edges TO 0.
        # In our graph, no edges point to 0.
        # But edges FROM 0 go to 1 and 2. For message passing, we need incoming edges.
        # Let's use an undirected graph for a clearer test.
        edge_index = mx.array(
            [
                [0, 1, 0, 2, 1, 2, 1, 3, 2, 3, 3, 4],
                [1, 0, 2, 0, 2, 1, 3, 1, 3, 2, 4, 3],
            ],
            dtype=mx.int64,
        )
        data_ud = GraphData(
            edge_index=edge_index,
            node_features=mx.ones((5, 2)),
        )
        # 1-hop from node 0, sample all
        loader_1hop = NeighborLoader(
            data_ud,
            num_neighbors=[-1],
            input_nodes=mx.array([0], dtype=mx.int64),
            batch_size=1,
        )
        batch_1 = next(iter(loader_1hop))
        nodes_1hop = set(batch_1.n_id.tolist())

        # 2-hop from node 0, sample all
        loader_2hop = NeighborLoader(
            data_ud,
            num_neighbors=[-1, -1],
            input_nodes=mx.array([0], dtype=mx.int64),
            batch_size=1,
        )
        batch_2 = next(iter(loader_2hop))
        nodes_2hop = set(batch_2.n_id.tolist())

        assert nodes_1hop.issubset(nodes_2hop)
        assert len(nodes_2hop) >= len(nodes_1hop)

    def test_num_neighbors_limits_sample(self):
        # Build a star graph: node 0 connects to 1..10
        sources = list(range(1, 11))
        targets = [0] * 10
        edge_index = mx.array([sources, targets], dtype=mx.int64)
        data = GraphData(
            edge_index=edge_index,
            node_features=mx.ones((11, 2)),
        )
        loader = NeighborLoader(
            data,
            num_neighbors=[3],
            input_nodes=mx.array([0], dtype=mx.int64),
            batch_size=1,
            replace=False,
        )
        batch = next(iter(loader))
        # Should sample at most 3 neighbors + the seed = 4 nodes
        assert batch.n_id.shape[0] <= 4


# ---------------------------------------------------------------------------
# Group 4: Edge feature handling
# ---------------------------------------------------------------------------


class TestEdgeFeatures:
    def test_edge_features_sliced(self):
        edge_index = mx.array([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=mx.int64)
        edge_features = mx.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        data = GraphData(
            edge_index=edge_index,
            node_features=mx.ones((3, 2)),
            edge_features=edge_features,
        )
        loader = NeighborLoader(
            data,
            num_neighbors=[-1],
            input_nodes=mx.array([0], dtype=mx.int64),
            batch_size=1,
        )
        batch = next(iter(loader))
        if batch.edge_index.shape[1] > 0:
            assert batch.edge_features is not None
            assert batch.edge_features.shape[0] == batch.edge_index.shape[1]
            assert batch.edge_features.shape[1] == 1

    def test_no_edge_features(self):
        data = _make_simple_graph()
        assert data.edge_features is None
        loader = NeighborLoader(data, num_neighbors=[2], batch_size=2)
        batch = next(iter(loader))
        assert batch.edge_features is None


# ---------------------------------------------------------------------------
# Group 5: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_isolated_seed_node(self):
        data = _make_simple_graph()
        input_nodes = mx.array([5], dtype=mx.int64)  # node 5 is isolated
        loader = NeighborLoader(
            data, num_neighbors=[2], input_nodes=input_nodes, batch_size=1
        )
        batch = next(iter(loader))
        assert batch.n_id.shape[0] >= 1
        assert batch.batch_size == 1
        assert batch.node_features.shape[0] == batch.n_id.shape[0]

    def test_no_node_features(self):
        edge_index = mx.array([[0, 1], [1, 2]], dtype=mx.int64)
        data = GraphData(edge_index=edge_index)
        loader = NeighborLoader(
            data,
            num_neighbors=[2],
            batch_size=1,
            input_nodes=mx.array([0], dtype=mx.int64),
        )
        batch = next(iter(loader))
        assert batch.node_features is None

    def test_single_seed(self):
        data = _make_simple_graph()
        loader = NeighborLoader(
            data,
            num_neighbors=[2],
            batch_size=1,
            input_nodes=mx.array([1], dtype=mx.int64),
        )
        batch = next(iter(loader))
        assert batch.batch_size == 1
        assert batch.n_id[0].item() == 1

    def test_all_neighbors_sampled(self):
        # -1 means sample ALL neighbors
        sources = list(range(1, 6))
        targets = [0] * 5
        edge_index = mx.array([sources, targets], dtype=mx.int64)
        data = GraphData(
            edge_index=edge_index,
            node_features=mx.ones((6, 2)),
        )
        loader = NeighborLoader(
            data,
            num_neighbors=[-1],
            input_nodes=mx.array([0], dtype=mx.int64),
            batch_size=1,
        )
        batch = next(iter(loader))
        # All 5 neighbors + seed = 6 nodes
        assert batch.n_id.shape[0] == 6

    def test_replace_true(self):
        data = _make_simple_graph()
        loader = NeighborLoader(data, num_neighbors=[2], batch_size=2, replace=True)
        batch = next(iter(loader))
        assert isinstance(batch, GraphData)

    def test_replace_false(self):
        data = _make_simple_graph()
        loader = NeighborLoader(data, num_neighbors=[2], batch_size=2, replace=False)
        batch = next(iter(loader))
        assert isinstance(batch, GraphData)


# ---------------------------------------------------------------------------
# Group 6: Integration / training pattern
# ---------------------------------------------------------------------------


class TestTrainingPattern:
    def test_training_loop_pattern(self):
        data = _make_simple_graph()
        # Give unique features per node
        data.node_features = mx.array(
            [[float(i)] * 3 for i in range(6)], dtype=mx.float32
        )
        train_indices = mx.array([0, 1, 2, 3], dtype=mx.int64)
        loader = NeighborLoader(
            data,
            num_neighbors=[2],
            input_nodes=train_indices,
            batch_size=2,
            shuffle=False,
        )

        for batch in loader:
            # Simulate forward pass
            num_nodes_in_batch = batch.node_features.shape[0]
            fake_logits = mx.ones((num_nodes_in_batch, 2))

            # Only use seed nodes for loss
            seed_logits = fake_logits[: batch.batch_size]
            seed_labels = batch.node_labels[: batch.batch_size]

            assert seed_logits.shape[0] == batch.batch_size
            assert seed_labels.shape[0] == batch.batch_size

            # Verify n_id maps back to original correctly
            for i in range(batch.batch_size):
                global_id = batch.n_id[i].item()
                assert global_id in train_indices.tolist()
