import pytest
import mlx.core as mx
import sys
print(sys.path.append('/Users/djamel/Repositories/personal/Thesis/gnn/mlx-graphs/'))
from mlx_graphs.data.batch import Batch
from mlx_graphs.data.data import GraphData

def test_batching():
    node_features1 = mx.array([[1,1,1,1],
            [1,1,1,1],
            [1,0,1,1],
            [0,0,1,1]])
    edge_index1 = mx.array([[0,1,1,2,3],
                            [1,0,2,3,1]])
    node_features2 = mx.array([[1,1,0,1],
                [1,0,1,1],
                [1,0,0,1]])
    edge_index2 = mx.array([[0,1,2,3,1],
                            [1,0,1,2,2]])
    g1 = GraphData(node_features=node_features1, edge_index=edge_index1)
    g2 = GraphData(node_features=node_features2, edge_index=edge_index2)

    batch = Batch.from_graph_list([g1,g2])

    expected_global_node_features = mx.array([[1,1,1,1],
                                            [1,1,1,1],
                                            [1,0,1,1],
                                            [0,0,1,1],
                                            [1,1,0,1],
                                            [1,0,1,1],
                                            [1,0,0,1]])
                                            
    expected_global_edge_index = mx.array([[0,1,1,2,3,4,5,6,7,5],
                                            [1,0,2,3,1,5,6,5,6,6]])


    assert mx.array_equal(batch.node_features, expected_global_node_features), "error in constructing global node features"
    assert mx.array_equal(batch.edge_index, expected_global_edge_index), "error in constructing global edge index"

