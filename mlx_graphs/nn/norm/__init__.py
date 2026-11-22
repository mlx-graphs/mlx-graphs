from .batch_norm import BatchNormalization  # noqa
from .layer_norm import LayerNormalization  # noqa
from .instance_norm import InstanceNormalization  # noqa
from .batch_norm import HeteroBatchNormalization  # noqa
from .layer_norm import HeteroLayerNormalization  # noqa
from .graph_norm import GraphNormalization  # noqa
from .pair_norm import PairNormalization  # noqa
from .graph_size_norm import GraphSizeNormalization  # noqa
from .mean_subtraction_norm import MeanSubtractionNormalization  # noqa
from .msg_norm import MessageNormalization  # noqa
from .diff_group_norm import DiffGroupNormalization  # noqa

# PyG names of the normalization layers.
BatchNorm = BatchNormalization
LayerNorm = LayerNormalization
InstanceNorm = InstanceNormalization
HeteroBatchNorm = HeteroBatchNormalization
HeteroLayerNorm = HeteroLayerNormalization
GraphNorm = GraphNormalization
PairNorm = PairNormalization
GraphSizeNorm = GraphSizeNormalization
MeanSubtractionNorm = MeanSubtractionNormalization
MessageNorm = MessageNormalization
DiffGroupNorm = DiffGroupNormalization
