# Datasets { #mlx_graphs.datasets }

Base classes and built-in datasets of `mlx_graphs.datasets`. The [datasets guide](../../tutorials/guides/datasets.md) lists their sizes.

## Base classes

<!-- api mlx_graphs.datasets -->
BaseDataset
Dataset
HeteroDataset
<!-- /api -->

## Built-in datasets

<!-- api mlx_graphs.datasets template=datasets -->
KarateClubDataset
PlanetoidDataset
AmazonDataset
CoauthorDataset
QM7bDataset
TUDataset
SuperPixelDataset
OGBDataset
OGBMAG
EllipticBitcoinDataset
MovieLens100K
IMDB
DBLP
CitationFull
CoraFull
HeterophilousGraphDataset
WebKB
WikipediaNetwork
Actor
WikiCS
FB15k_237
WordNet18
WordNet18RR
FakeDataset
FakeHeteroDataset
JODIEDataset
TGBLinkPropPredDataset
TGBNodePropPredDataset
LRGBDataset
GNNBenchmarkDataset
ModelNet
ShapeNet
Reddit
Reddit2
Flickr
Yelp
AmazonProducts
PPI
<!-- /api -->

Negative samples of the TGB link property prediction datasets:

<!-- api mlx_graphs.datasets -->
TGBNegativeEdgeSampler
<!-- /api -->

## Utilities

<!-- api mlx_graphs.datasets.utils.torch_serialization -->
load_torch_file
SparseCOOMatrix
<!-- /api -->

Helpers of `mlx_graphs.datasets.utils` for writing a dataset (see the
[datasets guide](../../tutorials/guides/datasets.md#add-your-own-dataset)):

<!-- api mlx_graphs.datasets.utils.download -->
download
<!-- /api -->

<!-- api mlx_graphs.datasets.utils -->
check_sha1
extract_archive
save_graphs
read_txt_array
parse_txt_array
is_floating_point
<!-- /api -->
