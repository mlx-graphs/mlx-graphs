# Datasets

MLX-graphs provides several built-in datasets for graph machine learning tasks.

!!! note

    Datasets are automatically downloaded to `.mlx_graphs_data/` in the current working directory upon first use. You can specify a custom directory by passing `base_dir` to the dataset constructor.

## Available Datasets

| Dataset                    | Graphs | Nodes   | Edges     | Avg Degree | Features   | Labels      |
|----------------------------|--------|---------|-----------|------------|------------|-------------|
| Karate Club                | 1      | 34      | 156       | 9.18       | node       | node        |
| Cora                       | 1      | 2,708   | 10,556    | 7.80       | node       | node        |
| CiteSeer                   | 1      | 3,312   | 9,104     | 5.50       | node       | node        |
| PubMed                     | 1      | 19,717  | 88,651    | 8.99       | node       | node        |
| Amazon Computers           | 1      | 13,752  | 491,722   | 71.51      | node       | node        |
| Amazon Photo               | 1      | 7,650   | 238,162   | 62.26      | node       | node        |
| Coauthor CS                | 1      | 18,333  | 163,788   | 17.87      | node       | node        |
| Coauthor Physics           | 1      | 34,493  | 495,924   | 28.76      | node       | node        |
| QM7b                       | 7,211  | \~15    | \~130     | \~8.67     | node, edge | graph       |
| Elliptic Bitcoin           | 1      | 203,769 | 234,355   | 2.30       | node       | node        |
| MovieLens 100K             | 1      | 1,682   | 100,000   | 118.90     | node       | edge        |
| Cora Full                  | 1      | 19,793  | 126,842   | 12.82      | node       | node        |
| Cora ML (full)             | 1      | 2,995   | 16,316    | 10.90      | node       | node        |
| CiteSeer (full)            | 1      | 4,230   | 10,674    | 5.05       | node       | node        |
| DBLP (full)                | 1      | 17,716  | 105,734   | 11.94      | node       | node        |
| PubMed (full)              | 1      | 19,717  | 88,648    | 8.99       | node       | node        |
| Roman-empire               | 1      | 22,662  | 65,854    | 5.81       | node       | node        |
| Amazon-ratings             | 1      | 24,492  | 186,100   | 15.20      | node       | node        |
| Minesweeper                | 1      | 10,000  | 78,804    | 15.76      | node       | node        |
| Tolokers                   | 1      | 11,758  | 1,038,000 | 176.56     | node       | node        |
| Questions                  | 1      | 48,921  | 307,080   | 12.55      | node       | node        |
| Cornell                    | 1      | 183     | 298       | 3.26       | node       | node        |
| Texas                      | 1      | 183     | 325       | 3.55       | node       | node        |
| Wisconsin                  | 1      | 251     | 515       | 4.10       | node       | node        |
| Chameleon                  | 1      | 2,277   | 36,101    | 31.71      | node       | node        |
| Squirrel                   | 1      | 5,201   | 217,073   | 83.47      | node       | node        |
| Actor                      | 1      | 7,600   | 30,019    | 7.90       | node       | node        |
| WikiCS                     | 1      | 11,701  | 431,726   | 73.79      | node       | node        |
| FB15k-237 (train)          | 1      | 14,541  | 272,115   | 37.43      | none       | edge (type) |
| WordNet18RR                | 1      | 40,943  | 93,003    | 4.54       | none       | edge (type) |
| JODIE Wikipedia (temporal) | 1      | 9,227   | 157,474   | 34.13      | edge       | edge        |


## Dataset Classes { #dataset-classes }

Each class is documented, with its parameters and methods, in the [API reference](../api/datasets/index.md).

<div class="mg-columns" markdown>

- [`KarateClubDataset`][mlx_graphs.datasets.KarateClubDataset]
- [`PlanetoidDataset`][mlx_graphs.datasets.PlanetoidDataset]
- [`AmazonDataset`][mlx_graphs.datasets.AmazonDataset]
- [`CoauthorDataset`][mlx_graphs.datasets.CoauthorDataset]
- [`QM7bDataset`][mlx_graphs.datasets.QM7bDataset]
- [`TUDataset`][mlx_graphs.datasets.TUDataset]
- [`SuperPixelDataset`][mlx_graphs.datasets.SuperPixelDataset]
- [`OGBDataset`][mlx_graphs.datasets.OGBDataset]
- [`EllipticBitcoinDataset`][mlx_graphs.datasets.EllipticBitcoinDataset]
- [`MovieLens100K`][mlx_graphs.datasets.MovieLens100K]
- [`IMDB`][mlx_graphs.datasets.IMDB]
- [`DBLP`][mlx_graphs.datasets.DBLP]
- [`CitationFull`][mlx_graphs.datasets.CitationFull]
- [`CoraFull`][mlx_graphs.datasets.CoraFull]
- [`HeterophilousGraphDataset`][mlx_graphs.datasets.HeterophilousGraphDataset]
- [`WebKB`][mlx_graphs.datasets.WebKB]
- [`WikipediaNetwork`][mlx_graphs.datasets.WikipediaNetwork]
- [`Actor`][mlx_graphs.datasets.Actor]
- [`WikiCS`][mlx_graphs.datasets.WikiCS]
- [`FB15k_237`][mlx_graphs.datasets.FB15k_237]
- [`WordNet18`][mlx_graphs.datasets.WordNet18]
- [`WordNet18RR`][mlx_graphs.datasets.WordNet18RR]
- [`FakeDataset`][mlx_graphs.datasets.FakeDataset]
- [`FakeHeteroDataset`][mlx_graphs.datasets.FakeHeteroDataset]
- [`JODIEDataset`][mlx_graphs.datasets.JODIEDataset]
- [`LRGBDataset`][mlx_graphs.datasets.LRGBDataset]
- [`GNNBenchmarkDataset`][mlx_graphs.datasets.GNNBenchmarkDataset]
- [`OGBMAG`][mlx_graphs.datasets.OGBMAG]

</div>

## Using Datasets

Loading a dataset is straightforward:

```python
from mlx_graphs.datasets import KarateClubDataset, PlanetoidDataset

# Simple built-in dataset
dataset = KarateClubDataset()
graph = dataset[0]
print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")

# Citation network dataset
cora = PlanetoidDataset(name="cora")
print(cora[0])
```

## Computing Dataset Statistics

To compute statistics for any dataset, use the included utility script:

```bash
# Stats for all available datasets
python examples/compute_dataset_stats.py

# Stats for a specific dataset
python examples/compute_dataset_stats.py --dataset karate_club

# Output as markdown table (for use in docs/issues)
python examples/compute_dataset_stats.py --format markdown

# Output as RST table (for Sphinx docs)
python examples/compute_dataset_stats.py --format rst
```
