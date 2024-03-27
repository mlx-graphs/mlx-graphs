import os
import shutil
from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.lazy_dataset import LazyDataset
from mlx_graphs.datasets.utils import (
    compress_and_remove_files,
    download,
    download_file_from_google_drive,
    extract_archive,
)
from mlx_graphs.datasets.utils.lanl_preprocessing import split
from mlx_graphs.utils.validators import validate_package

# Preprocessed auth csv file fields
LANL_TS = 0
LANL_SRC = 1
LANL_DST = 2
LANL_LABEL = 3
LANL_SUCCESS = 4
LANL_SRC_USER_TYPE = 5
LANL_SRC_USER = 6
LANL_DST_USER = 7

# Num nodes for the overall 58 days
LANL_NUM_NODES = 17685


class LANLDataset(LazyDataset):
    """
    The Los Alamos National Lab (LANL) is a very large dataset
    comprising ~73GB of authentication logs.

    It is made up of 58 consecutive days of data gathered from
    the Los Alamos National Laboratory’s internal
    computer network. This version of the dataset comprises 49,341,086 auth events
    from a typical APT campaign across 17,685 Windows machines, including
    705 malicious events. Graphs can be constructed from authentication events,
    where an edge represents an authentication action and a node represents
    a machine within the network.
    Each edge is associated with either a benign or malicious label, making this
    dataset suitable for edge prediction/detection tasks, as well as node-based
    tasks, where the machine at the origin of malicious activity can be identified.

    The version of the LANL dataset proposed within mlx-graphs is already
    preprocessed to only include data required to build the graphs along with
    2 default edge features extracted from the raw dataset. Consequently,
    this class will by default download a 340MB archive instead of the original
    files that represent more than 7GB compressed.

    The 2 provided edge features are:
    (i) success/failure: 1 if the authentication succeeded, 0 if it failed
    (ii) logon type: identifies the type of the source user that initiated the
    authentication (user -> 1, computer -> 2, anonymous -> 3)

    The LANL dataset doesn't come with many features. However, one can supplement
    these features with hand-crafted ones such as centrality-based features.

    Args:
        process_original_files: Whether to re-compute the original features from
            scratch. By default, this dataset uses a preprocessed version of the
            raw LANL dataset, which is much smaller and preserves all the required
            data to build the graphs. In this version, each minute of data is
            represented by one "csv.gz" file.
            If one wants to re-compute this preprocessed
            version of the dataset, just set ``process_original_files`` to ``True``
            and the original >7GB `auth.txt.gz` file will be downloaded and
            preprocessed locally. Warning: downloading this file is extremly slow.
        use_gzip: Whether to read the raw files as "csv.gz" or "csv". Default to
            ``True``.
        num_nodes: The number of nodes in the dataset. This number is required to
            compute the one-hot encoding of nodes as node features. Default to
            ``17685``.
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import LANLDataset
        from mlx_graphs.loaders import LANLDataLoader


        dataset = LANLDataset()  # Each of the 83519 graphs contains 1min of data
        >>> LANL(num_graphs=83519)

        dataset[0]  # Computes and yields a graph for the first minute of data
        >>> GraphData(
            edge_index(shape=(2, 224), int64)
            edge_features(shape=(224, 2), float32)
            edge_labels(shape=(224,), int64)
            edge_timestamps(shape=(224,), int64))

        dataset[:60 * 24]  # Computes a graph for the first day
        >>> GraphData(
            edge_index(shape=(2, 762125), int64)
            edge_features(shape=(762125, 2), float32)
            edge_labels(shape=(762125,), int64)
            edge_timestamps(shape=(762125,), int64))

        dataset[[0, 1, 2]]  # Computes a graph for the first 3 minutes \
(with array indexing)
        >>> dataset[[0, 1, 2]]
            GraphData(
                edge_index(shape=(2, 768), int64)
                edge_features(shape=(768, 2), float32)
                edge_labels(shape=(768,), int64)
                edge_timestamps(shape=(768,), int64))

        #  LANLDataLoader can be used to easily iterate over LANL, and is useful to
        #  reduce the size of the graphs with graph compression
        loader = LANLDataLoader(
            dataset, split="train", remove_self_loops=False, compress_edges=True, \
batch_size=60,
        )
        next(loader)
        >>> GraphData(
            edge_index(shape=(2, 10064), int64)
            node_features(shape=(17685, 17685), float32)
            edge_features(shape=(10064, 6), float32)
            edge_labels(shape=(10064,), int64))
    """

    _url = "https://csr.lanl.gov/data-fence/1711123950/6SZMOJi6hdg8xDqDbFiB9QrZqAA=/cyber1/"
    _preprocessed_drive_id = "11rSBKagmOzmC-Xru4mUyTfep6P7FGmcO"

    def __init__(
        self,
        process_original_files: bool = False,
        use_gzip: bool = True,
        num_nodes: int = LANL_NUM_NODES,
        base_dir: Optional[str] = None,
        **kwargs,
    ):
        self.auth_file = "auth.txt.gz"
        self.redteam_file = "redteam.txt.gz"
        self.tmp_archive_file = "LANL.zip"

        self._use_gzip = use_gzip
        self.process_original_files = process_original_files

        extension = "csv.gz" if use_gzip else "csv"
        super().__init__(
            name="LANL",
            raw_file_extension=extension,
            num_nodes=num_nodes,
            base_dir=base_dir,
            **kwargs,
        )

    @property
    def original_path(self) -> str:
        """
        We call original files, the files that are downloaded from the official
        LANL website. We only use 2 of these files: "auth.txt.gz" and "redteam.txt.gz".

        Returns:
            The path to the folder that contains original files.
        """
        root = "/".join(self.raw_path.split("/")[:-1])
        return os.path.join(root, "original")

    def download(self):
        """
        By default, downloads the preprocessed form of the LANL dataset.
        This version comprises a "LANL.zip" file of size ~340MB, containing
        a csv.gz snapshot file for each minute of data.

        If ``process_original_files=True``, the original "auth.txt.gz" and
        "redteam.txt.gz" files are downloaded and preprocessed to re generate
        the files in "LANL.zip".
        """
        if not self.process_original_files:
            download_file_from_google_drive(
                id=self._preprocessed_drive_id,
                path=os.path.join(self.raw_path, self.tmp_archive_file),
            )
            extract_archive(
                os.path.join(self.raw_path, self.tmp_archive_file), self.raw_path
            )

            # "LANL" is the name of the folder after decompression
            extracted_folder = os.path.join(self.raw_path, "LANL")
            files = os.listdir(extracted_folder)

            for file in files:
                source_file = os.path.join(extracted_folder, file)
                destination_file = os.path.join(self.raw_path, file)

                shutil.move(source_file, destination_file)

            shutil.rmtree(extracted_folder, ignore_errors=True)

        else:
            # NOTE: This downloads the 7.6GB auth.txt.gz file from LANL.
            # It is extremly long to dl due to LANL's slow servers (up to 2hr)
            if not os.path.exists(os.path.join(self.original_path, self.auth_file)):
                download(
                    url=os.path.join(self._url, self.auth_file),
                    path=os.path.join(self.original_path, self.auth_file),
                )

            if not os.path.exists(os.path.join(self.original_path, self.redteam_file)):
                download(
                    url=os.path.join(self._url, self.redteam_file),
                    path=os.path.join(self.original_path, self.redteam_file),
                )

    def process(self):
        """
        By default, there is no processing to perform as all
        computations are performed lazily when requested with indexing.

        If ``process_original_files`` is True, then the parsing of the
        large `auth.txt.gz` file occurs, and csv files are written to
        ``self.raw_path``.
        """
        if self.process_original_files:
            print("Processing all authentication logs...")
            split(
                auth_file=os.path.join(self.original_path, self.auth_file),
                redteam_file=os.path.join(self.original_path, self.redteam_file),
                dst_path=self.raw_path,
                graph_path_at_index_fn=self.graph_path_at_index,
            )

            print("Compressing all files to .gz...")
            compress_and_remove_files(
                folder_path=self.raw_path,
                file_extension="csv",
            )

    @validate_package("pandas")
    def load_lazily(self, file_path: str, as_pandas_df: bool = False) -> GraphData:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Warning: file not found: {file_path}")

        import pandas as pd

        # If gzip compression, we read the csv while decompressing it.
        compression = "gzip" if self._use_gzip else "infer"

        # Reads the csv and assign an int to each column index.
        df = pd.read_csv(
            file_path,
            index_col=False,
            header=None,
            names=range(LANL_TS, LANL_DST_USER + 1),
            dtype={
                LANL_TS: str,
                LANL_SRC: int,
                LANL_DST: int,
                LANL_LABEL: int,
                LANL_SUCCESS: int,
                LANL_SRC_USER_TYPE: int,
                LANL_SRC_USER: str,
                LANL_DST_USER: str,
            },
            compression=compression,
        )

        # Additional preprocessing, normalisation, ...
        df = self._preprocess_one_snapshot(df)

        # Output schema.
        df_adj = df[
            [LANL_SRC, LANL_DST, LANL_TS, LANL_LABEL, LANL_SRC_USER, LANL_DST_USER]
        ]
        edge_feats = df[[LANL_SUCCESS, LANL_SRC_USER_TYPE]].to_numpy()

        # If called from a loader, we usually return the df for further preprocessing
        if as_pandas_df:
            return (
                df_adj,
                edge_feats,
            )

        # By default, the dataset returns a GraphData with node features
        graph = self.to_graphdata(
            df_adj,
            edge_feats,
        )
        graph = self.add_one_hot_node_features(graph)
        return graph

    def __concat__(self, items: list[GraphData]) -> GraphData:
        return GraphData(
            edge_index=mx.concatenate([g.edge_index for g in items], axis=1),
            edge_labels=mx.concatenate([g.edge_labels for g in items], axis=0),
            edge_features=mx.concatenate([g.edge_features for g in items], axis=0),
            edge_timestamps=mx.concatenate([g.edge_timestamps for g in items], axis=0),
            node_features=items[0].node_features,
        )

    def to_graphdata(self, df_adj: "DataFrame", edge_feats: np.ndarray) -> GraphData:  # noqa: F821
        return GraphData(
            edge_index=mx.array(
                np.stack(
                    [df_adj[LANL_SRC].to_numpy(), df_adj[LANL_DST].to_numpy()], axis=0
                )
            ),
            edge_labels=mx.array(df_adj[LANL_LABEL].to_numpy()),
            edge_features=mx.array(edge_feats, dtype=mx.float32),
            edge_timestamps=mx.array(df_adj[LANL_TS].to_numpy()),
        )

    @validate_package("pandas")
    def _preprocess_one_snapshot(self, df: "DataFrame") -> "DataFrame":  # noqa: F821
        import pandas as pd

        # Normalization raises an unwanted warning.
        pd.set_option("mode.chained_assignment", None)

        df[LANL_TS] = pd.to_numeric(df[LANL_TS])
        df.fillna(0, inplace=True)
        return df
