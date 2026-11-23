"""Transforms of point clouds and meshes (node positions ``pos`` and mesh
triangles ``face``), following PyG's."""

import math
from typing import Optional, Sequence, Tuple, Union

import mlx.core as mx

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform
from mlx_graphs.transforms.spatial import _positions
from mlx_graphs.utils.transformations import to_undirected


def _edges(data: GraphData) -> mx.array:
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        raise ValueError("The graph has no `edge_index`")
    return edge_index


def _set_pseudo(data: GraphData, pseudo: mx.array, cat: bool) -> GraphData:
    """Stores the new edge features, after the existing ones if ``cat``."""
    features = data.edge_features
    if features is not None and cat:
        features = features.reshape(-1, 1) if features.ndim == 1 else features
        data.edge_features = mx.concatenate(
            [features, pseudo.astype(features.dtype)], axis=-1
        )
    else:
        data.edge_features = pseudo
    return data


class Center(BaseTransform):
    """Centers the node positions ``pos`` around the origin.

    Example:

    .. code-block:: python

        from mlx_graphs.transforms import Center

        graph = Center()(graph)  # graph.pos.mean(0) == 0
    """

    def process(self, data: GraphData) -> GraphData:
        pos = _positions(data)
        data.pos = pos - pos.mean(axis=-2, keepdims=True)
        return data


class NormalizeScale(BaseTransform):
    """Centers and scales the node positions ``pos`` to the interval
    :math:`(-1, 1)`.

    Example:

    .. code-block:: python

        from mlx_graphs.transforms import NormalizeScale

        graph = NormalizeScale()(graph)  # graph.pos.abs().max() < 1
    """

    def __init__(self):
        self.center = Center()

    def process(self, data: GraphData) -> GraphData:
        data = self.center.process(data)
        pos = data.pos
        scale = (1.0 / mx.abs(pos).max()) * 0.999999
        data.pos = pos * scale
        return data


class LinearTransformation(BaseTransform):
    """Transforms the node positions ``pos`` with a square transformation
    matrix: ``pos @ matrix.T``.

    Args:
        matrix: The ``[D, D]`` transformation matrix.
    """

    def __init__(self, matrix: Union[mx.array, Sequence[Sequence[float]]]):
        matrix = mx.array(matrix) if not isinstance(matrix, mx.array) else matrix
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"The transformation matrix must be square (got {matrix.shape})"
            )
        self.matrix = matrix.T

    def process(self, data: GraphData) -> GraphData:
        pos = _positions(data)
        pos = pos.reshape(-1, 1) if pos.ndim == 1 else pos
        if pos.shape[-1] != self.matrix.shape[-2]:
            raise ValueError(
                "Node position matrix and transformation matrix have "
                "incompatible shape"
            )
        # A broadcast product instead of a matmul: exact in float32 (MLX's GPU
        # matmul is less accurate) and as fast for 2D/3D positions.
        matrix = self.matrix.astype(pos.dtype)
        data.pos = (pos[:, :, None] * matrix[None]).sum(axis=1)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n{self.matrix.T}\n)"


class RandomJitter(BaseTransform):
    """Translates each node position ``pos`` by a random offset, sampled
    uniformly from :math:`[-t, t]` in each dimension (with ``mx.random``).

    Args:
        translate: The maximum translation :math:`t`, for all dimensions or
            for each of them.

    Example:

    .. code-block:: python

        from mlx_graphs.transforms import RandomJitter

        graph = RandomJitter(0.01)(graph)
    """

    def __init__(self, translate: Union[float, Sequence[float]]):
        self.translate = translate

    def process(self, data: GraphData) -> GraphData:
        pos = _positions(data)
        num_nodes, dim = pos.shape
        if isinstance(self.translate, (int, float)):
            translate = [float(self.translate)] * dim
        else:
            translate = [float(t) for t in self.translate]
            if len(translate) != dim:
                raise ValueError(f"translate must have {dim} entries")
        bound = mx.array([abs(t) for t in translate], dtype=pos.dtype)
        jitter = mx.random.uniform(-1.0, 1.0, (num_nodes, dim)) * bound
        data.pos = pos + jitter.astype(pos.dtype)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.translate})"


class RandomScale(BaseTransform):
    r"""Scales the node positions ``pos`` by a random factor sampled uniformly
    from ``scales`` (with ``mx.random``).

    Args:
        scales: The ``(min, max)`` scaling factors.

    Example:

    .. code-block:: python

        from mlx_graphs.transforms import RandomScale

        graph = RandomScale((0.9, 1.1))(graph)
    """

    def __init__(self, scales: Tuple[float, float]):
        if not isinstance(scales, (tuple, list)) or len(scales) != 2:
            raise ValueError("scales must be a (min, max) tuple")
        self.scales = tuple(scales)

    def process(self, data: GraphData) -> GraphData:
        pos = _positions(data)
        scale = mx.random.uniform(self.scales[0], self.scales[1], ()).item()
        data.pos = pos * scale
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.scales})"


class RandomRotate(BaseTransform):
    r"""Rotates the node positions ``pos`` by a random angle (in degrees)
    around an axis, sampled uniformly from ``degrees`` (with ``mx.random``).

    Args:
        degrees: The ``(min, max)`` rotation angles, or a value ``d`` for
            ``(-d, d)``.
        axis: The rotation axis (for 3D positions). Default: ``0``

    Example:

    .. code-block:: python

        from mlx_graphs.transforms import RandomRotate

        graph = RandomRotate(15, axis=2)(graph)
    """

    def __init__(self, degrees: Union[Tuple[float, float], float], axis: int = 0):
        if isinstance(degrees, (int, float)):
            degrees = (-abs(degrees), abs(degrees))
        if not isinstance(degrees, (tuple, list)) or len(degrees) != 2:
            raise ValueError("degrees must be a number or a (min, max) tuple")
        self.degrees = tuple(degrees)
        self.axis = axis

    def process(self, data: GraphData) -> GraphData:
        pos = _positions(data)
        angle = mx.random.uniform(self.degrees[0], self.degrees[1], ()).item()
        angle = math.pi * angle / 180.0
        sin, cos = math.sin(angle), math.cos(angle)
        if pos.shape[-1] == 2:
            matrix = [[cos, sin], [-sin, cos]]
        elif self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformation(mx.array(matrix)).process(data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.degrees}, axis={self.axis})"


class SamplePoints(BaseTransform):
    r"""Uniformly samples a fixed number of points on the faces of a mesh
    (``pos`` and ``face``), with probabilities proportional to the face areas
    (with ``mx.random``).

    Args:
        num: The number of points to sample.
        remove_faces: If ``True``, removes the ``face`` attribute. Default:
            ``True``
        include_normals: If ``True``, stores the normal of the face of each
            sampled point in ``normal``. Default: ``False``

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import ModelNet
        from mlx_graphs.transforms import SamplePoints

        dataset = ModelNet("10", transform=SamplePoints(1024))
    """

    def __init__(
        self, num: int, remove_faces: bool = True, include_normals: bool = False
    ):
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals

    def process(self, data: GraphData) -> GraphData:
        pos = _positions(data)
        face = getattr(data, "face", None)
        if face is None:
            raise ValueError("The graph has no `face` attribute")
        if pos.shape[1] != 3 or face.shape[0] != 3:
            raise ValueError("SamplePoints expects 3D positions and triangle faces")

        pos_max = mx.abs(pos).max()
        pos = pos / pos_max
        v0, v1, v2 = pos[face[0]], pos[face[1]], pos[face[2]]
        area = mx.linalg.norm(mx.linalg.cross(v1 - v0, v2 - v0, axis=1), axis=1) / 2
        sample = mx.random.categorical(mx.log(area / area.sum()), num_samples=self.num)
        face = face[:, sample]

        frac = mx.random.uniform(shape=(self.num, 2))
        frac = mx.where(frac.sum(axis=-1, keepdims=True) > 1, 1 - frac, frac)
        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]
        if self.include_normals:
            normal = mx.linalg.cross(vec1, vec2, axis=1)
            norm = mx.linalg.norm(normal, axis=1, keepdims=True)
            data.normal = normal / mx.maximum(norm, 1e-12)

        sampled = pos[face[0]] + frac[:, :1] * vec1 + frac[:, 1:] * vec2
        data.pos = sampled * pos_max
        if self.remove_faces:
            delattr(data, "face")
        if getattr(data, "_num_nodes", None) is not None:
            data.num_nodes = self.num
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.num})"


class FixedPoints(BaseTransform):
    r"""Samples a fixed number of points (nodes) from a point cloud, with all
    their node-level attributes (with ``mx.random``). Edges are left
    unchanged, as in PyG.

    Args:
        num: The number of points to sample.
        replace: If ``True``, samples with replacement. Default: ``True``
        allow_duplicates: Without replacement, whether to repeat points when
            ``num`` is larger than the number of points. Default: ``False``

    Example:

    .. code-block:: python

        from mlx_graphs.transforms import FixedPoints

        graph = FixedPoints(1024, replace=False)(graph)
    """

    def __init__(self, num: int, replace: bool = True, allow_duplicates: bool = False):
        self.num = num
        self.replace = replace
        self.allow_duplicates = allow_duplicates

    def process(self, data: GraphData) -> GraphData:
        num_nodes = data.num_nodes
        if self.replace:
            choice = mx.random.randint(0, num_nodes, (self.num,))
        elif not self.allow_duplicates:
            choice = mx.random.permutation(num_nodes)[: self.num]
        else:
            repeats = math.ceil(self.num / num_nodes)
            choice = mx.concatenate(
                [mx.random.permutation(num_nodes) for _ in range(repeats)]
            )[: self.num]
        choice = choice.astype(mx.int32)
        for key, value in list(vars(data).items()):
            if key.startswith(("_", "graph_")) or "edge" in key or key == "face":
                continue
            if (
                isinstance(value, mx.array)
                and value.ndim > 0
                and value.shape[0] == num_nodes
                and num_nodes != 1
            ):
                setattr(data, key, value[choice])
        data.num_nodes = choice.shape[0]
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.num}, replace={self.replace})"


class FaceToEdge(BaseTransform):
    """Converts the mesh faces ``face`` (triangles ``[3, num_faces]`` or
    tetrahedra ``[4, num_faces]``) to undirected, coalesced edges.

    Args:
        remove_faces: If ``True``, removes the ``face`` attribute. Default:
            ``True``

    Example:

    .. code-block:: python

        from mlx_graphs.transforms import FaceToEdge

        graph = FaceToEdge()(mesh)
    """

    def __init__(self, remove_faces: bool = True):
        self.remove_faces = remove_faces

    def process(self, data: GraphData) -> GraphData:
        face = getattr(data, "face", None)
        if face is None:
            return data
        if face.shape[0] == 3:
            edge_index = mx.concatenate([face[:2], face[1:], face[::2]], axis=1)
        elif face.shape[0] == 4:
            edge_index = mx.concatenate(
                [face[:2], face[1:3], face[2:4], face[::2], face[1::2], face[::3]],
                axis=1,
            )
        else:
            raise ValueError(
                "Expected 'face' of shape [3, num_faces] or [4, num_faces] "
                f"(got {list(face.shape)})"
            )
        num_nodes = data.num_nodes
        data.edge_index = to_undirected(
            edge_index, coalesce=True, num_nodes=num_nodes
        ).astype(mx.int32)
        if self.remove_faces:
            delattr(data, "face")
        return data


class Cartesian(BaseTransform):
    r"""Saves the relative Cartesian coordinates of linked nodes in its edge
    features: :math:`\mathbf{p}_{\text{src}} - \mathbf{p}_{\text{dst}}` for
    each edge, as in PyG.

    Args:
        norm: If ``True``, normalizes the coordinates to ``interval``.
            Default: ``True``
        max_value: If set and ``norm`` is ``True``, normalizes with this value
            instead of the largest absolute coordinate. Default: ``None``
        cat: If ``False``, replaces the existing edge features instead of
            concatenating. Default: ``True``
        interval: The interval of the normalized coordinates. Default:
            ``(0.0, 1.0)``

    Example:

    .. code-block:: python

        from mlx_graphs.transforms import Cartesian

        graph = Cartesian()(graph)  # pseudo-coordinates for SplineConv
    """

    def __init__(
        self,
        norm: bool = True,
        max_value: Optional[float] = None,
        cat: bool = True,
        interval: Tuple[float, float] = (0.0, 1.0),
    ):
        self.norm = norm
        self.max = max_value
        self.cat = cat
        self.interval = interval

    def process(self, data: GraphData) -> GraphData:
        row, col = _edges(data)
        pos = _positions(data)
        cart = pos[row] - pos[col]
        cart = cart.reshape(-1, 1) if cart.ndim == 1 else cart
        if self.norm and cart.size > 0:
            max_val = mx.abs(cart).max().item() if self.max is None else self.max
            length = self.interval[1] - self.interval[0]
            center = (self.interval[0] + self.interval[1]) / 2
            cart = length * cart / (2 * max_val) + center
        return _set_pseudo(data, cart, self.cat)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(norm={self.norm}, max_value={self.max})"


class Distance(BaseTransform):
    r"""Saves the Euclidean distance of linked nodes in its edge features.

    Args:
        norm: If ``True``, normalizes the distances to ``interval``. Default:
            ``True``
        max_value: If set and ``norm`` is ``True``, normalizes with this value
            instead of the largest distance. Default: ``None``
        cat: If ``False``, replaces the existing edge features instead of
            concatenating. Default: ``True``
        interval: The interval of the normalized distances. Default:
            ``(0.0, 1.0)``

    Example:

    .. code-block:: python

        from mlx_graphs.transforms import Distance

        graph = Distance(norm=False)(graph)
    """

    def __init__(
        self,
        norm: bool = True,
        max_value: Optional[float] = None,
        cat: bool = True,
        interval: Tuple[float, float] = (0.0, 1.0),
    ):
        self.norm = norm
        self.max = max_value
        self.cat = cat
        self.interval = interval

    def process(self, data: GraphData) -> GraphData:
        row, col = _edges(data)
        pos = _positions(data)
        dist = mx.linalg.norm(pos[col] - pos[row], axis=-1).reshape(-1, 1)
        if self.norm and dist.size > 0:
            max_val = dist.max().item() if self.max is None else self.max
            length = self.interval[1] - self.interval[0]
            dist = length * (dist / max_val) + self.interval[0]
        return _set_pseudo(data, dist, self.cat)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(norm={self.norm}, max_value={self.max})"


class Polar(BaseTransform):
    r"""Saves the polar coordinates (distance and angle in :math:`[0, 2\pi)`)
    of linked nodes of 2D graphs in its edge features.

    Args:
        norm: If ``True``, normalizes the distances to :math:`[0, 1]` and the
            angles by :math:`2\pi`. Default: ``True``
        max_value: If set and ``norm`` is ``True``, normalizes the distances
            with this value instead of the largest one. Default: ``None``
        cat: If ``False``, replaces the existing edge features instead of
            concatenating. Default: ``True``

    Example:

    .. code-block:: python

        from mlx_graphs.transforms import Polar

        graph = Polar()(graph)
    """

    def __init__(
        self, norm: bool = True, max_value: Optional[float] = None, cat: bool = True
    ):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def process(self, data: GraphData) -> GraphData:
        row, col = _edges(data)
        pos = _positions(data)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("Polar expects 2D positions")
        cart = pos[col] - pos[row]
        rho = mx.linalg.norm(cart, axis=-1).reshape(-1, 1)
        theta = mx.arctan2(cart[..., 1], cart[..., 0]).reshape(-1, 1)
        theta = theta + (theta < 0).astype(theta.dtype) * (2 * math.pi)
        if self.norm:
            rho = rho / (rho.max() if self.max is None else self.max)
            theta = theta / (2 * math.pi)
        polar = mx.concatenate([rho, theta], axis=-1)
        return _set_pseudo(data, polar, self.cat)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(norm={self.norm}, max_value={self.max})"


class PointPairFeatures(BaseTransform):
    r"""Saves the rotation-invariant point pair features of linked nodes in
    their edge features (see
    :func:`~mlx_graphs.nn.conv.ppf_conv.point_pair_features`). The normals are
    read from ``norm``, as in PyG, or else from ``normal``.

    Args:
        cat: If ``False``, replaces the existing edge features instead of
            concatenating. Default: ``True``

    Example:

    .. code-block:: python

        from mlx_graphs.transforms import PointPairFeatures

        graph = PointPairFeatures()(graph)
    """

    def __init__(self, cat: bool = True):
        self.cat = cat

    def process(self, data: GraphData) -> GraphData:
        from mlx_graphs.nn.conv.ppf_conv import point_pair_features

        row, col = _edges(data)
        pos = _positions(data)
        normal = getattr(data, "norm", None)
        if normal is None:
            normal = getattr(data, "normal", None)
        if normal is None:
            raise ValueError("The graph has no `norm` (or `normal`) attribute")
        if pos.shape[-1] != 3 or pos.shape != normal.shape:
            raise ValueError("PointPairFeatures expects 3D positions and normals")
        ppf = point_pair_features(pos[row], pos[col], normal[row], normal[col])
        return _set_pseudo(data, ppf, self.cat)
