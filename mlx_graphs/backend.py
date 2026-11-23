"""What the current platform provides to mlx-graphs.

mlx-graphs runs wherever MLX runs. Two accelerations are optional:

* custom Metal kernels (``mx.fast.metal_kernel``), used on Apple silicon GPUs;
* the ``mlx_cluster`` extension (fused sparse kernels, neighbor sampling and
  random walks), installed with mlx-graphs on macOS.

Every code path that uses one of them falls back to MLX operations (or NumPy
for sampling) when it is not available, so the results are the same on Linux
(MLX CPU or CUDA backend) as on a Mac. Two environment variables force the
fallbacks, e.g. to test them on a Mac:

* ``MLX_GRAPHS_DISABLE_METAL_KERNELS=1`` never launches custom Metal kernels;
* ``MLX_GRAPHS_DISABLE_MLX_CLUSTER=1`` behaves as if ``mlx_cluster`` were not
  installed.

Both are read once, when this module is first imported.
"""

import os

import mlx.core as mx

DISABLE_METAL_KERNELS_ENV = "MLX_GRAPHS_DISABLE_METAL_KERNELS"
DISABLE_MLX_CLUSTER_ENV = "MLX_GRAPHS_DISABLE_MLX_CLUSTER"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


_METAL_KERNELS_DISABLED = _env_flag(DISABLE_METAL_KERNELS_ENV)
_MLX_CLUSTER_DISABLED = _env_flag(DISABLE_MLX_CLUSTER_ENV)


def metal_kernels_available() -> bool:
    """Whether custom Metal kernels can run: Metal is available (Apple silicon)
    and they are not disabled with ``MLX_GRAPHS_DISABLE_METAL_KERNELS``."""
    return not _METAL_KERNELS_DISABLED and mx.metal.is_available()


def use_metal_kernels() -> bool:
    """Whether to launch a custom Metal kernel now: Metal kernels are
    available and the default device is the GPU."""
    return mx.default_device().type == mx.gpu and metal_kernels_available()


def _import_mlx_cluster():
    if _MLX_CLUSTER_DISABLED:
        return None, ImportError(
            f"mlx_cluster is disabled by the {DISABLE_MLX_CLUSTER_ENV} "
            "environment variable"
        )
    try:
        import mlx_cluster
    except ImportError as e:  # not installed, or built for another mlx
        return None, e
    return mlx_cluster, None


mlx_cluster, MLX_CLUSTER_IMPORT_ERROR = _import_mlx_cluster()
#: Whether the optional ``mlx_cluster`` extension is installed and enabled.
HAS_MLX_CLUSTER: bool = mlx_cluster is not None


__all__ = [
    "DISABLE_METAL_KERNELS_ENV",
    "DISABLE_MLX_CLUSTER_ENV",
    "HAS_MLX_CLUSTER",
    "MLX_CLUSTER_IMPORT_ERROR",
    "metal_kernels_available",
    "use_metal_kernels",
]
