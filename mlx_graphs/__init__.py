import sys as _sys

try:
    import mlx.core as _mx  # noqa: F401
except ImportError as _e:  # e.g. `pip install mlx` on Linux without a backend
    if _sys.platform.startswith("linux"):
        raise ImportError(
            f"{_e}. On Linux, MLX needs a backend: install mlx-graphs with "
            '`pip install "mlx-graphs[cpu]"` or `pip install "mlx-graphs[cuda]"`.'
        ) from _e
    raise
