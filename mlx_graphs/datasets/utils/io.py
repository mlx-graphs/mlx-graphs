from typing import Optional

import mlx.core as mx


def is_floating_point(dtype: mx.Dtype) -> bool:
    if dtype in [mx.float16, mx.float32, mx.bfloat16]:
        return True
    return False


def parse_txt_array(
    src: list[str],
    sep: Optional[str] = None,
    start: int = 0,
    end: Optional[int] = None,
    dtype: Optional[mx.Dtype] = None,
    device=None,
) -> mx.array:
    to_number = float if is_floating_point(dtype) else int

    parsed_src = [[to_number(x) for x in line.split(sep)[start:end]] for line in src]
    parsed_array = mx.array(parsed_src, dtype=dtype).squeeze()
    return parsed_array


def read_txt_array(
    path: str,
    sep: Optional[str] = None,
    start: int = 0,
    end: Optional[int] = None,
    dtype: Optional[mx.Dtype] = None,
    device=None,
) -> mx.array:
    with open(path, "r") as f:
        src = f.read().split("\n")[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)
