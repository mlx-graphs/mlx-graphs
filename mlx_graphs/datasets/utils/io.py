import mlx.core as mx


def is_floating_point(dtype):
    if dtype in [mx.float16, mx.float32, mx.bfloat16]:
        return True
    return False


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    to_number = float if is_floating_point(dtype) else int

    src = [[to_number(x) for x in line.split(sep)[start:end]] for line in src]
    src = mx.array(src, dtype=dtype).squeeze()
    return src


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, "r") as f:
        src = f.read().split("\n")[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)
