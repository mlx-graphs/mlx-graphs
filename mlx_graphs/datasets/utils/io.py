import glob
import gzip
import os
import re
from typing import Optional

import mlx.core as mx
from tqdm import tqdm


def is_floating_point(dtype: mx.Dtype) -> bool:
    if dtype in [mx.float16, mx.float32, mx.bfloat16]:
        return True
    return False


def parse_txt_array(
    src: list[str],
    sep: str,
    dtype: mx.Dtype,
    start: int = 0,
    end: Optional[int] = None,
) -> mx.array:
    to_number = float if is_floating_point(dtype) else int
    parsed_src = [[to_number(x) for x in line.split(sep)[start:end]] for line in src]
    parsed_array = mx.array(parsed_src, dtype=dtype).squeeze()
    return parsed_array


def read_txt_array(
    path: str,
    dtype: mx.Dtype,
    sep: str = None,
    start: int = 0,
    end: Optional[int] = None,
) -> mx.array:
    with open(path, "r") as f:
        src = f.read().split("\n")[:-1]
    return parse_txt_array(src, sep, dtype, start, end)


def compress_and_remove_files(
    folder_path: str, file_extension: str, use_tqdm: bool = True
):
    """
    Compress all files ending with `file_extension`
    in the specified folder into .gz files and remove the original files.

    Parameters:
        folder_path: The path to the folder containing files to be compressed.
        file_extension: The extension to match to compress and delete.
    """
    file_pattern = os.path.join(folder_path, f"*.{file_extension}")
    files = glob.glob(file_pattern)

    for file in tqdm(files, total=len(files)):
        gz_file = file + ".gz"

        with open(file, "rb") as f_in:
            with gzip.open(gz_file, "wb") as f_out:
                f_out.writelines(f_in)

        os.remove(file)


def get_index_from_filename(x):
    """
    Returns the number within the filename in a path

    Example:
        get_index_from_filename("/users/user01/path03/to/file0.csv")
        >>> 0
    """
    return int(re.findall(r"\d+(?=[^\d]*$)", x)[0])
