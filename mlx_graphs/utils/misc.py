import hashlib


def hash_string(*strings) -> int:
    """
    By default in Python, hash("a string") returns a different hash value
    for each different Python session. This is used as a mechanism to prevent
    some cryptographic attacks. However, in some cases, we may want to get
    a consistent hash value that remains always the same.

    Args:
        *strings: One or more strings to hash

    Returns:
        The hash of this string as an integer
    """
    hash_object = hashlib.sha256()
    for element in strings:
        hash_object.update(str(element).encode())
    # Convert the hash digest to an integer
    hash_int = int.from_bytes(hash_object.digest(), "big")
    return hash_int
