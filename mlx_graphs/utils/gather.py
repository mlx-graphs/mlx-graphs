import mlx.core as mx


def fast_gather(node_features, indices):
    index_size = len(indices)
    emb_size = node_features.shape[1]
    current_device = mx.default_device()

    # Switch to CPU for small slices
    if index_size <= 50_000 and emb_size <= 128:
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)

    gathered = node_features[indices]

    # Set the original device back
    mx.set_default_device(current_device)

    return gathered
