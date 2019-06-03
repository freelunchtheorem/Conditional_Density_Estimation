import operator
import numpy as np

def flatten_tensors(tensors):
    if len(tensors) > 0:
        return np.concatenate([np.reshape(x, [-1]) for x in tensors])
    else:
        return np.asarray([])


def unflatten_tensors(flattened, tensor_shapes):
    tensor_sizes = list(map(int, map(np.prod, tensor_shapes)))
    indices = np.cumsum(tensor_sizes)[:-1]
    return [np.reshape(pair[0], pair[1]) for pair in zip(np.split(flattened, indices), tensor_shapes)]