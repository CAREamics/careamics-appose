import numpy as np
from appose import NDArray


def numpy_to_shared_memory(arr: np.ndarray) -> NDArray:
    """Convert a NumPy array to a shared memory NDArray.

    Parameters
    ----------
    arr : np.ndarray
        The NumPy array to convert.

    Returns
    -------
    NDArray
        The shared memory NDArray containing the same data as the input NumPy array.
    """
    shared = NDArray(str(arr.dtype), list(arr.shape))
    shared.ndarray()[:] = arr

    return shared
