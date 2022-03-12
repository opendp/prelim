from typing import List
import numpy as np
from opendp.meas import make_base_geometric
from opendp.mod import enable_features

enable_features("contrib")


def histogramdd_indexes(x: np.ndarray, category_lengths: List[int]) -> np.ndarray:
    """Compute counts of each combination of categories in d dimensions.
    Discrete version of np.histogramdd.

    :param x: data of shape [n, len(`category_lengths`)] of non-negative category indexes
    :param category_lengths: the number of unique categories per column
    """

    assert x.shape[1] == len(category_lengths)
    assert x.ndim == 2
    if not len(category_lengths):
        return np.array(x.shape[0])

    # consider each row as a multidimensional index into an ndarray
    # determine what those indexes would be should the ndarray be flattened
    # the flat indices uniquely identify each cell
    flat_indices = np.ravel_multi_index(x.T, category_lengths)

    # count the number of instances of each index
    hist = np.bincount(flat_indices, minlength=np.prod(category_lengths))

    # map counts back to d-dimensional output
    return hist.reshape(category_lengths)


def release_histogramdd_indexes(
    x: np.ndarray, category_lengths: List[int], scale
) -> np.ndarray:
    """Release a d-dimensional histogram with noise `scale`.
    The ith column of x must range from 0 to category_lengths[i].

    :param x: data of shape [n, len(`category_lengths`)] of non-negative category indexes
    :param category_lengths: the number of unique categories per column
    """
    hist = histogramdd_indexes(x, category_lengths)
    meas = make_base_geometric(scale, D="VectorDomain<AllDomain<i64>>")
    return np.reshape(meas(hist.flatten()), hist.shape)



# TESTS
def test_histogramdd_discrete():
    cat_counts = [2, 3]
    x = np.array(
        [[0, 2], [0, 0], [1, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [0, 2], [1, 0]]
    )

    # size = 10
    # x = np.stack([np.random.randint(c, size=size) for c in cat_counts], axis=1)

    assert np.array_equal(histogramdd_indexes(x, cat_counts), [[1, 0, 2], [4, 3, 0]])


def test_histogram0d_discrete():
    x = np.empty(shape=(100, 0))
    print(histogramdd_indexes(x, []))
