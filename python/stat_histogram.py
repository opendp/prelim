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


def digitize_dataframe(df, labels, categorical):
    """Convert a pandas dataframe into an indexes array.

    :param df: columns must match the keys in `labels`
    :param labels: a {colname: list_of_edges_or_categories} dict
    :param categoricals: list of column names that should be considered categorical
    """
    columns = []
    for name in labels:
        if name in categorical:
            lookup = dict(zip(labels[name], range(len(labels[name]))))
            columns.append(df[name].map(lambda v: lookup[v]).values)
        else:
            # consider anything up to edge 2 to be part of bin 0
            #          anything above the second to last edge is part of the last bin
            columns.append(np.digitize(df[name].values, labels[name][1:-1]))

    return np.stack(columns, axis=1)



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


def test_digitize_dataframe():
    import pandas as pd

    data = pd.DataFrame(
        {
            "A": [1, 2, 4, 2],
            "B": ["nebraska", "montana", "idaho", "idaho"],
            "C": [0.1, 0.2, 0.5, 0.7],
        }
    )

    labels = {
        "A": [1, 2, 3, 4],  # categories
        "B": ["nebraska", "montana", "idaho", "utah"],  # categories
        "C": [0.0, 0.3, 0.5, 1.0],  # edges
    }

    categorical_names = ["A", "B"]

    digitized = digitize_dataframe(data, labels, categorical_names)
    ground_truth = [[0, 0, 0], [1, 1, 0], [3, 2, 2], [1, 2, 2]]

    # check that the digitization is correct
    assert np.array_equal(digitized, ground_truth)

    ### SYNTHETIC DATA DEMO
    # demonstrate generating synthetic data with a mixture of categorical and continuous variables
    from post_histogram_synthetic_data import (
        postprocess_synthesize_categorical,
        get_midpoints,
    )

    # standardize labels, which could be categories or edges, into categories
    categories = []
    for name in labels:
        if name in categorical_names:
            categories.append(np.array(labels[name]))
        else:
            categories.append(get_midpoints(labels[name]))

    # make a dp release of the digitized dataset
    dp_release = release_histogramdd_indexes(digitized, list(map(len, categories)), 1.0)
    
    # synthesize data from the empirical cdf
    synthetic_data = postprocess_synthesize_categorical(
        dp_release, categories, size=100
    )

    print(pd.DataFrame(synthetic_data, columns=data.columns))

