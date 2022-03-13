import numpy as np
from sklearn.isotonic import isotonic_regression
from opendp.mod import enable_features

from post_histogram_synthetic_data import get_midpoints

enable_features("contrib")


def postprocess_histogram_monotonic_cumsum(counts):
    """Postprocess `counts` to be non-negative. 
    Only applicable if the counts correspond to ordered bins.
    
    Any negative count is inconsistent with the previous bin.
    Can significantly reduce error for small range queries.
    Error increases linearly as the range increases.

    See Section 3.1: https://arxiv.org/pdf/0904.0942.pdf
    """
    return np.diff(isotonic_regression(counts.cumsum()), prepend=0)


def test_monotonic():
    epsilon = .05
    sensitivity = 1
    # setup
    data = np.sqrt(np.random.uniform(size=1000))
    edges = np.sort(np.sqrt(np.random.uniform(size=100)))
    edge_indexes = list(range(len(edges)))

    # make measurement
    from opendp.trans import make_find_bin, make_count_by_categories
    from opendp.meas import make_base_geometric

    trans = make_find_bin(edges) >> make_count_by_categories(edge_indexes, TIA="usize")
    meas = trans >> make_base_geometric(sensitivity / epsilon, D="VectorDomain<AllDomain<i32>>")

    # release
    # the first bin is anything below the first edge and last bin is anything after the last edge
    exact_counts = np.array(trans(data)[1:-1])
    noisy_counts = np.array(meas(data)[1:-1])
    monot_counts = postprocess_histogram_monotonic_cumsum(noisy_counts)

    import matplotlib.pyplot as plt

    midpoints = get_midpoints(edges)
    plt.plot(midpoints, exact_counts.cumsum(), label="exact")
    plt.plot(midpoints, noisy_counts.cumsum(), label="noisy")
    plt.plot(midpoints, monot_counts.cumsum(), label="consistent")
    plt.legend()
    plt.show()


# test_monotonic()
