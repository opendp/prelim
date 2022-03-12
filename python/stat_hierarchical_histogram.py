# BCDK+07 A Holistic Solution to Contingency Table Release
# * paper https://cseweb.ucsd.edu//~kamalika/pubs/bcdkmt07.pdf

# HRMS09 Boosting the Accuracy of Differentially-Private Histograms Through Consistency
# * paper https://arxiv.org/pdf/0904.0942.pdf

# LHR+10 Optimizing Linear Counting Queries Under Differential Privacy
# * paper https://people.cs.umass.edu/~mcgregor/papers/10-pods.pdf
# computationally heavy

# MMHM18 Optimizing error of high-dimensional statistical queries under differential privacy.
# * paper https://arxiv.org/pdf/1808.03537.pdf

# ENU19 The Power of Factorization Mechanisms in Local and Central Differential Privacy
# * paper https://arxiv.org/pdf/1911.08339.pdf

# YXLLW21 Improved Matrix Gaussian Mechanism for Differential Privacy
# * paper https://arxiv.org/pdf/2104.14808.pdf

# QYL13 HB
# * paper http://www.vldb.org/pvldb/vol6/p1954-qardaji.pdf
# implementation candidate


from typing import List
import numpy as np
from opendp.meas import make_base_geometric
from opendp.mod import enable_features

enable_features("contrib")


def histogramdd_discrete(x: np.ndarray, category_lengths: List[int]) -> np.ndarray:
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


def release_histogramdd_discrete(
    x: np.ndarray, category_lengths: List[int], scale
) -> np.ndarray:
    """Release a d-dimensional histogram with noise `scale`

    :param x: data of shape [n, len(`category_lengths`)] of non-negative category indexes
    :param category_lengths: the number of unique categories per column
    """
    hist = histogramdd_discrete(x, category_lengths)
    meas = make_base_geometric(scale, D="VectorDomain<AllDomain<i64>>")
    return meas(hist.flatten()).reshape(hist.shape)


def release_hierarchical_histogramdd_discrete(
    x: np.ndarray, category_lengths: List[int], hierarchy
) -> List[np.ndarray]:
    """Release histogram estimates at different levels of aggregation.

    :param x: data of shape [n, len(`category_lengths`)] of non-negative category indexes
    :param category_lengths: the number of unique categories per column
    :param hierarchy: a dict of {axes: scale}. Each key is a tuple of axes to preserve. Scale is the noise scale when privatizing.
    :returns a list of len(`hierarchy`) d-dimensional histograms, where each d is len(axes) per item in the hierarchy
    """
    hist = histogramdd_discrete(x, category_lengths)

    hierarchical_hist = {}
    for axes, scale in hierarchy.items():
        meas = make_base_geometric(scale, D="VectorDomain<AllDomain<i64>>")

        marginal = hist.sum(
            axis=tuple(i for i in range(len(category_lengths)) if i not in axes)
        )
        hierarchical_hist[axes] = np.array(meas(marginal.flatten())).reshape(
            marginal.shape
        )

    return hierarchical_hist


def _constrained_inference(x, weights, f_eqcons, nonnegative=False):
    from scipy.optimize import fmin_slsqp

    return fmin_slsqp(
        func=lambda y: np.linalg.norm((x - y) * weights),
        x0=x,
        f_eqcons=f_eqcons,
        f_ieqcons=(lambda x: x) if nonnegative else None,
    )


def _topological_neighbors(hierarchy):
    import itertools

    dag = {axes: [] for axes in hierarchy}

    def descendants(v):
        output = list(dag[v])
        for c in dag[v]:
            output.extend(descendants(c))
        return output

    for parent, child in itertools.permutations(hierarchy, 2):
        if set(parent).issubset(child) and child not in descendants(parent):
            dag[parent].append(child)

    edges = []
    for v in dag:
        edges.extend([(v, c) for c in dag[v]])

    return edges


def postprocess_hierarchical_histogramdd_discrete(
    hierarchical_hist, category_lengths, hierarchy, nonnegative=False
):
    """Make a set of noisy hypercubes at varying cross-tabulations consistent with each other."""
    category_lengths = np.array(category_lengths)

    # 1. flatten into one vector
    hist_flat = np.concatenate([hist.ravel() for hist in hierarchical_hist.values()])

    # 2. weight the error function by inverse-variance
    # laplace/geometric and gaussian share the same weights
    # - laplace/geometric variance is 2 * scale ^ 2, and the 2 cancels with the denominator
    # - gaussian variance is scale ^ 2
    weights = np.concatenate(
        [np.full(np.prod(category_lengths[np.array(axes, dtype=int)]), 1 / scale**2) for axes, scale in hierarchy.items()]
    )
    weights /= weights.sum()

    # 3. construct equality constraints for all edges
    # 3.a construct topological ordering that emits a directed acyclic graph, transitive reduction
    topological_neighbors = _topological_neighbors(hierarchy)
    offsets = np.array([hist.size for hist in hierarchical_hist.values()])[:-1].cumsum()

    def unpack(x):
        return {
            axes: x_i.reshape(category_lengths[np.array(axes, dtype=int)])
            for x_i, axes in zip(np.split(x, offsets), hierarchy)
        }

    def f_eqcons(x):
        # reconstruct hierarchical histogram from the lsq vector
        hierarchical_hist_p = unpack(x)

        errors = []
        for parent, child in topological_neighbors:
            errors.append(
                hierarchical_hist_p[parent]
                - hierarchical_hist_p[child].sum(
                    axis=tuple(child.index(i) for i in child if i not in parent)
                )
            )

        return np.concatenate([e.ravel() for e in errors])

    return unpack(
        _constrained_inference(hist_flat, weights, f_eqcons, nonnegative=nonnegative)
    )








# TESTS
def test_histogramdd_discrete():
    cat_counts = [2, 3]
    x = np.array(
        [[0, 2], [0, 0], [1, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [0, 2], [1, 0]]
    )

    # size = 10
    # x = np.stack([np.random.randint(c, size=size) for c in cat_counts], axis=1)

    assert np.array_equal(histogramdd_discrete(x, cat_counts), [[1, 0, 2], [4, 3, 0]])


def test_histogram0d_discrete():
    x = np.empty(shape=(100, 0))
    print(histogramdd_discrete(x, []))


def test_release_hierarchical_histogramdd_discrete():
    cat_counts = [2, 3]
    x = np.array(
        [[0, 2], [0, 0], [1, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [0, 2], [1, 0]]
    )
    # size = 100
    # cat_counts = [3, 5, 7]
    # x = np.stack([np.random.randint(c, size=size) for c in cat_counts], axis=1)

    ways = {(0, 1): 0.0, (0,): 0.0, (1,): 0.0}

    way01, way0, way1 = release_hierarchical_histogramdd_discrete(x, cat_counts, ways)

    assert np.array_equal(way01, [[1, 0, 2], [4, 3, 0]])
    assert np.array_equal(way0, [3, 7])
    assert np.array_equal(way1, [5, 3, 2])


# test_release_hierarchical_histogramdd_discrete()


def test_postprocess_hierarchical_histogramdd_discrete():
    cat_counts = [2, 3]
    size = 400
    x = np.stack([np.random.randint(c, size=size) for c in cat_counts], axis=1)

    ways = {(0, 1): 1.0, (0,): 1.0, (1,): 1.0}

    hierarchical_hist = release_hierarchical_histogramdd_discrete(x, cat_counts, ways)

    solved = postprocess_hierarchical_histogramdd_discrete(
        hierarchical_hist, cat_counts, ways
    )

    print("noisy:     ", hierarchical_hist)
    print("consistent:", solved)


# test_postprocess_hierarchical_histogramdd_discrete()


def test_postprocess_hierarchical_histogramdd_discrete_big():
    cat_counts = [2, 3, 7, 4, 2, 3]
    size = 10_000
    x = np.stack([np.random.randint(c, size=size) for c in cat_counts], axis=1)

    hierarchy = {
        (0, 1, 2, 3, 4, 5): 1.0,  # 6-dimensional array of cell counts
        (0,): 1.0,  # histogram of cell counts along variable 0
        (1,): 1.0,  # histogram of cell counts along variable 1
        (2, 3): 2.0,  # 2d histogram of cell counts over variables 2 and 3
        (3, 4, 5): 1.0, 
        (4, 5): 0.5,  # 2d histogram of cell counts over variables 4 and 5
        (5,): 1.,
        # (): .3
    }

    hierarchical_hist = release_hierarchical_histogramdd_discrete(
        x, cat_counts, hierarchy
    )

    consistent_hist = postprocess_hierarchical_histogramdd_discrete(
        hierarchical_hist, cat_counts, hierarchy, nonnegative=True
    )

    for axes in hierarchy:
        print("axes:      ", axes)
        print("noisy:     ", hierarchical_hist[axes])
        print("consistent:", consistent_hist[axes])


test_postprocess_hierarchical_histogramdd_discrete_big()
