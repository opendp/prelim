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

from stat_histogram import histogramdd_indexes
from opendp.meas import make_base_geometric


def release_hierarchical_histogramdd_indexes(
    x: np.ndarray, category_lengths: List[int], hierarchy
) -> List[np.ndarray]:
    """Release histogram estimates at different levels of aggregation.

    :param x: data of shape [n, len(`category_lengths`)] of non-negative category indexes
    :param category_lengths: the number of unique categories per column
    :param hierarchy: a dict of {axes: scale}. Each key is a tuple of axes to preserve. Scale is the noise scale when privatizing.
    :returns a list of len(`hierarchy`) d-dimensional histograms, where each d is len(axes) per item in the hierarchy
    """
    hist = histogramdd_indexes(x, category_lengths)

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


def postprocess_hierarchical_histogramdd(hists, scales, nonnegative=False):
    """Make a set of noisy hypercubes at varying cross-tabulations consistent with each other.

    :param hists: A dict of {axes: hist}, where each hist is a `len(axes)`-dimensional counts array.
    :param scales: a dict of {axes: scale}. Scale is the noise scale when privatizing.
    :returns the same data structure as `hists`, but each histogram is consistent with each other.
    """
    from scipy.optimize import fmin_slsqp
    from itertools import permutations, chain

    axes = list(hists.keys())
    assert axes == list(scales.keys()), "`hists` and `scales` must share the same keys"

    # A. translation between hists and flat representation
    # We will solve for consistency in the flattened space
    x = np.concatenate([hist.ravel() for hist in hists.values()])
    offsets = np.array([hist.size for hist in hists.values()])[:-1].cumsum()

    def flat_to_hierarchical(z):
        """Translate the lsq flat vector into the hierarchical histogram representation."""
        return {
            axes_i: z_i.reshape(hist.shape)
            for z_i, (axes_i, hist) in zip(np.split(z, offsets), hists.items())
        }

    # B. weight the error function by inverse-variance
    # laplace/geometric and gaussian share the same weights
    # - laplace/geometric variance is 2 * scale ^ 2, and the 2 cancels with the denominator
    # - gaussian variance is scale ^ 2
    weights = np.concatenate(
        [
            np.full(np.prod(hist.shape), 1 / scale**2)
            for hist, scale in zip(hists.values(), scales.values())
        ]
    )
    weights /= weights.sum()

    # C. find the smallest set of pairs of axes that must be equivalent under summation
    #   1. consider a topological ordering over all possible sets of axes:
    #      choose any two sets of axes, A and B. A is greater than B if B.is_subset(A)
    #   2. this topological ordering on `axes` admits a directed acyclic graph
    #      choose any edge connecting two sets of axes, a parent and child.
    #      The parent should be equivalent to the summation of certain axes in the child
    #   3. find the transitive reduction of this dag to minimize the number of constraints
    #   4. construct equality constraints for all edges in the dag
    dag = {axes_i: [] for axes_i in axes}

    def descendants(v):
        # all children of v, and all descendents of all children of v
        return [*dag[v], *chain(descendants(c) for c in list(dag[v]))]

    for parent, child in permutations(axes, 2):
        if set(parent).issubset(child) and child not in descendants(parent):
            dag[parent].append(child)
    
    isomorphic_axes = []
    for v in dag:
        isomorphic_axes.extend([(v, c) for c in dag[v]])

    # D. represent the equality of isomorphic axes with an error function
    def f_eqcons(y):
        """Computes how much y violates the set of equality constraints that are necessary for consistency."""

        # reconstruct hierarchical histogram from the lsq vector
        hists_y = flat_to_hierarchical(y)

        # compute the error for each equality constraint
        errors = []
        for parent, child in isomorphic_axes:
            axes_to_sum = tuple(child.index(i) for i in child if i not in parent)
            errors.append(hists_y[parent] - hists_y[child].sum(axis=axes_to_sum))

        return np.concatenate([e.ravel() for e in errors])

    # E. Find a y with the smallest l2 distance from x, such that counts are consistent
    y = fmin_slsqp(
        func=lambda y: np.linalg.norm((x - y) * weights),
        x0=x,
        f_eqcons=f_eqcons,
        f_ieqcons=(lambda x: x) if nonnegative else None,
    )
    return flat_to_hierarchical(y)


# TESTS
def test_release_hierarchical_histogramdd_discrete():
    cat_counts = [2, 3]
    x = np.array(
        [[0, 2], [0, 0], [1, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [0, 2], [1, 0]]
    )
    # size = 100
    # cat_counts = [3, 5, 7]
    # x = np.stack([np.random.randint(c, size=size) for c in cat_counts], axis=1)

    ways = {(0, 1): 0.0, (0,): 0.0, (1,): 0.0}

    way01, way0, way1 = release_hierarchical_histogramdd_indexes(x, cat_counts, ways)

    assert np.array_equal(way01, [[1, 0, 2], [4, 3, 0]])
    assert np.array_equal(way0, [3, 7])
    assert np.array_equal(way1, [5, 3, 2])


# test_release_hierarchical_histogramdd_discrete()


def test_postprocess_hierarchical_histogramdd_discrete():
    cat_counts = [2, 3]
    size = 400
    x = np.stack([np.random.randint(c, size=size) for c in cat_counts], axis=1)

    # want to release a histogram over axes 0 and 1 with noise scale of 1,
    #                 a histogram over axis 0 with noise scale of 1,
    #             and a histogram over axis 1 with a noise scale of 1
    scales = {(0, 1): 1.0, (0,): 1.0, (1,): 1.0}

    hierarchical_hist = release_hierarchical_histogramdd_indexes(x, cat_counts, scales)
    consistent_hist = postprocess_hierarchical_histogramdd(
        hierarchical_hist, scales
    )

    print("noisy:     ", hierarchical_hist)
    print("consistent:", consistent_hist)


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
        (5,): 1.0,
        # (): .3
    }

    hierarchical_hist = release_hierarchical_histogramdd_indexes(
        x, cat_counts, hierarchy
    )

    consistent_hist = postprocess_hierarchical_histogramdd(
        hierarchical_hist, hierarchy, nonnegative=False
    )

    for axes in hierarchy:
        print("axes:      ", axes)
        print("noisy:     ", hierarchical_hist[axes])
        print("consistent:", consistent_hist[axes].astype(int))


# test_postprocess_hierarchical_histogramdd_discrete_big()
