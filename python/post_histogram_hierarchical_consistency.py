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


def _get_isomorphic_axes(axes):
    # 1. consider a topological ordering over all possible sets of axes:
    #     choose any two sets of axes, A and B. A is greater than B if B.is_subset(A)
    # 2. this topological ordering on `axes` admits a directed acyclic graph
    #     choose any edge connecting two sets of axes, a parent and child.
    #     The parent should be equivalent to the summation of certain axes in the child
    # 3. find the transitive reduction of this dag to minimize the number of constraints
    # 4. construct equality constraints for all edges in the dag
    from itertools import permutations, chain

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
    return isomorphic_axes


def postprocess_hierarchical_histogramdd(hists, scales, nonnegative=False):
    """Make a set of noisy hypercubes at varying cross-tabulations consistent with each other.

    Runs in O(n^2) time. `postprocess_tree_histogramdd` is a linear time algorithm for linear hierarchies.

    :param hists: A dict of {axes: hist}, where each hist is a `len(axes)`-dimensional counts array.
    :param scales: a dict of {axes: scale}. Scale is the noise scale when privatizing.
    :returns the same data structure as `hists`, but each histogram is consistent with each other.
    """
    from scipy.optimize import fmin_slsqp

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
    isomorphic_axes = _get_isomorphic_axes(axes)

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


def _check_is_linear(axes):
    """Check that every histogram is a proper subset of its parent"""
    for parent, child in reversed(list(zip(axes[:-1], axes[1:]))):
        if not (child > parent):
            raise ValueError("histogram hierarchy is not linear")


def _axes_to_sum(child, parent):
    """Find the indexes of axes that should be summed in `child` to get `parent`"""
    return tuple(child.index(i) for i in child if i not in parent)


def _branching_factor(category_lengths, axes_to_sum):
    """branching factor between parent and child is the product of lengths of collapsed axes"""
    return np.prod(category_lengths[np.array(axes_to_sum)])


def _check_consistent(hists):
    axes = list(hists)
    for parent, child in reversed(list(zip(axes[:-1], axes[1:]))):
        assert np.allclose(
            hists[parent], hists[child].sum(axis=_axes_to_sum(child, parent))
        )


def postprocess_tree_histogramdd(hists, scales):
    """Make a set of noisy hypercubes of successive summations consistent with each other.
    Assumes that all hists were noised with the same noise scale.

    Runs in O(n) time.

    See 4.1: https://arxiv.org/pdf/0904.0942.pdf

    :param hists: A dict of {axes: hist}, where each hist is a `len(axes)`-dimensional counts array.
    :returns the leaf layer histogram
    """
    # sort the keys by number of axes
    hists = dict(sorted(hists.items(), key=lambda p: len(p[0])))
    # ensure all hists are float
    hists = {k: v.astype(float) for k, v in hists.items()}

    axes = list(hists)
    _check_is_linear(axes)  # algorithm assumes hierarchy is perfectly linear

    # find shape of each axis. Last histogram holds all axis lengths
    category_lengths = np.array(hists[axes[-1]].shape)

    # variance of postprocessed current layer. Starting at root, which is not postprocessed
    var = scales[axes[-1]] ** 2

    # bottom-up scan to compute z
    for parent, child in reversed(list(zip(axes[:-1], axes[1:]))):
        # we skip the root level
        axes_to_sum = _axes_to_sum(child=child, parent=parent)
        b = _branching_factor(category_lengths, axes_to_sum)

        # derive overall variance of parent after weighted averaging
        var = 1 / scales[parent]**2 + 1 / (b * var)

        # weight parent layer based on its proportion of overall variance
        alpha = (1 / scales[parent]**2) / var

        # hists[parent] has not been overriden because traversal order is bottom to top
        term1 = alpha * hists[parent]

        # hists[child] has been overwritten by previous loop
        term2 = (1 - alpha) * hists[child].sum(axis=axes_to_sum)

        hists[parent] = term1 + term2

    h_b = {a: h.copy() for a, h in hists.items()}

    # top down scan to compute h
    for parent, child in zip(axes[:-1], axes[1:]):
        axes_to_sum = _axes_to_sum(child=child, parent=parent)
        b = _branching_factor(category_lengths, axes_to_sum)

        correction = (h_b[parent] - hists[child].sum(axis=axes_to_sum)) / b
        h_b[child] += np.expand_dims(correction, axes_to_sum)

    # _check_consistent(h_b)

    # entire tree is consistent, so only the bottom layer is needed
    return h_b[axes[-1]]


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
    consistent_hist = postprocess_hierarchical_histogramdd(hierarchical_hist, scales)

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


def test_postprocess_tree_histogramdd():
    category_lengths = [2, 2, 4]

    hists = {(0, 1, 2): np.random.randint(0, 100, size=category_lengths)}
    hists[(0, 1)] = hists[(0, 1, 2)].sum(axis=2)
    hists[(0,)] = hists[(0, 1)].sum(axis=1)
    hists[()] = hists[(0,)].sum(axis=0)

    hists = dict(reversed(hists.items()))

    print(postprocess_tree_histogramdd(hists))


# test_postprocess_tree_histogramdd()


def test_postprocess_tree_histogramdd_2():
    cat_counts = [3, 4, 5, 7]
    size = 100
    x = np.stack([np.random.randint(c, size=size) for c in cat_counts], axis=1)

    ways = {(0, 1): 1.0, (0,): 0.5, (0, 1, 2, 3): 3.}

    noisy_hists = release_hierarchical_histogramdd_indexes(x, cat_counts, ways)
    
    final_counts = postprocess_tree_histogramdd(noisy_hists, ways)
    noisy_counts = noisy_hists[(0, 1, 2, 3)]
    exact_counts = histogramdd_indexes(x, cat_counts)

    final_mse = ((final_counts - exact_counts) ** 2).mean()
    noisy_mse = ((noisy_counts - exact_counts) ** 2).mean()
    print("final mse should be slightly smaller")
    print(f"{final_mse=}")
    print(f"{noisy_mse=}")
    print(f"{1 - final_mse / noisy_mse=:.4%} reduction in mse")
    # ~ 2-3% and counts are consistent


# test_postprocess_tree_histogramdd_2()
