import numpy as np


def _height_from_leaf_count(n, m):
    """Height of a balanced m-ary tree with `n` leaf nodes."""
    return np.ceil(np.log(n) / np.log(m)).astype(int) + 1


def transform_m_ary_tree(leaf_counts, m):
    """Transforms a 1d-array of leaf counts into a balanced m-ary tree implicitly stored in breadth-first order.

    :param leaf_counts: counts of each of the leaf nodes of the m-ary tree
    :param m: the maximum number of children per node
    :returns An array where the first element is the root sum,
             the next m are its summand children,
             the next m^2 are their children,
             and so on.
    """
    # find height of the rest of the tree
    height = _height_from_leaf_count(len(leaf_counts), m) - 1

    pad_length = m**height - len(leaf_counts)

    base = np.concatenate([leaf_counts, np.zeros(pad_length, dtype=int)]).reshape(
        (m,) * height
    )

    levels = []
    for _ in range(height):
        base = base.sum(axis=-1)
        levels.append(base.ravel())

    return np.concatenate([*levels[::-1], leaf_counts])


def transform_m_ary_tree_relation(num_leaves, m, sensitivity):
    """Derive sensitivity of a balanced m-ary tree from the sensitivity of the leaf nodes.
    Proposition 4: https://arxiv.org/pdf/0904.0942.pdf"""
    return sensitivity * _height_from_leaf_count(num_leaves, m)


def postprocess_m_ary_tree(tree, m):
    """Postprocess a balanced `m`-ary tree to be consistent.

    Tree is assumed to be complete, as in, all leaves on the last layer are on the left.
    Non-existent leaves are assumed to be zero.

    See 4.1: https://arxiv.org/pdf/0904.0942.pdf

    :param tree: a balanced `m`-ary tree implicitly stored in breadth-first order
    :param m: the maximum number of children
    :returns the consistent leaf nodes
    """
    h_t = np.array(tree, dtype=float)

    # height of tree
    l = np.ceil(np.log(len(h_t)) / np.log(m)).astype(int)
    last_row = m ** (l - 1) - 1

    def children(v):
        """return a slice into the children of v"""
        v_first_child = v * m + 1
        v_last_child = min(v_first_child + m, len(h_t))
        return slice(v_first_child, v_last_child)

    # bottom-up scan to compute z
    term1 = (m**l - m ** (l - 1)) / (m**l - 1)
    term2 = (m ** (l - 1) - 1) / (m**l - 1)
    z = h_t.copy()
    for v in np.arange(last_row)[::-1]:
        z[v] = term1 * h_t[v] + term2 * z[children(v)].sum()

    # top down scan to compute h
    h_b = z.copy()
    for v in range(1, len(h_t)):
        u = (v - 1) // m  # parent index
        h_b[v] += (h_b[u] - z[children(u)].sum()) / m

    # entire tree is consistent, so only the bottom layer is needed
    return h_b[last_row:]



# TESTS
def test_release_m_ary_tree():
    # setup
    leaf_counts = np.array([2, 3, 1, 4, 5, 6, 7]) * 20
    m = 2
    sensitivity = 1
    epsilon = 1

    # run transformation
    tree = transform_m_ary_tree(leaf_counts, m=m)
    sensitivity = transform_m_ary_tree_relation(len(leaf_counts), m, sensitivity)

    # privatize
    from opendp.meas import make_base_geometric
    from opendp.mod import enable_features

    enable_features("contrib")

    mech = make_base_geometric(sensitivity / epsilon, D="VectorDomain<AllDomain<i64>>")
    tree_noisy = np.array(mech(tree))

    print("exact tree:", tree)
    print("noisy tree:", tree_noisy)

    print("exact leaves:", leaf_counts)
    print("noisy leaves:", tree_noisy[-len(leaf_counts) :])
    print("post  leaves:", postprocess_m_ary_tree(tree_noisy, m).round().astype(int))


# test_release_m_ary_tree()
