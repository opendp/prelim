import numpy as np


def _height_from_leaf_count(n, b):
    """Height of a balanced `b`-ary tree with `n` leaf nodes."""
    return np.ceil(np.log(n) / np.log(b)).astype(int) + 1


def transform_b_ary_tree(leaf_counts, b):
    """Transforms a 1d-array of leaf counts into a balanced b-ary tree implicitly stored in breadth-first order.

    :param leaf_counts: counts of each of the leaf nodes of the b-ary tree
    :param m: the maximum number of children per node
    :returns An array where the first element is the root sum,
             the next m are its summand children,
             the next b^2 are their children,
             and so on.
    """
    # find height of the rest of the tree
    height = _height_from_leaf_count(len(leaf_counts), b) - 1

    pad_length = b**height - len(leaf_counts)

    base = np.concatenate([leaf_counts, np.zeros(pad_length, dtype=int)]).reshape(
        (b,) * height
    )

    levels = []
    for _ in range(height):
        base = base.sum(axis=-1)
        levels.append(base.ravel())

    return np.concatenate([*levels[::-1], leaf_counts])


def transform_b_ary_tree_relation(num_leaves, b, sensitivity):
    """Derive sensitivity of a balanced `b`-ary tree from the sensitivity of the leaf nodes.
    Proposition 4: https://arxiv.org/pdf/0904.0942.pdf"""
    return sensitivity * _height_from_leaf_count(num_leaves, b)


def _layers_from_nodes(num_nodes, b):
    return np.ceil(np.log((b - 1) * num_nodes + 1) / np.log(b)).astype(int)


def _nodes_from_layers(layers, b):
    return (b**layers - 1) // (b - 1)


def choose_b(n):
    """Choose the optimal branching factor.
    Proposition 1: http://www.vldb.org/pvldb/vol6/p1954-qardaji.pdf
    From "Optimal Branching Factor", try different values of b, up to flat.

    :param n: ballpark estimate of dataset size
    """

    def v_star_avg(b):
        """Formula (3) estimates variance"""
        h = np.ceil(np.log(n) / np.log(b))
        return (b - 1) * h**3 - 2 * (b + 1) * h**2 / 3

    # find the b with minimum average variance
    return min(range(2, n + 1), key=v_star_avg)


def postprocess_b_ary_tree(tree, b):
    """Postprocess a balanced `b`-ary tree to be consistent.

    Tree is assumed to be complete, as in, all leaves on the last layer are on the left.
    Non-existent leaves are assumed to be zero.

    See 4.1: https://arxiv.org/pdf/0904.0942.pdf

    At the cost of implementation complexity, output remains consistent even when leaf nodes are missing

    :param tree: a balanced `b`-ary tree implicitly stored in breadth-first order
    :param m: the maximum number of children
    :returns the consistent leaf nodes
    """
    layers = _layers_from_nodes(len(tree), b)
    # number of nodes in a perfect b-ary tree
    vars = np.ones(_nodes_from_layers(layers, b))
    zero_leaves = len(vars) - len(tree)

    tree = np.concatenate([tree, np.zeros(zero_leaves)])

    # zero out all zero variance zero nodes on tree
    for l in range(layers):
        l_zeros = zero_leaves // (b ** (layers - l - 1))
        l_end = _nodes_from_layers(l + 1, b)
        vars[l_end - l_zeros : l_end] = 0
        tree[l_end - l_zeros : l_end] = 0

    # bottom-up scan to compute z
    for l in reversed(range(layers - 1)):
        l_start = _nodes_from_layers(l, b)
        for offset in range(b**l):
            i = l_start + offset
            if vars[i] == 0:
                continue
            child_slice = slice(i * b + 1, i * b + 1 + b)

            child_var = vars[child_slice].sum()
            child_val = tree[child_slice].sum()

            # weight to give to self (part 1)
            alpha = 1 / vars[i]

            # update total variance of node to reflect postprocessing
            vars[i] = 1 / (1 / vars[i] + 1 / child_var)

            # weight to give to self (part 2)
            # weight of self is a proportion of total inverse variance (total var / prior var)
            alpha *= vars[i]

            # postprocess by weighted inverse variance
            tree[i] = alpha * tree[i] + (1 - alpha) * child_val

    # top down scan to compute h
    h_b = tree.copy()
    for l in range(layers - 1):
        l_start = _nodes_from_layers(l, b)

        for offset in range(b**l):
            i = l_start + offset
            child_slice = slice(i * b + 1, i * b + 1 + b)
            child_vars = vars[child_slice]

            # children need to be adjusted by this amount to be consistent with parent
            correction = h_b[i] - tree[child_slice].sum()
            if correction == 0.0:
                continue

            # apportion the correction among children relative to their variance
            h_b[child_slice] += correction * child_vars / child_vars.sum()

    # entire tree is consistent, so only the nonzero leaves in bottom layer are needed
    leaf_start = _nodes_from_layers(layers - 1, b)
    leaf_end = _nodes_from_layers(layers, b) - zero_leaves
    return h_b[leaf_start:leaf_end]


# TESTS
def test_release_b_ary_tree():
    # setup
    leaf_counts = np.random.randint(0, 2000, size=11)
    # leaf_counts = np.array([2, 3, 1, 4, 5, 6, 3]) * 20
    m = 3
    sensitivity = 1
    epsilon = 1

    # run transformation
    tree = transform_b_ary_tree(leaf_counts, b=m)
    sensitivity = transform_b_ary_tree_relation(len(leaf_counts), m, sensitivity)

    # privatize
    from opendp.meas import make_base_geometric
    from opendp.mod import enable_features

    enable_features("contrib")

    mech = make_base_geometric(sensitivity / epsilon, D="VectorDomain<AllDomain<i64>>")
    tree_noisy = np.array(mech(tree))
    post_counts = postprocess_b_ary_tree(tree_noisy, m)
    leaf_counts_noisy = tree_noisy[-len(leaf_counts) :]

    print("exact tree:", tree)
    print("noisy tree:", tree_noisy)

    print("exact leaves:", leaf_counts)
    print("noisy leaves:", leaf_counts_noisy)
    print("post  leaves:", post_counts.astype(int))

    print("exact sum:", leaf_counts.sum())
    print("post  sum:", post_counts.sum())

    final_mse = ((post_counts - leaf_counts) ** 2).mean()
    noisy_mse = ((leaf_counts_noisy - leaf_counts) ** 2).mean()
    print("final mse should be slightly smaller")
    print(f"{final_mse=}")
    print(f"{noisy_mse=}")
    print(f"{1 - final_mse / noisy_mse=:.4%} reduction in mse")
    # ~ 30% reduction in mse


def test_layers_from_nodes():
    assert _layers_from_nodes(1, b=2) == 1
    assert _layers_from_nodes(2, b=2) == 2
    assert _layers_from_nodes(3, b=2) == 2
    assert _layers_from_nodes(7, b=2) == 3
    assert _layers_from_nodes(8, b=2) == 4

    assert _layers_from_nodes(2, b=3) == 2
    assert _layers_from_nodes(4, b=3) == 2
    assert _layers_from_nodes(5, b=4) == 2
    assert _layers_from_nodes(13, b=3) == 3
    assert _layers_from_nodes(14, b=3) == 4


# test_release_b_ary_tree()


# TESTS
def test_tree_and_hist_equal():
    # setup
    leaf_counts = np.random.randint(0, 2000, size=27)
    # leaf_counts = np.array([2, 3, 1, 4, 5, 6, 3]) * 20
    m = 3
    sensitivity = 1
    epsilon = 1
    scale = sensitivity / epsilon

    # run transformation
    tree = transform_b_ary_tree(leaf_counts, b=m)
    sensitivity = transform_b_ary_tree_relation(len(leaf_counts), m, sensitivity)

    # privatize
    from opendp.meas import make_base_geometric
    from opendp.mod import enable_features

    enable_features("contrib")

    mech = make_base_geometric(scale, D="VectorDomain<AllDomain<i64>>")
    tree_noisy = np.array(mech(tree))
    post_counts_tree = postprocess_b_ary_tree(tree_noisy, m)

    from post_histogram_hierarchical_consistency import postprocess_tree_histogramdd

    colla = np.array(tree[13:]).reshape((3, 3, 3))
    collb = np.array(tree[4:13]).reshape((3, 3))
    collc = np.array(tree[1:4])
    colld = np.array(tree[0])

    assert np.array_equal(colla.sum(axis=2), collb), "collapse axis is wrong"
    assert np.array_equal(collb.sum(axis=1), collc), "collapse axis is wrong"
    assert np.array_equal(collc.sum(axis=0), colld), "collapse axis is wrong"

    post_counts_hist = postprocess_tree_histogramdd(
        {
            (): np.array(tree_noisy[0]),
            (0,): np.array(tree_noisy[1:4]),
            (0, 1): np.array(tree_noisy[4:13]).reshape((3, 3)),
            (0, 1, 2): np.array(tree_noisy[13:]).reshape((3, 3, 3)),
        },
        {
            (): scale,
            (0,): scale,
            (0, 1): scale,
            (0, 1, 2): scale,
        },
    ).ravel()

    assert np.array_equal(post_counts_hist, post_counts_tree)


# test_tree_and_hist_equal()
