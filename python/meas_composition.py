import numpy as np

### MEASURES
# pureDP          pure                   epsilon-DP
# approxDP        approximate            (epsilon, delta)-DP
# zCDP            zero concentrated      (xi, rho)-zCDP         renyi divergence for all alpha
# smoothedzCDP    approximate zero conc  (xi, rho, delta)-zCDP  the delta is equivalent to approxDP
# renyiDP         renyi                  (alpha, epsilon')-RDP

### COMPOSITION
# composition_[measure]_[static|dynamic]_[homo|hetero]_[name]

# "static" when the choice of distances is fixed up-front
# "dynamic" when the choice of parameters is chosen adaptively

# "hetero" for heterogeneous, where each epsilon_i and delta_i may vary
# "homo" for homogeneous, where all k queries share the same `distance_0`.
#   Omitted if a trivial simplification of heterogeneous composition


def composition_approxDP_static_hetero_basic(distance_is):
    """apply composition on `distance_is`, a list of individual distances

    :param distance_is: a list of (epsilon, delta), or ndarray of shape [k, 2]
    """
    epsilon_is, delta_is = zip(*distance_is)
    return sum(epsilon_is), sum(delta_is)


def composition_approxDP_static_homo_advanced(distance_0, k, delta_p):
    """apply composition on `distance_0` in k-folds

    "advanced" composition from Theorem 3.3 in https://guyrothblum.files.wordpress.com/2014/11/drv10.pdf
    Sometimes also referred to as "strong" composition.

    :param distance_0: per-query epsilon, delta
    :param k: how many folds, number of queries
    :param delta_p: how much additional delta to add, beyond basic composition of `delta_0`
    :returns global (epsilon, delta) of k-fold composition of a (epsilon_0, delta_0)-DP mechanism
    """
    epsilon_0, delta_0 = distance_0
    epsilon_g = np.sqrt(2 * k * np.log(1 / delta_p)) * epsilon_0 + k * epsilon_0 * (
        np.exp(epsilon_0) - 1
    )
    delta_g = delta_0 * k + delta_p
    return epsilon_g, delta_g


def composition_approxDP_static_homo_optimal_analytic(distance_0, k, delta_p):
    """apply composition on `distance_0` in k-folds

    "optimal" composition from KOV15
    "analytic" because this is the looser closed form expression in Theorem 3.5: https://arxiv.org/pdf/1311.0776.pdf#subsection.3.3

    :param distance_0: (epsilon, delta)
    :param delta_p: p as in prime. Slack term for delta. Allows for nontrivial epsilon composition
    """
    eps_0, del_0 = distance_0

    bound1 = k * eps_0
    bound2 = k * eps_0**2 + eps_0 * np.sqrt(
        2 * k * np.log(np.exp(1) + np.sqrt(k * eps_0**2) * delta_p)
    )
    bound3 = k * eps_0**2 + eps_0 * np.sqrt(2 * k * np.log(1 / delta_p))

    # Corresponds to Theorem 3.5 in KOV15. Ignoring nan.
    epsilon = np.nanmin([bound1, bound2, bound3])

    delta = 1 - (1 - delta_p) * (1 - del_0) ** k

    return epsilon, delta


def composition_approxDP_static_hetero_optimal_analytic(distance_is, delta_p):
    """Find the (epsilon, delta) composition of `distances_is`.

    "optimal" composition from KOV15
    "analytic" because this is the looser closed form expression in Theorem 3.5: https://arxiv.org/pdf/1311.0776.pdf#subsection.3.3

    :param distance_is: a list of (epsilon, delta), or ndarray of shape [k, 2]
    :param delta_p: slack term for delta. Allows for tighter composition on epsilons
    """
    epsilon_is, delta_is = np.array(distance_is).T

    sum_of_squares = (epsilon_is**2).sum()
    first_term = sum(ep * (np.exp(ep) - 1) / (np.exp(ep) + 1) for ep in epsilon_is)

    # want to find the smallest of three bounds
    bound1 = sum(epsilon_is)
    bound2 = first_term + np.sqrt(
        (2 * np.log(np.exp(1) + (np.sqrt(sum_of_squares) / delta_p))) * sum_of_squares
    )
    bound3 = first_term + np.sqrt(2 * np.log(1 / delta_p) * sum_of_squares)

    # Corresponds to Theorem 3.5 in KOV15. Ignoring nan.
    epsilon = np.nanmin([bound1, bound2, bound3])

    delta = 1 - (1 - delta_p) * np.prod(1 - delta_is)

    return epsilon, delta


def composition_zCDP_static_hetero(distance_is):
    """Find the global (xi, rho)-zCDP composition of a list of (xi_i, rho_i)
    Lemma 1.7: https://arxiv.org/pdf/1605.02065.pdf#subsubsection.1.2.3
    """
    xi_is, rho_is = np.array(distance_is).T
    return sum(xi_is), sum(rho_is)


def composition_renyiDP_static_hetero(alpha, epsilon_is):
    """Find the global (alpha, epsilon)-RDP composition of a list of epsilons of order-alpha renyi measure
    Proposition 4: https://arxiv.org/pdf/1702.07476.pdf#section.5
    Which extends Lemma 1 to the case of n mechanisms
    """
    return alpha, sum(epsilon_is)


def composition_approxDP_static_homo_shuffle_composition_analytic(distance_0, k):
    """Find the global (epsilon, delta)-DP composition of `k` (epsilon_0, delta_0)-DP releases.
    The dataset is shuffled and each release made on a disjoint subset.
    This uses the weaker, simpler, analytic bound.

    Partial Rust implementation: https://github.com/opendp/smartnoise-core/blob/f81dfcb5cd48a2e4331e50577c97ea67281a48a6/runtime-rust/src/utilities/shuffling.rs
    """
    epsilon_0, delta_0 = distance_0
    from utils.shuffling_amplification import closedformanalysis

    return closedformanalysis(k, epsilon_0, delta_0)


def composition_approxDP_static_homo_shuffle_composition_empirical(
    distance_0, k, iterations=10, step=100, bound="upper"
):
    """Find the global (epsilon, delta)-DP composition of `k` (epsilon_0, delta_0)-DP releases.
    The dataset is shuffled and each release made on a disjoint subset.
    This uses the stronger analytic bound.

    Partial Rust implementation: https://github.com/opendp/smartnoise-core/blob/f81dfcb5cd48a2e4331e50577c97ea67281a48a6/runtime-rust/src/utilities/shuffling.rs

    :param distance_0: (epsilon_0, delta_0) for each individual release
    :param k: number of releases
    :param iterations: number of iterations of binary search. The higher T is, the more accurate the result
    :param step: The larger step is, the less accurate the result, but more efficient the algorithm.
    :param bound: One of {"lower", "upper"}.
    """
    epsilon_0, delta_0 = distance_0
    assert bound in ("upper", "lower")

    from utils.shuffling_amplification import numericalanalysis

    return numericalanalysis(
        k,
        epsilon_0,
        delta_0,
        num_iterations=iterations,
        step=step,
        upperbound=bound == "upper",
    )


def composition_tCDP_static_hetero(distance_is):
    """Find the global (rho, omega)-tCDP composition of a list of (rho_i, omega_i)
    Section 1.1: https://projects.iq.harvard.edu/files/privacytools/files/bun_mark_composable_.pdf#%5B%7B%22num%22%3A40%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22Fit%22%7D%5D
    """
    rho_is, omega_is = np.array(distance_is).T
    return sum(rho_is), omega_is.min()


### SOLVE
# solve_[measure]_[static|dynamic]_[homo|hetero]_[name]
def solve_approxDP_static_homo_basic(distance_g, k):
    """solve for `distance_0`, the per-query epsilon and delta

    :param distance_g: global (epsilon, delta)
    :param k: compose to `distance_g` in `k` folds
    """
    epsilon_g, delta_g = distance_g
    return epsilon_g / k, delta_g / k


def solve_approxDP_static_homo_advanced(distance_g, k, delta_p):
    """Solve for the per-query (epsilon, delta) such that k-fold composition sums to (epsilon_g, delta_g)"""
    # TODO: binary search can make this tighter
    epsilon_g, delta_g = distance_g
    epsilon_0 = epsilon_g / np.sqrt(2 * k * np.log(1 / delta_p))
    delta_0 = (delta_g - delta_p) / k
    return epsilon_0, delta_0


### TESTS
def test_basic():
    distances = [
        (0.2, 1e-6),
        (0.4, 1e-6),
        (0.7, 1e-6),
    ]

    print(composition_approxDP_static_hetero_basic(distances))


def test_advanced():
    epsilon_g = 1.0
    delta_g = 1e-9

    # find the per-query epsilon
    epsilon_0, delta_0 = solve_approxDP_static_homo_advanced(
        epsilon_g, delta_g, 5, delta_g
    )
    assert delta_0 == 0.0

    # how much epsilon do we use if we were to
    epsilon_gp, delta_gp = composition_approxDP_static_homo_advanced(
        epsilon_0, 0.0, 5, delta_g
    )

    print("per-query epsilon:", epsilon_0)
    print("total epsilon:    ", epsilon_gp)


def test_optimal():
    distances = [
        (0.2, 1e-6),
        (0.4, 1e-6),
        (0.7, 1e-6),
    ] * 200

    print(composition_approxDP_static_hetero_basic(distances))
    print(composition_approxDP_static_hetero_optimal_analytic(distances, 1e-8))


# test_optimal()


def test_renyiDP_find_alpha():
    """Find the ideal alpha for a given analysis."""
    from meas_gaussian import map_l2dist_gaussianmech_renyiDP
    from meas_cast import cast_renyiDP_approxDP_original_fix_delta

    def fix_params(sensitivity, scale, k, delta):
        def estimate_approxDP_epsilon(alpha):
            epsilon_0 = map_l2dist_gaussianmech_renyiDP(sensitivity, scale, alpha)
            epsilon_g = composition_renyiDP_static_hetero(alpha, [epsilon_0] * k)[1]
            return cast_renyiDP_approxDP_original_fix_delta(alpha, epsilon_g, delta)[0]

        return estimate_approxDP_epsilon

    # L2 sensitivity of the query
    sensitivity = 0.3

    # gaussian noise scale
    scale = 3.0

    # number of releases
    k = 30

    estimate_approxDP_epsilon = fix_params(sensitivity, scale, k, 1e-6)

    alphas = list(range(2, 101))
    distances = list(map(estimate_approxDP_epsilon, alphas))

    # print(alphas)
    # print(distances)

    print("ideal alpha:          ", alphas[np.argmin(distances)])
    print("corresponding epsilon:", np.min(distances))

    import matplotlib.pyplot as plt

    plt.plot(alphas, distances)
    plt.show()


# test_renyiDP_find_alpha()
