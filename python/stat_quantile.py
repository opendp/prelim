from opendp.trans import make_count_by_categories
from opendp.meas import make_base_geometric

from opendp.mod import enable_features

import numpy as np

from meas_exponential import *


enable_features("contrib")


# QUANTILE BY HISTOGRAMS
def postprocess_quantile(categories, counts, alphas: np.ndarray, interpolate=True):
    """Postprocess a histogram release into a quantile estimate.

    :param categories: ordered data of length t
    :param counts: estimates of the counts of `categories`, of length t
    :param alphas: vector, each is the proportion of entries to the left, ranging from [0, 1]
    :param interp: if True, interpolate amongst the bin labels
    :return the category corresponding to the q-quantile of counts"""
    # approximate the cdf via `counts`
    cdf = np.cumsum(counts).astype(np.float)
    cdf /= cdf[-1]

    alphas = np.atleast_1d(alphas)

    indices = np.argmax(cdf >= alphas[:, None], axis=1)

    if interpolate:
        interps = np.empty_like(alphas, dtype=float)
        for i, idx in enumerate(indices):
            interps[i] = np.interp(alphas[i], cdf[idx - 1: idx + 1], categories[idx - 1: idx + 1])
        return interps
    
    return categories[indices]


# MEDIAN BY DISCRETE EXPONENTIAL
def release_dp_median_via_de(x, candidates, epsilon):
    """Release the dp median via the Discrete Exponential mechanism"""

    def score_median_discrete(x, candidates):
        """Scores each candidate based on proximity to the median."""
        return np.minimum(np.arange(len(candidates)), np.arange(len(candidates))[::-1])

        # slower to compute, but more intuitive and considers equality of candidates
        # return np.array(
        #     [min(sum(x <= candidate), sum(x >= candidate)) for candidate in candidates]
        # )

    # sensitivity is 1 regardless of if input metric is Hamming or Symmetric
    return mechanism_exponential_discrete(
        x, candidates, epsilon, scorer=score_median_discrete, sensitivity=1
    )


# MEDIAN BY CONTINUOUS EXPONENTIAL
def release_dp_median_via_ce(x, bounds, epsilon):
    """Release the DP median via the Continuous Exponential mechanism.
    See Algorithm 2 in http://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf"""
    # sensitivity is 1 regardless of if input metric is Hamming or Symmetric

    def score_median_continuous(x):
        """Scores all entries in `x` based on proximity to the median. Assumes `x` is sorted."""
        # SLOW SOLUTION
        # edge_scores = np.array([min(sum(x <= candidate), sum(x >= candidate)) for candidate in x])

        # Bin scores are the minimum of left and right edge scores.
        # Hence, the bin score is the min of 4 numbers:
        #      a = sum(x_i     <= candidate), b = sum(x_i     >= candidate)
        #      c = sum(x_{i+1} <= candidate), d = sum(x_{i+1} >= candidate)

        # Since the rank is monotonic, b >= a and c >= d, so we only need to take the min of a and d:
        # bin_scores = np.minimum(edge_scores[1:], edge_scores[:-1])
        # return bin_scores

        # FAST SOLUTION
        # To score efficiently, find minimal ranks from left and right side.
        # Omitting the max rank does not affect the rankings,
        #    so we can similarly take advantage of monotonicity
        # from scipy.stats import rankdata
        # return np.minimum(rankdata(x[:-1], method="max"), rankdata(-x[1:], method="max"))

        # FAST FAST SOLUTION
        ranks = np.arange(len(x) - 1)
        return -np.minimum(
            np.abs(ranks - 0.5 * len(x)), np.abs(ranks + 1 - 0.5 * len(x))
        )

    return mechanism_exponential_1d(
        x, bounds, epsilon, scorer=score_median_continuous, sensitivity=1
    )


# QUANTILE BY DISCRETE EXPONENTIAL
def release_dp_quantile_via_de(x, alpha, bounds, epsilon, neighboring):
    """Release the DP median via the Discrete Exponential mechanism.
    See Algorithm 2 in http://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf"""

    def score_quantile_discrete(_x, candidates):
        """Assuming `x` is sorted, scores every element in `x`
        according to rank distance from the `alpha`-quantile."""
        return -np.abs(np.arange(len(candidates)) - alpha * len(candidates))

        # nearly equivalent, slower to compute, scores equivalent candidates the same
        # num_leq = (x[None] <= candidates[:, None]).sum(axis=1)
        # return -np.abs(num_leq - alpha * len(x))

    return mechanism_exponential_discrete(
        x,
        bounds,
        epsilon,
        scorer=score_quantile_discrete,
        sensitivity={"symmetric": max(alpha, 1 - alpha), "hamming": 1}[neighboring],
    )


# QUANTILE BY CONTINUOUS EXPONENTIAL
def release_dp_quantile_via_ce(x, alpha, bounds, epsilon, neighboring):
    """Release the DP `alpha`-quantile via the Continuous Exponential mechanism.
    See Algorithm 2 in http://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf
    """

    def score_quantile_continuous(x):
        """Assuming `x` is sorted, scores every element in `x`
        according to rank distance from the `alpha`-quantile."""
        ranks = np.arange(len(x) - 1)
        return -np.minimum(
            np.abs(ranks - alpha * len(x)), np.abs(ranks + 1 - alpha * len(x))
        )

    return mechanism_exponential_1d(
        x,
        bounds,
        epsilon,
        scorer=score_quantile_continuous,
        sensitivity={"symmetric": max(alpha, 1 - alpha), "hamming": 1}[neighboring],
    )


# QUANTILE BY BINARY SEARCH ON RANGE QUERIES
def release_noisy_range_count_zCDP(x, subset_range, rho):
    """Counts the number of values within `subset_range` and noises according to zCDP"""
    lower, upper = subset_range
    sensitivity = 1
    return np.random.normal(
        loc=sum(np.bitwise_and(lower <= x, x <= upper)),
        scale=sensitivity / np.sqrt(2 * rho),
    )


def release_dp_quantile_via_bs(x, alpha, bounds, rho):
    """Release the DP `alpha`-quantile via binary search with range counts on integers
    See Algorithm 1 in https://arxiv.org/pdf/2106.00463.pdf
    """
    m = alpha * len(x)
    left, right = bounds
    rho_0 = rho / np.log2(right - left)
    assert type(left) == int
    while left < right:
        mid = (left + right) // 2
        if release_noisy_range_count_zCDP(x, (left, mid), rho_0) <= m:
            left = mid + 1
        else:
            right = mid
    return (left + right) // 2


def dp_quantile_via_bs_rank_error(beta, bounds, rho):
    """The dp estimate differs by at most rank error (number of ranks the answer differs) with 100(1-beta)% confidence

    See Algorithm 1 in https://arxiv.org/pdf/2106.00463.pdf
    :param beta: significance level
    :returns rank error at the given hyperparameters
    """
    lower, upper = bounds
    log_u = np.log2(upper - lower)
    return np.sqrt(log_u * np.log(log_u / beta) / (2 * rho))


# TESTS
def test_postprocess_quantile():
    categories = np.arange(100)
    histogrammer = make_count_by_categories(categories) >> make_base_geometric(
        1.0, D="VectorDomain<AllDomain<i32>>"
    )

    data = np.random.randint(100, size=1000)
    dp_histogram_release = np.array(histogrammer(data))
    quantiles = postprocess_quantile(
        categories, dp_histogram_release[:-1], alphas=[0.3, 0.7]
    )
    print("quantiles", quantiles)


def test_medians():
    epsilon = 1.0
    data = np.random.uniform(size=100)
    cands = np.linspace(0.0, 1.0, num=101)
    bounds = (0.0, 1.0)

    from meas_cast import cast_pureDP_zCDP

    rho = cast_pureDP_zCDP(1.0)[1]
    bs_quantile = (
        release_dp_quantile_via_bs((data * 100).astype(int), 0.5, (0, 100), rho) / 100
    )

    print("ground truth:", np.median(data))
    print("median de:   ", release_dp_median_via_de(data, cands, epsilon))
    print("median ce:   ", release_dp_median_via_ce(data, bounds, epsilon))
    print(
        "quantile de: ",
        release_dp_quantile_via_de(data, 0.5, cands, epsilon, neighboring="symmetric"),
    )
    print(
        "quantile ce: ",
        release_dp_quantile_via_ce(data, 0.5, bounds, epsilon, neighboring="symmetric"),
    )
    print("quantile bs: ", bs_quantile)


# test_postprocess_quantile()
# test_medians()
