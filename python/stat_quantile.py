from opendp.trans import make_count_by_categories
from opendp.meas import make_base_geometric

from opendp.mod import enable_features

import numpy as np
from scipy.stats import rankdata

from meas_exponential import *


enable_features("contrib")


# QUANTILE BY HISTOGRAMS
def postprocess_quantile(categories, counts, alphas: np.ndarray):
    """Postprocess a histogram release into a quantile estimate.

    :param categories: ordered data of length t
    :param counts: estimates of the counts of `categories`, of length t
    :param alphas: vector, each is the proportion of entries to the left, ranging from [0, 1]
    :return the category corresponding to the q-quantile of counts"""
    # approximate the cdf via `counts`
    cdf = np.cumsum(counts).astype(np.float)
    cdf /= cdf[-1]

    alphas = np.atleast_1d(alphas)[:, None]

    return categories[np.argmax(cdf >= alphas, axis=1)]


# MEDIAN BY DISCRETE EXPONENTIAL
def score_median_discrete(x, candidates):
    """Scores a candidate based on proximity to the median."""
    return np.array([
        min(sum(x <= candidate), sum(x >= candidate))
        for candidate in candidates
    ])


def release_dp_median_via_de(x, candidates, epsilon):
    """Release the dp median via the Discrete Exponential mechanism"""
    # sensitivity is 1 regardless of if input metric is Hamming or Symmetric
    return exponential_mechanism_discrete(x, candidates, epsilon,
        scorer=score_median_discrete, 
        sensitivity=1)


# MEDIAN BY CONTINUOUS EXPONENTIAL
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
    return np.minimum(rankdata(x[:-1], method="max"), rankdata(-x[1:], method="max"))


def release_dp_median_via_ce(x, bounds, epsilon):
    """Release the DP median via the Continuous Exponential mechanism.
    See Algorithm 2 in http://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf"""
    # sensitivity is 1 regardless of if input metric is Hamming or Symmetric
    return exponential_mechanism_1d(x, bounds, epsilon,
        scorer=score_median_continuous, 
        sensitivity=1)


# QUANTILE BY DISCRETE EXPONENTIAL
def score_quantile_discrete(x, candidates, alpha):
    """Assuming `x` is sorted, scores every element in `x` 
    according to rank distance from the `alpha`-quantile."""
    num_leq = (x[None] <= candidates[:, None]).sum(axis=1)
    return -np.abs(num_leq - alpha * len(x))


def release_dp_quantile_via_de(x, alpha, bounds, epsilon):
    """Release the DP median via the Discrete Exponential mechanism.
    See Algorithm 2 in http://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf"""
    return exponential_mechanism_discrete(x, bounds, epsilon,
        scorer=lambda x, cands: score_quantile_discrete(x, cands, alpha=alpha), 
        sensitivity=1/2)


# QUANTILE BY CONTINUOUS EXPONENTIAL
def score_quantile_continuous(x, alpha):
    """Assuming `x` is sorted, scores every element in `x` 
    according to rank distance from the `alpha`-quantile."""
    ranks = np.arange(len(x) - 1)
    return -np.minimum(np.abs(ranks - alpha * len(x)), np.abs(ranks + 1 - alpha * len(x)))


def release_dp_quantile_via_ce(x, alpha, bounds, epsilon):
    """Release the DP `alpha`-quantile via the Continuous Exponential mechanism.
    See Algorithm 2 in http://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf"""
    # sensitivity is 1 regardless of if input metric is Hamming or Symmetric
    return exponential_mechanism_1d(x, bounds, epsilon,
        scorer=lambda x: score_quantile_continuous(x, alpha=alpha), 
        sensitivity=1/2)


# QUANTILE BY BINARY SEARCH ON RANGE QUERIES
def release_noisy_range_count_zCDP(x, subset_range, rho):
    """Counts the number of values within `subset_range` and noises according to zCDP"""
    lower, upper = subset_range
    sensitivity = 1
    return np.random.normal(
        loc=sum(np.bitwise_and(lower <= x, x <= upper)), 
        scale=sensitivity / np.sqrt(2 * rho))


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


# SIMULTANEOUS QUANTILE RELEASE VIA RECURSIVE SPLITTING
def release_dp_approximate_quantiles_(x, alphas, bounds, epsilon):
    if len(alphas) == 0:
        return []
    
    if len(alphas) == 1:
        return [release_dp_quantile_via_ce(x, alphas[0], bounds, epsilon)]
    
    mid = (len(alphas) + 1) // 2
    p = alphas[mid]
    
    v = release_dp_quantile_via_ce(x, p, bounds, epsilon)
    x_l, x_u = x[x < v], x[x > v]

    alphas_l, alphas_u = alphas[:mid] / p, (alphas[mid + 1:] - p) / (1 - p)
    return [
        *release_dp_approximate_quantiles_(x_l, alphas_l, (bounds[0], v), epsilon),
        v,
        *release_dp_approximate_quantiles_(x_u, alphas_u, (v, bounds[1]), epsilon),
    ]


def release_dp_approximate_quantiles(x, alphas, bounds, epsilon):
    """Release a collection of `alphas`-quantiles via the Continuous Exponential mechanism.
    Takes advantage of information gained in prior quantile estimates.
    See Algorithm 1: https://arxiv.org/pdf/2110.05429.pdf
    """
    return release_dp_approximate_quantiles_(x, alphas, bounds, epsilon / (np.log2(len(alphas)) + 1))


# TESTS
def test_postprocess_quantile():
    categories = np.arange(100)
    histogrammer = make_count_by_categories(categories) >> make_base_geometric(1.0, D="VectorDomain<AllDomain<i32>>")

    data = np.random.randint(100, size=1000)
    dp_histogram_release = histogrammer(data)
    quantiles = postprocess_quantile(categories, dp_histogram_release[:-1], alphas=[.3, .7])
    print("quantiles", quantiles)


def test_medians():
    epsilon = 1.
    data = np.random.uniform(size=100)
    cands = np.linspace(0., 1., num=101)
    bounds = (0., 1.)

    from meas_cast import cast_pureDP_zCDP
    rho = cast_pureDP_zCDP(1.)[1]

    print("ground truth:", np.median(data))
    print("median de:   ", release_dp_median_via_de(data, cands, epsilon))
    print("median ce:   ", release_dp_median_via_ce(data, bounds, epsilon))
    print("quantile de: ", release_dp_quantile_via_de(data, 0.5, cands, epsilon))
    print("quantile ce: ", release_dp_quantile_via_ce(data, 0.5, bounds, epsilon))
    print("quantile bs: ", release_dp_quantile_via_bs((data * 100).astype(int), 0.5, (0, 100), rho) / 100)


def test_release_dp_approximate_quantiles():
    print(release_dp_approximate_quantiles(
        x=np.random.uniform(size=1000), 
        alphas=np.array([0.2, 0.3, 0.5, 0.7, 0.84]), 
        bounds=(0., 1.), 
        epsilon=1.))


# test_postprocess_quantile()
# test_medians()
# test_release_dp_approximate_quantiles()
