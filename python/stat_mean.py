import numpy as np
from stat_quantile import release_dp_quantile_via_ce


def release_dp_mean(x: np.ndarray, bounds, epsilon):
    x = np.clip(x, *bounds)
    return np.random.laplace(x.mean(), (bounds[1] - bounds[0]) / len(x) / epsilon)


def release_dp_trimmed_mean(x, percentile, bounds, epsilon):
    """Release a DP estimate of the trimmed mean
    where the percentile/2 smallest and 1-percentile/2 largest data points are ignored.

    An unbiased estimator for the mean, assuming the data is in bounds and symmetrically distributed.

    Assumes x ~ x' by hamming or edit distance, and that len(x) is public information
    """
    lower, upper = bounds
    alpha = percentile / 100

    # TRANSFORMATIONS
    # 1: clip (1-stable) (in this problem, this should be a no-op)
    x = np.clip(x, *bounds)

    # 2: cast (1-stable)
    x = x.astype(float)

    # 3: trim (1-stable)
    x = np.sort(x)
    min_idx = int(np.round(len(x) * (alpha / 2)))
    max_idx = len(x) - min_idx
    x_trimmed = x[min_idx:max_idx]

    # 4: aggregate (just a dp sum!)
    exact_aggregate = np.sum(x_trimmed)
    sensitivity = (upper - lower)

    # 5: lipschitz transform
    lipschitz_constant = 1 / ((1 - alpha) * len(x))
    exact_aggregate *= lipschitz_constant
    sensitivity *= lipschitz_constant

    # MEASUREMENT
    # 6: standard application of laplace mechanism
    scale = sensitivity / epsilon
    return np.random.laplace(exact_aggregate, scale)


def release_dp_percentile_mean(x, percentile, bounds, epsilon):
    """Release a DP estimate of the trimmed mean with `percentile` data removed, half from each tail.

    Assumes x ~ x' by hamming or edit distance, and that len(x) is public information.
    """
    lower, upper = bounds
    alpha = percentile / 100

    # TRANSFORMATIONS
    # 1: clip (1-stable) (in this problem, this should be a no-op)
    x = np.clip(x, *bounds)

    # 2: cast (1-stable)
    x = x.astype(float)

    # 3: filter by predicates (1-stable)
    lower = release_dp_quantile_via_ce(x, alpha / 2, bounds, epsilon / 3)
    upper = release_dp_quantile_via_ce(x, 1 - alpha / 2, bounds, epsilon / 3)
    x_filtered = x[(lower <= x) & (x <= upper)]

    # 4: aggregate (just a dp sum!)
    exact_aggregate = np.sum(x_filtered)
    sensitivity = (upper - lower)

    # 5: lipschitz transform
    lipschitz_constant = 1 / ((1 - alpha) * len(x))
    exact_aggregate *= lipschitz_constant
    sensitivity *= lipschitz_constant

    # MEASUREMENT
    # 6: standard application of laplace mechanism
    scale = sensitivity / (epsilon / 3)
    return np.random.laplace(exact_aggregate, scale), lower, upper


def release_dp_winsorized_mean(x, percentile, bounds, epsilon):
    """Release a DP estimate of the winsorized mean where `percentile` data is clamped, half from each tail.

    Assumes x ~ x' by hamming or edit distance, and that len(x) is public information.
    """
    lower, upper = bounds
    alpha = percentile / 100

    # TRANSFORMATIONS
    # 1: cast (1-stable)
    x = x.astype(float)

    # 2: clamp (1-stable)
    # reassigns over bounds
    lower = release_dp_quantile_via_ce(x, alpha / 2, bounds, epsilon / 3)
    upper = release_dp_quantile_via_ce(x, 1 - alpha / 2, bounds, epsilon / 3)
    x = np.clip(x, lower, upper)

    # 3: aggregate (just a dp sum!)
    exact_aggregate = np.sum(x) / len(x)
    sensitivity = (upper - lower) / len(x)

    # MEASUREMENT
    # 6: standard application of laplace mechanism
    scale = sensitivity / (epsilon / 3)
    return np.random.laplace(exact_aggregate, scale), lower, upper

