# measures of dispersion
import numpy as np

from stat_quantile import release_dp_median_via_ce
from opendp.trans import make_sized_bounded_mean, make_clamp, make_bounded_resize
from opendp.meas import make_base_laplace
from opendp.mod import binary_search_chain, enable_features

enable_features("contrib", "floating-point")


# see also the OpenDP variance estimator, and estimating variance from a DP mean in moments.py


def release_dp_median_absolute_deviation(x, median, bounds, epsilon):
    """A simple transformation to release the dp median absolute deviation."""
    # shift the bounds
    lower, upper = bounds
    error_bounds = (lower - median, upper - median)

    return release_dp_median_via_ce(x - median, error_bounds, epsilon)


def release_dp_average_absolute_deviation(x, mu, bounds, epsilon):
    """A simple transformation to release the dp mean absolute deviation.
    Assumes dataset size len(`x`) is public.
    """

    # shift the bounds
    lower, upper = bounds
    error_bounds = (lower - mu, upper - mu)

    # build the estimator
    bounded_mean = (
        make_clamp(bounds) >> 
        make_bounded_resize(len(x), bounds, mu) >>
        make_sized_bounded_mean(len(x), error_bounds)
    )
    dp_mean_estimator = binary_search_chain(lambda s: bounded_mean >> make_base_laplace(s), 1, epsilon)

    # release
    return dp_mean_estimator(np.abs(x - mu))
