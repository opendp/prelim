import numpy as np

from opendp.trans import make_sized_bounded_mean, make_clamp, make_bounded_resize
from opendp.meas import make_base_laplace
from opendp.mod import binary_search_chain, enable_features

enable_features("contrib", "floating-point")


# MOMENTS
def release_dp_standard_moment(x, k, mu, sigma, bounds, epsilon):
    """A simple transformation to release the dp standard moment of order `k` on `x`.
    Relatively low utility, because increasing k makes results very sensitive to perturbation.
    DP estimates of mu and sigma can also significantly increase the variance of this estimator.
    Assumes dataset size len(`x`) is public.
    """
    def moment_trans(v):
        return ((v - mu) / sigma) ** k

    lower, upper = bounds
    x, *bounds = moment_trans(x), moment_trans(lower), moment_trans(upper)

    # build the estimator
    
    bounded_mean = (
        make_clamp(bounds) >> 
        make_bounded_resize(len(x), bounds, mu) >>
        make_sized_bounded_mean(len(x), bounds)
    )
    dp_mean_estimator = binary_search_chain(lambda s: bounded_mean >> make_base_laplace(s), 1, epsilon)

    # release
    return dp_mean_estimator(x)


def release_dp_central_moment(x, k, mu, bounds, epsilon):
    """A simple transformation to release the dp central moment of order `k` on `x`.
    Relatively low utility at larger k, because increasing k makes results very sensitive to perturbation.
    A DP estimate of mu can also significantly increase the variance of this estimator.
    """
    return release_dp_standard_moment(x, k, mu, 1, bounds, epsilon)


def release_dp_raw_moment(x, k, bounds, epsilon):
    """A simple transformation to release the dp raw moment of order `k` on `x`.
    Relatively low utility at larger k, because increasing k makes results very sensitive to perturbation.
    """
    return release_dp_standard_moment(x, k, 0, 1, bounds, epsilon)


def release_dp_skewness(x, mu, sigma, bounds, epsilon):
    """A simple transformation to release the dp skewness.
    Relatively low utility, because skewness is sensitive to perturbation
    """
    return release_dp_standard_moment(x, 3, mu, sigma, bounds, epsilon)


def release_dp_kurtosis(x, mu, sigma, bounds, epsilon):
    """A simple transformation to release the dp kurtosis.
    Relatively low utility, because kurtosis is sensitive to perturbation
    """
    return release_dp_standard_moment(x, 4, mu, sigma, bounds, epsilon)


def release_dp_variance(x, mu, bounds, epsilon):
    """A simple transformation to release the dp variance."""
    return release_dp_central_moment(x, 2, mu, bounds, epsilon)


def release_dp_mean(x, bounds, epsilon):
    """A simple transformation to release the dp mean."""
    return release_dp_raw_moment(x, 1, bounds, epsilon)


