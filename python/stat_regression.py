import numpy as np

from python.post_fractional_estimate import debias_reciprocal


def _sens_sum(a, b):
    return max(abs(a), abs(b))


def release_sxy(data, bounds, epsilon):
    """Releases an `epsilon`-DP estimate of the dot product of the two columns in `data`, x and y."""
    x, y = data.T

    x_bounds, y_bounds = bounds
    sxy_sensitivity = 2 * _sens_sum(*x_bounds) * _sens_sum(*y_bounds)

    return np.random.laplace(
        loc=x @ y, 
        scale=sxy_sensitivity / epsilon)


def release_sxx(x, bounds, epsilon):
    """Releases an `epsilon`-DP estimate of the sum of squares of x."""
    sxx_sensitivity = _sens_sum(*bounds) ** 2
    return np.random.laplace(
        loc=x @ x, 
        scale=sxx_sensitivity / epsilon)


def release_beta(data, bounds, epsilon, alpha=0.66, debias_recip=False):
    """Release an `epsilon`-DP estimate of the slope between the two columns in `data`.
    
    :param data: array of shape [n, 2]
    :param bounds: Upper and lower bounds for x and y. ((x_l, x_u), (y_l, y_u))
    :param epsilon: privacy parameter
    :param alpha: proportion of budget to allocate to numerator
    :param debias_recip: apply a transformation to make 1 / sxx unbiased
    :returns slope regression parameter
    """
    sxy = release_sxy(data, bounds, epsilon * alpha)
    sxx = release_sxx(data[:, 0], bounds[0], epsilon * (1 - alpha))

    if debias_recip:
        sxx = debias_reciprocal(
            sxx, _sens_sum(*bounds[0]) ** 2 / (epsilon * (1 - alpha)),
            np.random.laplace(size=10_000)
        )
    
    return sxy / sxx


