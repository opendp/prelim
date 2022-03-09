import numpy as np

# See relevant paper:
# https://projects.iq.harvard.edu/files/privacytools/files/bun_mark_composable_.pdf

# Compositor may be found here:
# from meas_composition import composition_tCDP_static_hetero

# Casting to approxDP may be found here:
# from meas_cast import cast_tCDP_approxDP_fix_delta


def map_l2dist_sinhnormalmech_tCDP(sensitivity, scale, A):
    """map via the sinhnormal mechanism an L2 distance `sensitivity` 
    with parameters `scale` and `A` to (rho, omega)-tCDP
    """
    rho = (sensitivity / scale) ** 2 / 2
    assert 1 < 1 / np.sqrt(rho) <= A / sensitivity

    return 16 * rho, A / (8 * sensitivity)


def mechanism_sinhhormal(x, scale, A):
    """Privatize an estimate `x` with the sinhnormal mechanism."""
    shape = x.shape if isinstance(x, np.ndarray) else None
    return x + A * np.arcsinh(np.random.normal(scale=scale, shape=shape) / A)



# amplify_[measure]_[sampler]
def amplify_tCDP_simple(distance, m, n):
    """amplify `distance` and return a new functional, smaller, `distance` 
    by sampling `m` from a population of size `n` uniformly without replacement
    Section 3: https://projects.iq.harvard.edu/files/privacytools/files/bun_mark_composable_.pdf#%5B%7B%22num%22%3A140%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22Fit%22%7D%5D
    """
    s = m / n
    rho, omega = distance
    assert 0 < s <= 0.1
    assert 0 < rho <= 0.1
    assert omega > 1 / (2 * rho)

    return 13 * s ** 2 * rho, np.log(1 / s) / (4 * rho)

