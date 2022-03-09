import numpy as np


def bernoulli(p, size=None):
    return np.random.uniform(size=size) < p


def discrete_laplace(loc, scale):
    if not scale:
        return loc
    alpha = np.exp(-1 / scale)

    noise = 0 if bernoulli(p=(1 - alpha) / (1 + alpha)) else np.random.geometric(1 - alpha)
    if bernoulli(p=0.5):
        noise *= -1
    return loc + noise


def cond_laplace(shift, scale):
    """Conditionally sample from laplace or discrete laplace depending on the dtype of `shift`"""
    if np.issubdtype(type(shift), np.integer):
        return discrete_laplace(shift, scale)
    if np.issubdtype(type(shift), np.float):
        return np.random.laplace(shift, scale)
    else:
        raise ValueError(f"unrecognized type {type(shift)}")
    

def cond_uniform(low, high):
    """Conditionally sample from discrete or continuous uniform based on the dtype of `low`"""
    if np.issubdtype(type(low), np.integer):
        return low if low == high else np.random.randint(low, high)
    if np.issubdtype(type(low), np.float):
        return np.random.uniform(low, high)
    else:
        raise ValueError(f"unrecognized type {type(low)}")
