import numpy as np


def debias_reciprocal(shift, scale, dist=np.random.normal(size=10_000)):
    """Corrects for the bias induced by symmetric noise in a denominator of a fractional estimate.

    Consider a random variable X with mean mu.
    `shift` is an unbiased estimator for mu.
    X is approximated by the empirical distribution: `shift` + `scale` * `dist`

    PROBLEM
    By Jensen's inequality, 1 / E[X] < E[1 / X] because the reciprocal > 0 is convex.
    Then 1 / `shift` is a biased estimator for 1 / mu.

    SOLUTION
    Estimate the multiplicative bias:
        MB / E[X] = E[1 / X]
        MB = E[X] * E[1 / X]
        MB = `shift` * (1 / X).mean()
    Return `shift` * MB, an asymptotically unbiased estimate for 1 / mu
        `shift`^2 * (1 / X).mean()

    :param shift: differentially private estimate to be used in denominator
    :param scale: scale of symmetric noise that was added to `shift`
    :param dist: samples from the standard distribution of `shift`. Optional, defaults to normal.
                 The larger dist is, the more accurate the correction is.
    :returns a slightly larger value that avoids bias
    """
    # construct empirical distribution
    X = shift + scale * dist

    # apply correction factor to shift
    return shift**2 * (1 / X).mean()


def _tester(x, scale, dist):
    y = np.empty(len(x))
    for i, x_i in enumerate(x):
        y[i] = debias_reciprocal(x_i, scale, dist)

    print("average biased reciprocal:   ", 1 / (1 / x).mean())
    print("average corrected reciprocal:", 1 / (1 / y).mean())


def test_normal():
    # Slowly converges to an unbiased estimate after many simulations

    true_denom = 1_000
    scale = 80
    sims = 1_000_000

    x = np.random.normal(true_denom, scale, size=sims)
    empirical_dist = np.random.normal(size=1_000)

    print('normal')
    _tester(x, scale, empirical_dist)


def test_laplace():
    # Also converges, but is more sensitive to scale than the gaussian

    true_denom = 1_000
    scale = 50
    sims = 1_000_000

    x = np.random.laplace(true_denom, scale, size=sims)
    empirical_dist = np.random.laplace(size=1_000)

    print('laplace')
    _tester(x, scale, empirical_dist)


# test_normal()
# test_laplace()
