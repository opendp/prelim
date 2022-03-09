import numpy as np


### NOISE SCALE
# map_[input metric]_[mechanism]_[output measure]

def map_l2dist_gaussianmech_zCDP(sensitivity, scale):
    """map an L2 distance `sensitivity` through the gaussian mechanism with parameter `scale` to rho-zCDP
    Lemma 2.4 https://arxiv.org/pdf/1605.02065.pdf#subsubsection.1.2.2
    
    :param sensitivity: maximum L2 distance perturbation of a query
    :param scale: standard deviation of gaussian noise
    :returns rho"""
    return (sensitivity / scale) ** 2 / 2


def map_l2dist_gaussianmech_renyiDP(sensitivity, scale, alpha):
    """map an L2 distance `sensitivity` through the gaussian mechanism with parameter `scale` to (alpha, epsilon)-RDP
    Proposition 7 and Corollary 3: https://arxiv.org/pdf/1702.07476.pdf#subsection.6.3

    :param sensitivity: maximum L2 distance perturbation of a query
    :param scale: standard deviation of gaussian noise
    :param alpha: order of renyi divergence > 1
    :returns epsilon
    """
    return alpha * (sensitivity / scale) ** 2 / 2


def map_l2dist_sampledgaussianmech_renyiDP_poisson_analytic(sensitivity, scale, alpha, q):
    """map an L2 distance `sensitivity` through the poisson-sampled gaussian mechanism with parameter `scale` to (alpha, epsilon)-RDP
    A loose, analytic bound from Table 1 row 3: https://arxiv.org/pdf/1908.10530.pdf

    :param sensitivity: maximum L2 distance perturbation of a query
    :param scale: standard deviation of gaussian noise
    :param alpha: order of renyi divergence > 1
    :param q: sampling rate
    :returns epsilon
    """
    assert q <= .2
    assert scale >= 4

    L = np.ln(1 + 1 / (q * (alpha - 1)))
    T1 = scale ** 2 * L / 2 - 2 * np.ln(scale)
    T2 = ((scale * L) ** 2 / 2 - 5 * np.ln(5) - 2 * np.ln(scale)) / (L + np.ln(q * alpha) + 1 / (2 * scale ** 2))
    assert alpha <= min(T1, T2)

    return 4 * q ** 2 * map_l2dist_gaussianmech_renyiDP(sensitivity, scale, alpha)


def map_l2dist_sampledgaussianmech_renyiDP_poisson(sensitivity, scale, alpha, q):
    """map an L2 distance `sensitivity` through the poisson-sampled gaussian mechanism with parameter `scale` to (alpha, epsilon)-RDP
    From the numerically stable computation: https://arxiv.org/pdf/1908.10530.pdf#subsection.3.3
    Implementation from Opacus/Tensorflow.

    We need ln(A_alpha) / (alpha - 1). Focus on computing ln(A_alpha).
    1. Binomial expansion
    2. Apply linearity of expectation
    3. Substitute analytical closed form for the expectation
    4. Distribute the ln through the binomial expansion, breaking up products into sums

    Similar process for fractional alpha, refer to paper.

    :param sensitivity: maximum L2 distance perturbation of a query
    :param scale: standard deviation of gaussian noise
    :param alpha: order of renyi divergence > 1
    :param q: sampling rate
    :returns epsilon
    """
    from utils.renyi_sgm import compute_rdp
    # Paper assumes sensitivity is 1. 
    # Consider a lipschitz transform `v / sensitivity` such that resulting sensitivity is 1. 
    # Apply mechanism, then postprocess `v' * sensitivity`. Effective noise scale is `scale * sensitivity`
    return compute_rdp(q=q, noise_multiplier=scale * sensitivity, steps=1, orders=alpha)


def map_l2dist_sampledgaussianmech_renyiDP_analytic_simple(sensitivity, scale, alpha, m, n):
    """map an L2 distance `sensitivity` through the simple-sampled gaussian mechanism with parameter `scale` to (alpha, epsilon)-RDP
    A loose, analytic bound from Table 1 row 4: https://arxiv.org/pdf/1908.10530.pdf

    :param sensitivity: maximum L2 distance perturbation of a query
    :param scale: standard deviation of gaussian noise
    :param alpha: order of renyi divergence > 1
    :param m: number of samples to take
    :param n: dataset size
    :returns epsilon
    """
    s = m / n

    assert s <= .1
    assert scale >= np.sqrt(5)
    assert alpha <= scale ** 2 * np.ln(1 / s) / 2

    return 12 * s ** 2 * map_l2dist_gaussianmech_renyiDP(sensitivity, scale, alpha)
