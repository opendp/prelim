import numpy as np

### CAST
# cast_[input measure]_[output measure]

def cast_pureDP_approxDP(epsilon):
    return epsilon, 0


def cast_pureDP_zCDP(epsilon):
    """cast a epsilon-DP measurement to a (xi, rho)-zCDP measurement
    Proposition 1.4: https://arxiv.org/pdf/1605.02065.pdf#subsubsection.1.2.1
    """
    return 0, epsilon ** 2 / 2
    # also valid under Lemma 8.3
    # return epsilon, 0


def cast_approxDP_approxzCDP(distance):
    """cast a (epsilon, delta)-DP measurement to a delta-approximate (xi, rho)-zCDP measurement
    Proposition 1.13: https://arxiv.org/pdf/1605.02065.pdf#subsubsection.1.2.6
    "for stability-based techniques"
    """
    epsilon, delta = distance
    return delta, 0, cast_pureDP_zCDP(epsilon)


def cast_approxDP_zCDP_fix_xi(distance, xi):
    """cast a (epsilon, delta)-DP measurement to a (xi, rho)-zCDP measurement where xi is fixed
    Lemma 3.5 https://arxiv.org/pdf/1605.02065.pdf#subsection.3.2
    """
    epsilon, delta = distance
    return xi, (np.sqrt(epsilon - xi + np.log(1 / delta)) - np.sqrt(np.log(1 / delta))) ** 2


def cast_approxDP_zCDP_fix_rho(distance, rho):
    """cast a (epsilon, delta)-DP measurement to a (rho, xi)-zCDP measurement where rho is fixed
    Lemma 3.5 https://arxiv.org/pdf/1605.02065.pdf#subsection.3.2
    """
    epsilon, delta = distance
    return epsilon - rho - np.sqrt(4 * rho * np.log(1 / delta)), rho


def cast_zCDP_approxDP_fix_delta(distance, delta):
    """cast a (xi, rho)-zCDP measurement to a (epsilon, delta)-DP measurement where delta is fixed
    Lemma 3.6: https://arxiv.org/pdf/1605.02065.pdf#subsection.3.2
    Looser version in Proposition 1.3
    """
    xi, rho = distance
    return xi + rho + np.sqrt(4 * rho * np.log(1 / delta)), delta


def cast_zCDP_approxDP_fix_epsilon(distance, epsilon):
    """cast a (xi, rho)-zCDP measurement to a (epsilon, delta)-DP measurement where epsilon is fixed
    Lemma 3.6: https://arxiv.org/pdf/1605.02065.pdf#subsection.3.2
    """
    xi, rho = distance
    a = (epsilon - xi - rho) / (2 * rho)

    # the third option for least dominates
    # least = np.sqrt(np.pi * rho)
    # least = 1 / (1 + a)
    least = 2 / (1 + a + np.sqrt((1 + a) ** 2 + 4 / (np.pi * rho)))

    return epsilon, np.exp(-a ** 2 * rho) * least


def cast_renyiDP_approxDP_original_fix_delta(alpha, epsilon, delta):
    """cast a (alpha, epsilon)-RDP measurement to a (epsilon, delta)-DP measurement
    Proposition 3: https://arxiv.org/pdf/1702.07476.pdf#section.4
    """
    return epsilon + np.log(1 / delta) / (alpha - 1), delta


def cast_renyiDP_approxDP_fix_delta(alpha, epsilon, delta):
    """cast a (alpha, epsilon)-RDP measurement to a (epsilon, delta)-DP measurement

    Theorem 21: https://arxiv.org/pdf/1905.09982.pdf

    `alpha` and `epsilon` may each be length-n vectors. 
    The choice of alpha that minimizes the resulting epsilon is returned.
    """

    # Implementation from paper:
    # return epsilon + np.log((alpha - 1) / alpha) - (np.log(delta) + np.log(alpha)) / (alpha - 1), delta

    # Implementation from Opacus/Tensorflow:
    # with extra bounds checking and alpha selection
    from utils.renyi_sgm import get_privacy_spent
    return get_privacy_spent(alpha, epsilon, delta)

