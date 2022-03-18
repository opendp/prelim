import numpy as np
from utils.samplers import *


def mechanism_exponential_discrete(x, candidates, epsilon, scorer, sensitivity, monotonic=False):
    """Return an `epsilon`-DP sample from the set of discrete `candidates`.
    The sampling probabilities is constructed by running `scorer` on `x` for each candidate.

    :param x: 1d dataset for which to release the estimate
    :param bounds: lower and upper bounds for `x`
    :param epsilon: privacy parameter
    :param scorer: Function that accepts data and returns len(data) - 1 bin scores
    :param sensitivity: the greatest that `scorer` can change when perturbing one individual
    :param monotonic: boolean, set to true if the scorer is a monotonic function over x
    :returns a sample from `bounds` with probability proportional to the `scorer`
    """

    # score each candidate (can be more computationally efficient to score all at once)
    scores = scorer(x, candidates)

    # for numerical stability; omitting this line results in the same probabilities
    scores -= scores.max()

    # compute likelihood of selecting each candidate
    sensitivity *= 1 if monotonic else 2
    likelihoods = np.exp(epsilon * scores / sensitivity)

    # normalize to a probability
    probabilities = likelihoods / likelihoods.sum()

    # sample one index wrt the selection probabilities
    cdf = probabilities.cumsum()
    index = np.argmax(cdf >= np.random.uniform())

    # return the respective candidate
    return candidates[index]


def mechanism_exponential_1d(x, bounds, epsilon, scorer, sensitivity, monotonic=False):
    """Return an `epsilon`-DP sample from the continuous 1-d distribution over `bounds`.
    The distribution is constructed by running `scorer` on `x`.

    :param x: 1d dataset for which to release the estimate
    :param bounds: lower and upper bounds for `x`
    :param epsilon: privacy parameter
    :param scorer: Function that accepts data and returns len(data) - 1 bin scores
    :param sensitivity: the greatest that `scorer` can change when perturbing one individual
    :param monotonic: boolean, set to true if the scorer is a monotonic function over x
    :returns a sample from `bounds` with probability proportional to the `scorer`
    """
    lower, upper = bounds

    # sort, clip and bookend x with bounds
    x = np.concatenate(([lower], np.clip(np.sort(x), *bounds), [upper]))

    # score all intervals in x
    scores = scorer(x)

    # for numerical stability; omitting this line results in the same probabilities
    scores -= scores.max()

    # compute likelihood of selecting each interval
    sensitivity *= 1 if monotonic else 2
    likelihoods = np.diff(x) * np.exp(scores * epsilon / sensitivity)

    # normalize to a probability
    probabilities = likelihoods / likelihoods.sum()

    # select one interval wrt the selection probabilities
    cdf = probabilities.cumsum()
    index = np.argmax(cdf >= np.random.uniform())

    # sample uniformly from the selected interval
    return cond_uniform(low=x[index], high=x[index + 1])


# GUMBEL VERSIONS
# Using this trick to get the same outcome:
# https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/

def mechanism_exponential_discrete_gumbel(x, candidates, epsilon, scorer, sensitivity, monotonic=False):
    """Return an `epsilon`-DP sample from the set of discrete `candidates`.
    The sampling probabilities is constructed by running `scorer` on `x` for each candidate.

    :param x: 1d dataset for which to release the estimate
    :param bounds: lower and upper bounds for `x`
    :param epsilon: privacy parameter
    :param scorer: Function that accepts data and returns len(data) - 1 bin scores
    :param sensitivity: the greatest that `scorer` can change when perturbing one individual
    :param monotonic: boolean, set to true if the scorer is a monotonic function over x
    :returns a sample from `bounds` with probability proportional to the `scorer`
    """

    # score each candidate (can be more computationally efficient to score all at once)
    scores = scorer(x, candidates)

    # for numerical stability; omitting this line results in the same probabilities
    scores -= scores.max()

    # compute likelihood of selecting each candidate
    sensitivity *= 1 if monotonic else 2
    log_likelihoods = epsilon * scores / sensitivity

    # add gumbel noise in log space (when hardening, don't forget to reject Uniform = 1)
    log_likelihoods -= np.log(-np.log(np.random.uniform(size=log_likelihoods.shape)))

    # equivalent approach 
    # log_likelihoods = np.random.gumbel(log_likelihoods, size=log_likelihoods.shape)

    # the value with the largest noisy score is the selected bin index
    index = np.argmax(log_likelihoods)

    # return the respective candidate
    return candidates[index]



def mechanism_exponential_1d_gumbel(x, bounds, epsilon, scorer, sensitivity, monotonic=False):
    """Return an `epsilon`-DP sample from the continuous 1-d distribution over `bounds`.
    The distribution is constructed by running `scorer` on `x`.

    :param x: 1d dataset for which to release the estimate
    :param bounds: lower and upper bounds for `x`
    :param epsilon: privacy parameter
    :param scorer: Function that accepts data and returns len(data) - 1 bin scores
    :param sensitivity: the greatest that `scorer` can change when perturbing one individual
    :param monotonic: boolean, set to true if the scorer is a monotonic function over x
    :returns a sample from `bounds` with probability proportional to the `scorer`
    """
    lower, upper = bounds

    # sort, clip and bookend x with bounds
    x = np.concatenate(([lower], np.clip(np.sort(x), *bounds), [upper]))

    # score all intervals in x
    scores = scorer(x)

    # compute likelihood of selecting each interval
    sensitivity *= 1 if monotonic else 2

    # compute likelihoods in log space
    log_likelihoods = np.log(np.diff(x)) + scores * epsilon / sensitivity

    # add gumbel noise in log space (when hardening, don't forget to reject Uniform = 1)
    log_likelihoods -= np.log(-np.log(np.random.uniform(size=log_likelihoods.shape)))

    # equivalent approach 
    # log_likelihoods = np.random.gumbel(log_likelihoods, size=log_likelihoods.shape)

    # the value with the largest noisy score is the selected bin index
    index = np.argmax(log_likelihoods)

    # sample uniformly from the selected interval
    return cond_uniform(low=x[index], high=x[index + 1])
