import numpy as np


def mechanism_sparse_vector(x, sensitivity, queries, threshold, k, epsilon_svt, epsilon_query=None):
    """Returns up to the first k indexes for which the query is above the threshold.
    Described in Section 4.1: https://arxiv.org/pdf/1603.01699.pdf

    :param x: dataset for which to release queries on
    :param sensitivity: sensitivity of the queries on the dataset
    :param queries: a streaming iterable of queries mapping from x -> R
    :param threshold: query answers must roughly be greater than this value to be released
    :param k: the maximum number of releases to make
    :param epsilon_svt: amount of epsilon to spend on determining which queries are to be released
    :param epsilon_query: optional, if specified, then DP estimates of the query are also returned.
    :returns a generator. Iterating over the generator yields dp releases from queries
    """
    assert k > 0

    # choose the optimal allocation of epsilons amongst svt releases
    # https://arxiv.org/pdf/1603.01699.pdf#subsection.4.2
    weights = np.array([1, (2 * k) ** (2 / 3)])
    weights /= weights.sum()
    eps_svt_1, eps_svt_2 = epsilon_svt * weights

    rho = np.random.laplace(sensitivity / eps_svt_1)

    count = 0
    for q, thresh in zip(queries, threshold):
        # when we release, decrement k, return, and reset the threshold
        exact = q(x)
        if exact + np.random.laplace(2 * k * sensitivity / eps_svt_2) >= thresh + rho:
            
            response = True
            if epsilon_query:
                response = True, exact + np.random.laplace(k * sensitivity / epsilon_query)
            
            count += 1
            if count == k:
                return response
            yield response

        else:
            yield False, None if epsilon_query else False
