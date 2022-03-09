import numpy as np


def estimate_bias_randomized_response_bool(prior, p):
    """estimates the bias of randomized response 
    when the probability of returning the true answer is `p`,
    and the likelihood that each answer is given is held in `priors`.

    For example, say you have a prior that your survey question will be answered "yes" 90% of the time.
    You run randomized response with p = 0.5.
    Then on average, the randomized responses will exhibit a bias of -0.2.
    As in, randomized responses will be False 2% more often than in the real data.
    
    :returns the bias of the randomized response"""
    assert 0 <= prior <= 1
    assert 0 <= p <= 1

    expectation = p * prior + (1 - p) / 2
    return expectation - prior


def debias_randomized_response_bool(release, p):
    """Adjust the mean release to remove bias."""
    assert 0 <= release <= 1
    assert 0 <= p <= 1

    return (1 - p - 2 * release) / (2 * (p - 1))


def estimate_bias_randomized_response(priors, p):
    """estimates the bias of randomized response 
    when the probability of returning the true answer is `p`,
    and the likelihood that each answer is given is held in `priors`.

    For example, say you have a multiple choice question with three levels.
    You also have a prior that the answers will be distributed [0.2, 0.7, 0.1].
    You run randomized response with p = 0.7 (the correct answer is returned 70% of the time).
    Then on average, randomized responses will exhibit a bias of [0.04, -.11, .07].
    As in, randomized responses of...
        level 1 will be 4% more frequent than the true values,
        level 2 will be 11% less frequent than the true values,
        level 3 will be 7% more frequent than the true values
    """
    priors = np.array(priors)
    assert all(priors >= 0) and abs(sum(priors) - 1) < 1e-6
    assert 0 <= p <= 1

    expectation = p * priors + (1 - p) / len(priors)
    return expectation - priors


def debias_randomized_response(releases, p):
    """Adjust the mean release to remove bias."""
    releases = np.array(releases)
    assert all(releases >= 0) and abs(sum(releases) - 1) < 1e-6
    assert 0 <= p <= 1

    return (1 - p - len(releases) * releases) / (len(releases) * (p - 1))


def test_debias_rr():
    # Say we collect 12 T in a survey of 20. RR is run with p=.5.
    # Then we would expect 14 individuals to have answered T truthfully
    assert debias_randomized_response_bool(12 / 20, .5) * 20 == 14

    release = [.2, .3, .5]
    print("Frequencies in release are:", release)

    debiased = debias_randomized_response(release, .5)
    print("To see these kinds of results, the actual frequencies are:", debiased)

    bias = estimate_bias_randomized_response(debiased, .5)
    print("We have an expected bias of:", bias)

    # The release - bias should be equivalent to the debiased answer
    assert sum(release - bias - debiased) < 1e-6
    

# test_debias_rr()
