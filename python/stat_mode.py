from meas_exponential import *


def score_mode_discrete(x, candidates):
    """Scores a candidate based on likelihood of being the mode."""

    # Count the number of entries in x that match each candidate
    return (x[None] == candidates[:, None]).sum(axis=1)


def release_dp_mode_via_de(x, candidates, epsilon, neighboring):
    """Release the dp mode via the Discrete Exponential mechanism"""
    # sensitivity is 1 regardless of if input metric is Hamming or Symmetric

    return mechanism_exponential_discrete(x, candidates, epsilon,
        scorer=score_mode_discrete, 
        sensitivity=1,
        monotonic={
            'symmetric': True,
            'hamming': False
        }[neighboring])


def test_release_dp_mode_via_de():
    data = np.array([4] * 4 + [90] * 3 + list(range(100)))
    candidates = np.array(list(range(100)))

    print(release_dp_mode_via_de(data, candidates, 1., "symmetric"))

# test_release_dp_mode_via_de()
