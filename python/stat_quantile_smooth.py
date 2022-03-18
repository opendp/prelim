import numpy as np

from utils.samplers import cond_laplace


def compute_median_smooth_sensitivity(x, bounds, beta):
    """Using Definition 3.1
    This is a quadratic-time algorithm. 
    Did not implement the linear-time J-list algorithm because other methods seem to out-perform.
    """
    lower, upper = bounds
    n = len(x)
    m = (n + 1) // 2
    
    # For notational convenience, define x_i = lower for any i <= 0 and x_i = upper for any i > n
    x = np.concatenate(([lower], x, [upper]))

    def local_sens(k, t):
        """Local sensitivity at neighboring distance k, with offset t"""
        return x[min(m + t, n + 1)] - x[max(0, m + t - k - 1)]

    def A(k):
        """Maximum of the local sensitivities at neighboring distance k"""
        return max(local_sens(k, t) for t in range(k + 2))

    def log_smooth_sens(k):
        """Log smooth sensitivity weighted by neighboring distance k.
        S(x, k) = A(k) * exp(k * beta)
        Log-transformed for numerical stability."""
        sens = A(k)
        return (np.log(sens) if sens > 0 else np.NINF) - k * beta

    # find the maximum of the log smooth sensitivities at each possible neighboring distance
    return np.exp(max(log_smooth_sens(k) for k in range(n + 1)))



def release_dp_median_via_ss(x, bounds, epsilon, delta):
    """Release the dp median via smooth sensitivity noise calibration
    See https://cs-people.bu.edu/ads22/pubs/NRS07/NRS07-full-draft-v1.pdf
    Uses L1 distance and laplace noise. Runs in time O(n^2)
    """

    # from Corollary 2.4, 2.
    alpha = epsilon / 2
    # from lemma 2.9
    beta = epsilon / (2 * np.log(2 / delta))

    x = np.clip(np.sort(x), *bounds)
    
    # NOTE: ignoring the odd length assumption in section 3.1
    median = x[(len(x) + 1) // 2]

    sens = compute_median_smooth_sensitivity(x, bounds, beta)

    return cond_laplace(median, sens/alpha)