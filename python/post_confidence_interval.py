import numpy as np


def postprocess_dp_mean_confidence_interval_normal(
    data, data_size, alpha, measurement, sim_size=1_000
):
    """Construct a DP 100(1-alpha)% confidence interval for the DP mean measurement.
    Assumes underlying data is normally distributed.

    Algorithm 3: https://arxiv.org/pdf/2001.02285.pdf

    :param data: a tuple of (mean, variance)
    :param data_size: number of records to sample per simulation
    :param alpha: statistical significance level
    :param measurement: a function that takes a vector of `data_size` iid gaussian samples
    :param sim_size: number of simulations to run. Optional, higher is asymptotically more accurate
    :returns a tuple with lower and upper confidence bounds
    """

    mean, var = data

    def simulate():
        """simulate measurement on `data_size` iid gaussians"""
        return measurement(np.random.normal(mean, np.sqrt(var), size=data_size))

    sims = [simulate() for _ in range(sim_size)]

    # margin of error
    q_lower, q_upper = np.quantile(sims, q=[alpha / 2, 1 - alpha / 2])
    moe = (q_upper - q_lower) / 2

    return mean - moe, mean + moe


def postprocess_mean_confidence_interval_normal(data, data_size, alpha):
    """Construct a DP 100(1-alpha)% confidence interval for the true mean.
    Assumes underlying data is normally distributed.

    :param data: a tuple of (mean, variance)
    :param data_size: number of records to sample per simulation
    :param alpha: statistical significance level
    """
    import scipy.stats

    mean, var = data
    moe = np.sqrt(var / data_size) * scipy.stats.norm.ppf(1 - alpha / 2)
    return mean - moe, mean + moe


def _exact_ci(x: np.ndarray, alpha):
    """Construct a 100(1-alpha)% confidence interval for the true mean."""
    return postprocess_mean_confidence_interval_normal(
        (x.mean(), x.std()), x.size, alpha
    )


def test_postprocess_normal_mean_confidence_interval():
    alpha = 0.05

    data_size = 1000
    bounds = (10.0, 30.0)
    scale = 0.5

    data = np.random.normal(loc=23.0, scale=2.0, size=data_size)

    from opendp.trans import make_sized_bounded_mean, make_sized_bounded_variance
    from opendp.trans import make_clamp, make_bounded_resize
    from opendp.meas import make_base_laplace

    from opendp.mod import enable_features

    enable_features("contrib", "floating-point")

    preprocess = make_clamp(bounds) >> make_bounded_resize(data_size, bounds, bounds[0])
    meas_mean = (
        preprocess
        >> make_sized_bounded_mean(data_size, bounds)
        >> make_base_laplace(scale)
    )
    meas_var = (
        preprocess
        >> make_sized_bounded_variance(data_size, bounds)
        >> make_base_laplace(scale)
    )

    release = meas_mean(data), meas_var(data)
    mean, _ = release

    # confidence interval for noise added to compute DP mean
    from opendp.accuracy import laplacian_scale_to_accuracy
    mech_accuracy = laplacian_scale_to_accuracy(scale, alpha)
    mech_lower, mech_upper = mean - mech_accuracy, mean + mech_accuracy

    # DP confidence interval of DP mean
    dp_dp_lower, dp_dp_upper = postprocess_dp_mean_confidence_interval_normal(
        release, data_size=data_size, alpha=alpha, measurement=meas_mean
    )

    # DP confidence interval of true mean
    dp_ex_lower, dp_ex_upper = postprocess_mean_confidence_interval_normal(
        release, data_size, alpha
    )

    # true confidence interval of true mean
    ex_lower, ex_upper = _exact_ci(data, alpha)

    print(f"{1 - alpha:.0%} confidence bounds")
    print(f"CI of DP mean mech:   ({mech_lower:.4f}, {mech_upper:.4f})")
    # wider than usual to account for both the true CI and DP noise
    print(f"DP CI of DP mean:     ({dp_dp_lower:.4f}, {dp_dp_upper:.4f})")
    # coverage probability will be poor because the interval doesn't account for DP noise
    print(f"DP CI of true mean:   ({dp_ex_lower:.4f}, {dp_ex_upper:.4f})")
    # non-DP comparison
    print(f"True CI of true mean: ({ex_lower:.4f}, {ex_upper:.4f})")


test_postprocess_normal_mean_confidence_interval()
