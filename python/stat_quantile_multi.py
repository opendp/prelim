from stat_quantile import release_dp_quantile_via_ce
import numpy as np

# have not implemented JointExp and IndExp from GJK21 Differentially Private Quantiles
#    yet, because KSS21 claims better performance


def release_dp_approximate_quantiles(x, alphas, bounds, epsilon):
    """Release a collection of `alphas`-quantiles via the Continuous Exponential mechanism.
    Takes advantage of information gained in prior quantile estimates.
    See Algorithm 1: https://arxiv.org/pdf/2110.05429.pdf
    """

    def impl(x, alphas, bounds, epsilon):
        # base cases
        if len(alphas) == 0:
            return []

        if len(alphas) == 1:
            return [release_dp_quantile_via_ce(x, alphas[0], bounds, epsilon)]

        # estimate mid
        mid = (len(alphas) + 1) // 2
        p = alphas[mid]
        v = release_dp_quantile_via_ce(x, p, bounds, epsilon)

        # split x and alphas apart
        x_l, x_u = x[x < v], x[x > v]
        alphas_l, alphas_u = alphas[:mid] / p, (alphas[mid + 1 :] - p) / (1 - p)

        # recurse down left and right partitions
        return [
            *impl(x_l, alphas_l, (bounds[0], v), epsilon),
            v,
            *impl(x_u, alphas_u, (v, bounds[1]), epsilon),
        ]
    
    # recursively split the domain and set of quantiles
    return impl(x, alphas, bounds, epsilon / (np.log2(len(alphas)) + 1))


# TESTS
def test_release_dp_approximate_quantiles():
    print(
        release_dp_approximate_quantiles(
            x=np.random.uniform(size=1000),
            alphas=np.array([0.2, 0.3, 0.5, 0.7, 0.84]),
            bounds=(0.0, 1.0),
            epsilon=1.0,
        )
    )


# test_release_dp_approximate_quantiles()
