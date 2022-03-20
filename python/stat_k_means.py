import numpy as np
from opendp.mod import enable_features

enable_features("contrib", "floating-point")


def distance_l2(x, y):
    return np.linalg.norm(x - y, axis=1)


def center_mean(data):
    return np.mean(data, axis=0)


def clip_norm(x, bound):
    return x / np.maximum(1, np.linalg.norm(x, axis=1) / bound)[:, None]


def sample_uniform_ball(n, d, bound):
    """Sample `n` points uniformly from the `d` dimensional ball with norm `bound`

    From Voelker et al. "Efficiently sampling vectors and coordinates from the n-sphere and n-ball"
        http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """

    # Sec 2.1: sample n points from the (d+1)-dimensional unit sphere
    u = np.random.normal(scale=1, size=(n, d + 2))
    u /= np.linalg.norm(u, axis=1)[:, None]

    # Sec 3.1: first d coordinates are uniformly distributed in the d-dimensional unit ball
    return u[:, :d] * bound


def kmeans_lloyd_step(x, centroids, distance=distance_l2, center=center_mean):
    """K-Means algorithm step"""
    # 1. vector quantization
    # distances of shape [n, num_centroids] from each point to each centroid
    dist_to_cluster = np.stack([distance(x, c) for c in centroids], axis=1)

    # index array of shape [n] denoting the cluster
    centroid_idx = np.argmin(dist_to_cluster, axis=1)

    # 2. find new centroids
    return np.stack([center(x[centroid_idx == i]) for i in range(len(centroids))])


def release_dp_kmeans_lloyd_step(x, bound, epsilon, centroids, distance=distance_l2):
    """Execute one `epsilon`-DP step of Lloyd's algorithm for computing K-means."""
    from post_fractional_estimate import debias_reciprocal
    from opendp.meas import make_base_geometric, make_base_gaussian

    numer_scale = 2 / epsilon * bound
    denom_scale = 2 / epsilon

    denom_base = make_base_geometric(numer_scale)
    numer_base = make_base_gaussian(denom_scale, D="VectorDomain<AllDomain<f64>>")

    def center_dp_mean(x_cluster):
        dp_numer = np.array(numer_base(x_cluster.sum(axis=0)))
        dp_denom = np.array(denom_base(len(x_cluster)))

        # divide with bias correction in denominator
        center = dp_numer / np.maximum(debias_reciprocal(dp_denom, denom_scale), 1)

        # optional sanity check to keep noisy means within the clipping ball
        center /= np.maximum(1, np.linalg.norm(center) / bound)

        return center

    # simply run the regular k means, but with an adjusted center function
    return kmeans_lloyd_step(x, centroids, distance, center_dp_mean)


def kmeans_lloyd(x, centroids, steps, distance=distance_l2, center=center_mean):
    """Non-dp KMeans algorithm"""
    for _ in range(steps):
        centroids = kmeans_lloyd_step(x, centroids, distance, center)
    return centroids


def release_dp_kmeans_lloyd(x, bound, epsilon, centroids, steps, distance=distance_l2):
    """Release an `epsilon`-DP estimate of K-means using Lloyd's algorithm."""
    step_epsilon = epsilon / steps

    for _ in range(steps):
        centroids = release_dp_kmeans_lloyd_step(
            x, bound, step_epsilon, centroids, distance
        )

    return centroids


def test_dp_kmeans():
    # dataset params
    true_data_size = 1_000
    true_center_norm_bound = 5.0
    true_n_clusters = 3

    # k-means params
    norm_bound = 5.0
    epsilon = 0.3
    n_centroids = 3
    steps = 4

    # ~~~~ Generate a dataset ~~~~
    true_centers = sample_uniform_ball(true_n_clusters, 3, true_center_norm_bound)

    cluster_size = np.random.uniform(0.1, size=true_n_clusters)
    cluster_size *= true_data_size / cluster_size.sum()

    clusters = []
    for center, size in zip(true_centers, cluster_size):
        # generate a positive semidefinite covariance matrix for each cluster
        rand = np.random.uniform(-1, 1, size=(3, 3))
        covariance = rand @ rand.T
        clusters.append(
            np.random.multivariate_normal(center, covariance, size=int(size))
        )

    # combine the point sets and shuffle rows
    x = np.vstack(clusters)
    np.random.shuffle(x)

    # ~~~~ K-Means Clustering ~~~~
    # only need to clip data once up-front
    x = clip_norm(x, norm_bound)

    # data-independent centroid initialization
    centroids = sample_uniform_ball(n_centroids, 3, norm_bound)

    # fit kmeans with and without privacy
    centroids_noisy = release_dp_kmeans_lloyd(x, norm_bound, epsilon, centroids, steps)
    centroids_exact = kmeans_lloyd(x, centroids, steps)

    # ~~~~ Visualization ~~~~
    print("True Centers:")
    print(true_centers)
    print("Noisy Centroids:")
    print(centroids_noisy)
    print("Exact Centroids:")
    print(centroids_exact)

    import matplotlib.pyplot as plt

    ax = plt.axes(projection="3d")

    # plot the dataset, colored by the nearest centroid
    ax.scatter(*true_centers.T, c="black", label="true centers")
    ax.scatter(*centroids_noisy.T, c="red", label="dp centroids")
    # ax.scatter(*centroids_exact.T, c="green", label="exact centroids")

    # vector quantization assigns each point to the nearest DP centroid
    from scipy import cluster

    centroid_idx, _ = cluster.vq.vq(x, centroids_noisy)
    ax.scatter(*x.T, c=plt.cm.tab10(centroid_idx), s=3, alpha=0.3)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.legend()
    plt.show()


test_dp_kmeans()
