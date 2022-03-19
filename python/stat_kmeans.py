import numpy as np
from opendp.mod import enable_features

enable_features("contrib", "floating-point")


def distance_l2(x, y):
    return np.linalg.norm(x - y, axis=1)


def center_mean(data):
    return np.mean(data, axis=0)


def kmeans_lloyd_step(x, centroids, distance=distance_l2, center=center_mean):
    """Non-dp KMeans algorithm step"""
    # 1. vector quantization
    # distances of shape [n, num_centroids] from each point to each centroid
    dist_to_cluster = np.stack([distance(x, c) for c in centroids], axis=1)

    # index array of shape [n] denoting the cluster
    centroid_idx = np.argmin(dist_to_cluster, axis=1)

    # 2. find new centroids
    return np.stack([center(x[centroid_idx == i]) for i in range(len(centroids))])
    


def kmeans_lloyd(x, centroids, steps, distance=distance_l2, center=center_mean):
    """Non-dp KMeans algorithm"""
    if isinstance(centroids, (int, np.integer)):
        centroids = x[np.random.randint(len(x), size=centroids)]

    for _ in range(steps):
        centroids = kmeans_lloyd_step(x, centroids, distance, center)

    return centroids


def release_dp_kmeans_lloyd_step(x, bound, epsilon, centroids, distance=distance_l2):
    """Execute one `epsilon`-DP step of lloyd's algorithm for computing K-means."""
    from post_fractional_estimate import debias_reciprocal
    from opendp.meas import make_base_geometric, make_base_gaussian

    numer_scale = 2 / epsilon * bound
    denom_scale = 2 / epsilon

    numer_base = make_base_gaussian(denom_scale, D="VectorDomain<AllDomain<f64>>")
    denom_base = make_base_geometric(numer_scale)

    def center(data):
        data = np.array(data)

        data /= np.minimum(1., np.linalg.norm(data, axis=1) / bound)[:,None]

        dp_numer = numer_base(data.sum(axis=0))
        dp_denom = denom_base(len(data))

        return tuple(dp_numer / debias_reciprocal(dp_denom, denom_scale))

    return kmeans_lloyd_step(x, centroids, distance, center)


def release_dp_kmeans_lloyd(x, bound, epsilon, centroids, steps, distance=distance_l2):
    """Release an `epsilon`-DP estimate of K-means using lloyd's algorithm."""

    if isinstance(centroids, (int, np.integer)):
        centroids = np.random.uniform(size=(centroids, x.shape[1]))
        centroids = set(map(tuple, centroids))

    step_epsilon = epsilon / steps

    for _ in range(steps):
        centroids = release_dp_kmeans_lloyd_step(
            x, bound, step_epsilon, centroids, distance
        )

    return centroids




def test_dp_kmeans():
    # ~~~~ Generate a dataset ~~~~
    # Generate a positive semidefinite covariance matrix for each cluster
    rand = np.random.uniform(-1, 1, size=(3, 3))
    covariance_1 = rand @ rand.T
    source_one = np.random.multivariate_normal((-2, -1, 4), covariance_1, size=400)

    rand = np.random.uniform(-1, 1, size=(3, 3))
    covariance_2 = rand @ rand.T
    source_two = np.random.multivariate_normal((4, 0, -3), covariance_2, size=200)

    # Combine the two point sets and shuffle rows
    dataset = np.vstack((source_one, source_two))
    np.random.shuffle(dataset)

    # ~~~~ K-Means Clustering ~~~~
    from scipy import cluster
    bound = 6.
    epsilon = 0.3
    centroids = 2
    steps = 4
    noisy_codebook = release_dp_kmeans_lloyd(dataset, bound, epsilon, centroids, steps)
    exact_codebook = kmeans_lloyd(dataset, centroids, steps)

    print("Noisy Centroids:")
    print(noisy_codebook)
    print("Exact Centroids:")
    print(exact_codebook)

    # Use the codebook to assign each observation to a cluster via vector quantization
    labels_noisy, _ = cluster.vq.vq(dataset, noisy_codebook)

    # Use boolean indexing to extract points in a cluster from the dataset
    cluster_one = dataset[labels_noisy == 0]
    cluster_two = dataset[labels_noisy == 1]

    # ~~~~ Visualization ~~~~
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*noisy_codebook.T, c='red', alpha=1, label="noisy")
    ax.scatter(*exact_codebook.T, c='black', alpha=1, label="exact")
    ax.scatter(*cluster_one.T, c='b', s=3, alpha=0.2)
    ax.scatter(*cluster_two.T, c='g', s=3, alpha=0.2)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()

# test_dp_kmeans()