import numpy as np


def postprocess_synthesize_indexes(hist, size):
    """Use an n-dimensional histogram to generate a synthetic dataset of cell indexes

    
    :param hist: an n-dimensional array of counts
    :param size: how many rows to synthesize in the output dataset
    :returns synthetic index dataset with `hist.ndim` columns and `size` records.
    """

    # the cdf is wrt the flattened bins
    cdf = hist.ravel().cumsum().astype(np.float)
    cdf /= cdf[-1]

    # use inverse transform sampling to sample a set of unraveled bin indexes
    values = np.random.uniform(size=size)
    value_bins = np.searchsorted(cdf, values)

    # translate the unraveled bin indexes to nd indexes
    return np.unravel_index(value_bins, hist.shape)


def postprocess_synthesize_categorical(hist, categories, size):
    """Use an n-dimensional histogram to generate a synthetic dataset of categorical data.
    
    :param hist: an n-dimensional array of counts
    :param edges: edges from which each dimension of hist was binned
    :param size: how many rows to synthesize in the output dataset
    :returns synthetic dataset with `hist.ndim` columns and `size` records.
    """
    # generate the synthetic index dataset
    synthetic_indices = postprocess_synthesize_indexes(hist, size)

    # retrieve the midpoint values from the respective bins
    return np.stack([cats[idxs] for cats, idxs in zip(categories, synthetic_indices)], axis=1)


def get_midpoints(edges):
    return edges[:-1] + np.diff(edges) / 2


def postprocess_synthesize_continuous(hist, edges, size):
    """Use an n-dimensional histogram to generate a synthetic dataset of continuous data.
    
    :param hist: an n-dimensional array of counts
    :param edges: edges from which each dimension of hist was binned
    :param size: how many rows to synthesize in the output dataset
    :returns synthetic dataset with `hist.ndim` columns and `size` records.
    """
    # can consider the continuous case to be a special case of categorical sampling
    midpoints = [get_midpoints(edge_set) for edge_set in edges]
    return postprocess_synthesize_categorical(hist, midpoints, size)



def test_synthetic_data():
    true_beta = 1.


    def generate_data(size, bounds):
        x_bounds, y_bounds = bounds
        x = np.random.uniform(*x_bounds, size)
        y = np.clip(np.random.normal(true_beta * x, scale=.2), *y_bounds)
        return x, y


    def plot_histograms(true, synth, bins, bounds):
        import matplotlib.pyplot as plt

        plt.subplot(121, aspect='equal')
        plt.hist2d(*true, bins=bins, range=bounds)
        plt.subplot(122, aspect='equal')
        plt.hist2d(*synth, bins=bins, range=bounds)
        plt.show()

    def simulate(x, y, size, bounds, n_bins, epsilon=None, threshold=0):

        # Compute histogram
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins, range=bounds)

        # Make a DP release of the histogram
        if epsilon:
            hist = np.round(np.random.laplace(hist, scale=1. / epsilon))
            hist[hist < max(0, threshold)] = 0

        x_synth, y_synth = postprocess_synthesize_continuous(hist, [x_edges, y_edges], size).T

        return x_synth, y_synth

    data_size = 500
    data_bounds = [[-5, 5], [-5, 5]]
    n_bins = 20

    x, y = generate_data(data_size, data_bounds)
    x_synth, y_synth = simulate(x, y, size=data_size, bounds=data_bounds, n_bins=n_bins, epsilon=1., threshold=0.75)

    plot_histograms((x, y), (x_synth, y_synth), n_bins, data_bounds)


# test_synthetic_data()