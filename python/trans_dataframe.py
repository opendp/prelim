import numpy as np


def transform_digitize_dataframe(df, labels, categorical):
    """Convert a pandas dataframe into an indexes array.

    :param df: columns must match the keys in `labels`
    :param labels: a {colname: list_of_edges_or_categories} dict
    :param categoricals: list of column names that should be considered categorical
    """
    columns = []
    for name in labels:
        if name in categorical:
            lookup = dict(zip(labels[name], range(len(labels[name]))))
            columns.append(df[name].map(lambda v: lookup[v]).values)
        else:
            # consider anything up to edge 2 to be part of bin 0
            #          anything above the second to last edge is part of the last bin
            columns.append(np.digitize(df[name].values, labels[name][1:-1]))

    return np.stack(columns, axis=1)


def test_digitize_dataframe():
    import pandas as pd

    data = pd.DataFrame(
        {
            "A": [1, 2, 4, 2],
            "B": ["nebraska", "montana", "idaho", "idaho"],
            "C": [0.1, 0.2, 0.5, 0.7],
        }
    )

    labels = {
        "A": [1, 2, 3, 4],  # categories
        "B": ["nebraska", "montana", "idaho", "utah"],  # categories
        "C": [0.0, 0.3, 0.5, 1.0],  # edges
    }

    categorical_names = ["A", "B"]

    digitized = transform_digitize_dataframe(data, labels, categorical_names)
    ground_truth = [[0, 0, 0], [1, 1, 0], [3, 2, 2], [1, 2, 2]]

    # check that the digitization is correct
    assert np.array_equal(digitized, ground_truth)
