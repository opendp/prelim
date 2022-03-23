import numpy as np
from utils.mock import MockTransformation, MockMeasurement
from utils.samplers import bernoulli

def make_private_selection_queryable_fix_candidate(candidates, measurement):
    # TODO: this returns a queryable that, on each invocation, returns a release
    #       but the forward map only accounts for it being called once
    #       makes sense in the context of the private selection mechanism, but not on its own

    # x is a 2d-array, where first column is category
    def function(x):
        import random

        while True:
            candidate = random.choice(candidates)

            yield *measurement(x[x[:, 0] == candidate, 1].astype(np.int32)), candidate

    return MockTransformation(function, forward_map=measurement.forward_map)


def mechanism_private_selection(queryable_builder, stop_probability):
    """
    Algorithm 2: https://arxiv.org/pdf/1811.07971.pdf#subsection.3.2

    :param queryable_builder: a measurement that, when invoked, creates a queryable
        Each invocation of the queryable returns a tuple where the first element is a score.
        Each tuple is "queryable_builder.forward_map(d_in)"-DP
    :param stop_probability: probability of stopping each iteration
    :returns tuple from `queryable_builder` with greatest first element
    """

    def function(x):
        queryable = queryable_builder(x)
        best_score = -float("inf")
        best_y = None
        while True:
            score, *y = next(queryable)
            if score > best_score:
                best_score = score
                best_y = y

            if bernoulli(stop_probability):
                return best_score, *best_y

    def forward_map(d_in):
        return queryable_builder.forward_map(d_in) * 3

    return MockMeasurement(function, forward_map)



def mechanism_private_selection_threshold(
    queryable_builder,
    stop_probability,
    threshold,
    epsilon_selection,
    steps=None,
):
    """

    Algorithm 1: https://arxiv.org/pdf/1811.07971.pdf#subsection.3.1

    :param queryable_builder: a measurement that, when invoked, creates a queryable
        Each invocation of the queryable returns a tuple where the first element is a score.
        Each tuple is "queryable_builder.forward_map(d_in)"-DP
    :param stop_probability: probability of stopping early at any iteration
    :param threshold: Return immediately if score is above this threshold.
    :param epsilon_selection: epsilon allocated to the private selection
    :param steps: optional. How many steps to run. If not specified, will run minimum number of steps
    :returns tuple from `queryable_builder` with greatest first element
    """
    # T. From proof for (b), budget consumption
    # https://arxiv.org/pdf/1811.07971.pdf#page=25
    min_steps = int(
        np.ceil(max(np.log(2 / epsilon_selection) / stop_probability, 1 + 1 / np.exp(stop_probability)))
    )

    steps = steps or min_steps

    if steps < min_steps:
        raise ValueError(f"must run at least {min_steps} steps")

    def function(x):
        queryable = queryable_builder(x)
        for _ in range(steps):
            score, *y = next(queryable)

            if score >= threshold:
                return score, *y

            if bernoulli(stop_probability):
                return

    def forward_map(d_in):
        """Theorem 3.1 (b)"""
        return queryable_builder.forward_map(d_in) * 2 + d_in * epsilon_selection
    
    def forward_map_approx(d_in):
        """Theorem 3.1 (c)"""
        epsilon_q, delta_q = queryable_builder.forward_map(d_in)
        epsilon = epsilon_q * 2 + d_in * epsilon_selection
        delta = 3 * np.exp(epsilon) * delta_q / stop_probability
        return epsilon, delta
    
    return MockMeasurement(function, forward_map)
    


def test_private_selection():
    epsilon = 1.0
    bounds = (0, 100)
    categories = [1, 2, 3]

    from opendp.trans import make_count, make_clamp, make_bounded_sum
    from opendp.meas import make_base_geometric
    from opendp.mod import enable_features

    enable_features("contrib")

    # make a measurement that Q invokes, a randomized mechanism that
    # 1. consumes epsilon-DP
    # 2. returns a tuple where first argument is the score
    count = make_count(TIA=int, TO=int) >> make_base_geometric(2 / epsilon)
    mean = (
        make_clamp(bounds)
        >> make_bounded_sum(bounds)
        >> make_base_geometric(200 / epsilon)
    )
    # hardcode epsilon in here. But if library had forward maps, it would be the sum of the forward maps eval'ed on d_in
    composed = MockMeasurement(
        function=lambda x: (count(x), mean(x)), forward_map=lambda d_in: d_in * epsilon
    )

    # create Q, a queryable that
    # 1. randomly selects a candidate
    # 2. evaluates `composed` on it
    psq = make_private_selection_queryable_fix_candidate(
        candidates=categories, measurement=composed
    )

    # create a private selection mechanism that will release one output from the queryable with the greatest score
    private_selector = mechanism_private_selection(
        queryable_builder=psq, stop_probability=0.1
    )

    # categories in first column, data in second column
    data = np.stack(
        [np.random.choice(categories, size=100), np.random.randint(0, 100, size=100)],
        axis=1,
    )

    # returns (num elements in category, sum of second column in category, category)
    selected = private_selector(data)
    print(selected)

    # find what the epsilon usage is when input symmetric dataset distance is 2.
    # Should be d_in * epsilon * 3
    print(private_selector.forward_map(2))


# test_private_selection()

def test_private_selection_threshold():
    epsilon = 1.0
    bounds = (0, 100)
    categories = [1, 2, 3]

    from opendp.trans import make_count, make_clamp, make_bounded_sum
    from opendp.meas import make_base_geometric
    from opendp.mod import enable_features

    enable_features("contrib")

    # make a measurement that Q invokes, a randomized mechanism that
    # 1. consumes epsilon-DP
    # 2. returns a tuple where first argument is the score
    count = make_count(TIA=int, TO=int) >> make_base_geometric(2 / epsilon)
    mean = (
        make_clamp(bounds)
        >> make_bounded_sum(bounds)
        >> make_base_geometric(200 / epsilon)
    )
    # hardcode epsilon in here. But if library had forward maps, it would be the sum of the forward maps eval'ed on d_in
    composed = MockMeasurement(
        function=lambda x: (count(x), mean(x)), forward_map=lambda d_in: d_in * epsilon
    )

    # create Q, a queryable that
    # 1. randomly selects a candidate
    # 2. evaluates `composed` on it
    psq = make_private_selection_queryable_fix_candidate(
        candidates=categories, measurement=composed
    )

    # create a private selection mechanism that will release one output from the queryable with the greatest score
    private_selector = mechanism_private_selection_threshold(
        queryable_builder=psq, stop_probability=0.1, threshold=34, epsilon_selection=0.5
    )

    # categories in first column, data in second column
    data = np.stack(
        [np.random.choice(categories, size=100), np.random.randint(0, 100, size=100)],
        axis=1,
    )

    # returns (num elements in category, sum of second column in category, category)
    selected = private_selector(data)
    print(selected)

    # find what the epsilon usage is when input symmetric dataset distance is 2.
    # Should be d_in * epsilon * 2 + d_in * epsilon_selection
    print(private_selector.forward_map(2))


# test_private_selection_threshold()
