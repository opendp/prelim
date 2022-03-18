from meas_exponential import mechanism_exponential_discrete
import numpy as np


def score_auction_price_discrete(x, candidates):
    return candidates * (x[None] >= candidates[:, None]).sum(axis=1)


def release_dp_auction_price_via_de(x, candidate_prices, epsilon):
    """Release a price for a digital auction that maximizes revenue.
    See Section 4.1, Theorem 9: http://kunaltalwar.org/papers/expmech.pdf
    :param x: The maximum each buyer is willing to spend.
    :param candidate_prices: potential price levels
    :returns a price that nearly maximizes revenue"""
    return mechanism_exponential_discrete(x, candidate_prices, epsilon,
        scorer=score_auction_price_discrete,
        sensitivity=max(candidate_prices))


def score_auction_constrained_price_discrete(x, candidate_pairs):
    return np.array([
        price * sum(x[product_id] >= price) 
        for price, product_id in candidate_pairs
    ])


def release_dp_auction_constrained_price_via_de(x, candidate_pairs, epsilon):
    """Release a price and product for a constrained pricing auction that maximizes revenue.
    See Section 4.3, Theorem 12: http://kunaltalwar.org/papers/expmech.pdf

    :param x: The maximum each buyer is willing to spend for each of k items.
    :param candidate_prices: list of potential (price, product_id) pairs, where product id is a column index into `x`
    :returns a (price, product_id) that nearly maximizes revenue"""
    prices = map(lambda v: v[0], candidate_pairs)

    return mechanism_exponential_discrete(x, candidate_pairs, epsilon,
        scorer=score_auction_constrained_price_discrete,
        sensitivity=max(prices))



def test_constrained_auction():
    # amount that each of 100 individuals are willing to offer for each of 4 products
    data = np.random.uniform(size=(1000, 4))

    # each candidate is a price and offering pair
    candidates = [(np.random.uniform(), np.random.randint(4)) for _ in range(100)]

    ideal_price, ideal_product = release_dp_auction_constrained_price_via_de(data, candidates, epsilon=1.)

    print(f"To maximize revenue, sell product {ideal_product} at {ideal_price}.")


def test_auction():
    data = np.random.uniform(size=100)
    candidates = np.random.uniform(size=5)
    
    print(release_dp_auction_price_via_de(data, candidates, epsilon=1.))

# test_constrained_auction()