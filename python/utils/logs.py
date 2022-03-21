import numpy as np


def log_sub_exp(x, y):
    """Evaluate log(exp(x) - exp(y))"""
    if x == y:
        # x == y implies exp(x) - exp(y) == 0
        return np.NINF

    # https://stackoverflow.com/q/52045105/10221612
    # if x > y
    #   = log(exp(x) * (exp(x)-exp(y)) / exp(x))
    #   = x + log(1-exp(y-x))
    # if x < y, then swap
    #   = log(exp(y) * (exp(x)-exp(y)) / exp(y))
    #   = y + log(1-exp(x-y))
    a, b = sorted((x, y))
    return b + np.log(1 - np.exp(b - a))

