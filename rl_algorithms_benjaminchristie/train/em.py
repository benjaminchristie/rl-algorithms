import torch
import numpy as np
from copy import deepcopy

from utils.misc import es_params


def fitness_shaping(returns):
    """
    A rank transformation on the rewards, which reduces the chances
    of falling into local optima early in training.
    """
    sorted_returns_backwards = sorted(returns)[::-1]
    lamb = len(returns)
    shaped_returns = []
    denom = sum(
        [
            max(
                0,
                np.log2(lamb / 2 + 1) - np.log2(sorted_returns_backwards.index(r) + 1),
            )
            for r in returns
        ]
    )
    for r in returns:
        num = max(
            0, np.log2(lamb / 2 + 1) - np.log2(sorted_returns_backwards.index(r) + 1)
        )
        shaped_returns.append(num / denom + 1 / lamb)
    return shaped_returns


def unperturbed_rank(returns, unperturbed_results):
    nth_place = 1
    for r in returns:
        if r > unperturbed_results:
            nth_place += 1
    rank_diag = "%d out of %d (1 means gradient " "is uninformative)" % (
        nth_place,
        len(returns) + 1,
    )
    return rank_diag, nth_place


def perturb(model, sigma: float):
    new_model = deepcopy(model)
    for k, v in es_params(new_model):
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(sigma * eps).float()
    return new_model
