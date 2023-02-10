import itertools
import torch
from typing import Dict, List, Tuple, Iterable


def generate_hparam_configs(base_config:Dict, hparam_ranges:Dict) -> Tuple[List[Dict], List[str]]:
    """
    Generate a list of hyperparameter configurations for hparam sweeping

    :param base_config (Dict): base configuration dictionary
    :param hparam_ranges (Dict): dictionary mapping hyperparameter names to lists of values to sweep over
    :return (Tuple[List[Dict], List[str]]): list of hyperparameter configurations and swept parameter names
    """

    keys, values = zip(*hparam_ranges.items())
    hparam_configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    swept_params = list(hparam_ranges.keys())

    new_configs = []
    for hparam_config in hparam_configurations:
        new_config = base_config.copy()
        new_config.update(hparam_config)
        new_configs.append(new_config)

    return new_configs, swept_params


def grid_search(num_samples: int, min: float = None, max: float = None, **kwargs)->Iterable:
    """ Implement this method to set hparam range over a grid of hyperparameters.
    :param num_samples (int): number of samples making up the grid
    :param min (float): minimum value for the allowed range to sweep over
    :param max (float): maximum value for the allowed range to sweep over
    :param kwargs: additional keyword arguments to parametrise the grid.
    :return (Iterable): tensor/array/list/etc... of values to sweep over

    Example use: hparam_ranges['batch_size'] = grid_search(64, 512, 6, log=True)

    **YOU MAY IMPLEMENT THIS FUNCTION FOR Q5**

    """
    raise NotImplementedError
    values = torch.zeros(num_samples)
    return values


def random_search(num_samples: int, distribution: str, min: float=None, max: float=None, **kwargs) -> Iterable:
    """ Implement this method to sweep via random search, sampling from a given distribution.
    :param num_samples (int): number of samples to take from the distribution
    :param distribution (str): name of the distribution to sample from
        (you can instantiate the distribution using torch.distributions, numpy.random, or else).
    :param min (float): minimum value for the allowed range to sweep over (for continuous distributions)
    :param max (float): maximum value for the allowed range to sweep over (for continuous distributions)
    :param kwargs: additional keyword arguments to parametrise the distribution.

    Example use: hparam_ranges['lr'] = random_search(1e-6, 1e-1, 10, distribution='exponential', lambda=0.1)

    **YOU MAY IMPLEMENT THIS FUNCTION FOR Q5**

    """
    raise NotImplementedError
    values = torch.zeros(num_samples)
    return values
