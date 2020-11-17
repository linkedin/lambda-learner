from abc import ABC, abstractmethod

import numpy as np


class Trainer(ABC):
    @abstractmethod
    def __init__(self):
        """Initialize model parameters."""
        pass

    @abstractmethod
    def loss(self, theta):
        """Compute the loss.

        :return: The value of the loss at `theta`.
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, theta):
        """Compute the derivative of the loss.

        :return: The value of the loss gradient at `theta`.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, num_corrections: int, precision: float, max_iterations: int):
        """Optimize the model coefficients.

        :return: Optimized coefficient vector.
        """
        raise NotImplementedError


# Keys used with the [[with_previous_theta_transform_memoized]] decorator.
KEY = "theta"
VALUE = "value"


def with_previous_theta_transform_memoized(func):
    """Memoize the latest result.

    Decorator for a Trainer method of the form `func(self, theta)`, which memoizes
    the result of the previous computation. This is designed specifically as an
    performance optimization for methods called repeatedly during LBFGS, such as
    `loss` and `gradient`. LBFGS will call `loss` and `gradient` as a pair,
    repeatedly. Typically `loss` and `gradient` share terms in common (e.g.
    residuals or other affine transforms of theta), whose computation is somewhat
    expensive and can be cached. However full memoization is unnecessary, as we
    are unlikely to ever need a cached result again once the next LBFGS step starts.
    Hence we only keep the previous result (letting `gradient` reuse the most
    expensive part of computing `loss`), while keeping memory overhead minimal.

    :param func: a term computation method used in `loss` or `gradient` of the form func(self, theta).
    :return: prev-result memoized version of func.
    """

    def func_wrapper(self, theta: np.ndarray):
        # Setup cache on the instance
        if not hasattr(self, "__cache"):
            self.__cache = {}

        # Since this decorator could be used with multiple methods,
        if func not in self.__cache:
            self.__cache[func] = {KEY: None, VALUE: None}

        cache_for_func = self.__cache[func]
        candidate_theta_key = theta.tobytes()

        if cache_for_func[KEY] != candidate_theta_key:
            cache_for_func[KEY] = candidate_theta_key
            cache_for_func[VALUE] = func(self, theta)

        return cache_for_func[VALUE]

    return func_wrapper
