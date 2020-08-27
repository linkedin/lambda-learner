from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def __init__(self):
        """
        Initialize model parameters
        """
        pass

    @abstractmethod
    def loss(self, theta):
        """
        Compute the loss for a batch using the error function and sample weights
        :return: float
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, theta):
        """
        Compute the derivative of the loss function used for gradient descent
        :return: float
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, num_corrections: int, precision: float, max_iterations: int):
        """
        Runs a few iterations of gradient descent to compute the new model coefficients
        :return: coefficient vectors
        """
        raise NotImplementedError
