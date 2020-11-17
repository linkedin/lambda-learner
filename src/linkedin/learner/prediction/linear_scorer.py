import numpy as np
from scipy.stats import multivariate_normal

from linkedin.learner.ds.indexed_dataset import IndexedDataset
from linkedin.learner.ds.indexed_model import IndexedModel


def score_linear_model(model: IndexedModel, test_data: IndexedDataset, exploration_threshold: float = 0) -> np.ndarray:
    """Score test data using the given model.

    This function supports using the coefficient variance to allow scoring using
    coefficients sampled from near the mean coefficient value (Thompson Sampling).
    This is controlled by the `exploration_threshold` parameter.

    :param model: The model.
    :param test_data: The test data.
    :param exploration_threshold: The exploration threshold.
    :returns: The scores for every test example.
    """
    if exploration_threshold != 0:
        if exploration_threshold < 0:
            raise ValueError("Exploration_threshold must be non-negative.")
        if model.hessian is None:
            raise ValueError("Hessian must be available to perform exploration.")

    # When exploring, we only use the Hessian's diagonal, for efficiency.
    coefficient_samples = (
        model.theta
        if exploration_threshold == 0
        else multivariate_normal.rvs(model.theta, exploration_threshold * model.hessian.diagonal())  # type: ignore
    )

    scores = test_data.X * coefficient_samples + test_data.offsets
    return scores
