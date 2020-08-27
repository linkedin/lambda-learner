import numpy as np
from scipy.stats import multivariate_normal

from linkedin.lambdalearnerlib.ds.indexed_dataset import IndexedDataset
from linkedin.lambdalearnerlib.ds.indexed_model import IndexedModel


def score_linear_model(model: IndexedModel, test_data: IndexedDataset, exploration_threshold: float = 0) -> np.ndarray:
    if exploration_threshold != 0:
        assert exploration_threshold > 0, "exploration_threshold must be non-negative."
        assert model.hessian is not None, "hessian must be available to perform exploration."

    # If exploration is on, we use a coefficient sample (from near the mean).
    # If not, we use the exact means.
    # When exploring, we only use the hessian's diagonal, for efficiency.
    coefficient_samples = (
        model.theta
        if exploration_threshold == 0
        else multivariate_normal.rvs(model.theta, exploration_threshold * model.hessian.diagonal())
    )

    scores = test_data.X * coefficient_samples + test_data.offsets
    return scores
