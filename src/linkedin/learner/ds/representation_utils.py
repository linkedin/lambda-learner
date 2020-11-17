from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from linkedin.learner.ds.feature import Feature
from linkedin.learner.ds.index_map import IndexMap
from linkedin.learner.ds.indexed_dataset import IndexedDataset
from linkedin.learner.ds.indexed_model import IndexedModel
from linkedin.learner.ds.record import TrainingRecord
from linkedin.learner.ds.types import NameTerm


def nt_domain_data_to_index_domain_data(records: List[TrainingRecord], index_map: IndexMap) -> IndexedDataset:
    """Transforms a dataset from name-term to index domain representations.

    :param records: A dataset for training/scoring
    :param index_map: An index map defining the name-term to index mapping.

    :return: A representation of the data in the index domain, suitable for
             efficient training or scoring.
    """
    feature_values: List[float] = []
    rows: List[int] = []
    cols: List[int] = []
    offsets: List[float] = []
    weights: List[float] = []
    labels: List[float] = []

    for idx, training_record in enumerate(records):
        # Add intercept
        feature_values.append(1.0)
        cols.append(index_map.intercept_index)
        rows.append(idx)

        # Add other features
        features = training_record.features
        feature_values.extend(ft.value for ft in features)
        cols.extend(index_map.get_index(ft.name_term) for ft in features)
        rows.extend([idx] * len(features))

        offsets.append(training_record.offset)
        weights.append(training_record.weight)
        labels.append(training_record.label)

    data = sparse.coo_matrix((feature_values, (rows, cols)), shape=(len(records), len(index_map)), dtype=np.float64).tocsc()

    offsets = np.array(offsets)
    weights = np.array(weights)
    labels = np.array(labels)

    return IndexedDataset(data, labels, weights, offsets)


def nt_domain_coeffs_to_index_domain_coeffs(
    coeff_means: List[Feature], coeff_variances: List[Feature], index_map: IndexMap, regularization_param: float
) -> IndexedModel:
    """Transform a name-term domain model into an index domain model.

    This function uses the given IndexMap to transform a model representation
    from the human-readable name-term domain, into the vectorized index domain
    appropriate for optimization.

    :param coeff_means: Coefficient means.
    :param coeff_variances: Coefficient variances.
    :param index_map: Index map defining the name-term to index mapping.
    :param regularization_param: The regularization parameter used for training.
                                 Needed to initialize Hessian values when
                                 coefficients are missing in the model.
    :return: An IndexedModel, representing the model coefficients in the index
             domain, suitable for efficient training or scoring.
    """
    num_features = len(index_map)
    theta = np.zeros(num_features)
    for coefficient in coeff_means:
        index = index_map.get_index(coefficient.name_term)
        theta[index] = coefficient.value
    diag_hessian = regularization_param * np.ones(num_features)
    for coefficient in coeff_variances:
        index = index_map.get_index(coefficient.name_term)
        diag_hessian[index] = 1.0 / coefficient.value
    hessian = sparse.diags(diag_hessian).tocsc()
    return IndexedModel(theta, hessian)


def index_domain_coeffs_to_nt_domain_coeffs(
    model: IndexedModel, index_map: IndexMap
) -> Tuple[Dict[NameTerm, float], Optional[Dict[NameTerm, float]]]:
    """Transform an index domain model into a name-term domain model.

    Transform an IndexedModel into dictionaries from name-terms to values for
    the coefficient means and variances, using the provided IndexMap to define
    the translation from the index domain to the name-term domain.

    :param model: The index domain model (holding coefficient means and variances).
    :param index_map: The index map previously used to index the model and data.
    :return: Coefficient means and variances as name-term -> value dicts.
    """
    coeff_means = {}

    has_hessian = model.hessian is not None

    if has_hessian:
        coeff_variances = {}
        # At present, we only extract and save the diagonal Hessian.
        hessian_diagonal = model.hessian.diagonal()  # type: ignore

    for i in range(len(index_map)):
        name_term = index_map.get_nt(i)
        coeff_means[name_term] = model.theta[i]
        if has_hessian:
            coeff_variances[name_term] = 1.0 / hessian_diagonal[i]
    return coeff_means, coeff_variances
