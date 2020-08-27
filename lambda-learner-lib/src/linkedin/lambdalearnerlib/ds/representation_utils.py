from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

from linkedin.lambdalearnerlib.ds.feature import Feature
from linkedin.lambdalearnerlib.ds.index_map import IndexMap
from linkedin.lambdalearnerlib.ds.indexed_dataset import IndexedDataset
from linkedin.lambdalearnerlib.ds.indexed_model import IndexedModel
from linkedin.lambdalearnerlib.ds.record import TrainingRecord
from linkedin.lambdalearnerlib.ds.types import NameTerm


def nt_domain_data_to_index_domain_data(records: List[TrainingRecord], index_map: IndexMap) -> IndexedDataset:
    """
    Transforms a dataset from a name-term domain representation into an index domain representation.
    :param records: A dataset for training/scoring
    :param index_map: Index map defining the name-term to index mapping.
    :return: A representation of the data in the index domain, suitable for efficient training or scoring.
    """
    feature_values, rows, cols, offsets, weights, labels = [], [], [], [], [], []

    for idx, tr in enumerate(records):
        # Add intercept
        feature_values.append(1.0)
        cols.append(index_map.intercept_index)
        rows.append(idx)

        # Add other features
        features = tr.features
        feature_values.extend(ft.value for ft in features)
        cols.extend(index_map.get_index(ft.name_term) for ft in features)
        rows.extend([idx] * len(features))

        offsets.append(tr.offset)
        weights.append(tr.weight)
        labels.append(tr.label)

    # Note: Why sparse and why CSC? CSR and CSC are better suited for linear algebra operations than DOK, DIA, LIL, etc.
    # In practice, performance testing has revealed little difference between CSR and CSC representations for the sorts
    # of operations lambda performs. However, sparse is considerably faster than dense for our typical data.
    # See go/ll-matrix-performance.
    data = sparse.coo_matrix((feature_values, (rows, cols)), shape=(len(records), len(index_map)), dtype=np.float64).tocsc()

    offsets = np.array(offsets, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    labels = np.array(labels)

    return IndexedDataset(data, labels, weights, offsets)


def nt_domain_coeffs_to_index_domain_coeffs(
    coeff_means: List[Feature], coeff_variances: List[Feature], index_map: IndexMap, regularization_param: float
) -> IndexedModel:
    """
    Transforms a model represented in the nam-term domain into one represented in the index domain, using the given index map.
    :param coeff_means: Coefficient means
    :param coeff_variances: Coefficient variances
    :param index_map: Index map defining the name-term to index mapping.
    :param regularization_param: The regularization parameter that will be used for training. Needed to initialize
                                 hessian values when coefficients are missing in the model.
    :return: An IndexedModel, representing the model coefficients in the index domain, suitable for efficient training or scoring.
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

    # Note: Why sparse and why CSC? See go/ll-matrix-performance.
    hessian = sparse.diags(diag_hessian).tocsc()

    return IndexedModel(theta, hessian)


def index_domain_coeffs_to_nt_domain_coeffs(
    model: IndexedModel, index_map: IndexMap
) -> Tuple[Dict[NameTerm, float], Dict[NameTerm, float]]:
    """
    Transforms an IndexedModel into dictionaries from name-terms to values for the coefficient means and variances,
    using the provided IndexMap to define the translation from the index domain to the name-term domain.
    :param model: The index domain model (containing the coefficient means and variances).
    :param index_map: The index map previously used to index the model and data.
    :return: Coefficient means and variances as name-term -> value dicts.
    """
    coeff_means, coeff_variances = {}, {}
    # Note - We only extract and save the diagonal hessian (as of v0.0.115)
    hessian_diagonal = model.hessian.diagonal()
    for i in range(len(index_map)):
        name_term = index_map.get_nt(i)
        coeff_means[name_term] = model.theta[i]
        coeff_variances[name_term] = 1.0 / hessian_diagonal[i]
    return coeff_means, coeff_variances
