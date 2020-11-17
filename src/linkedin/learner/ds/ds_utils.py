from typing import Dict, List

from linkedin.learner.ds.feature import Feature
from linkedin.learner.ds.types import NameTerm, Value


def coefficient_dict_to_feature_list(coefficient_dict: Dict[NameTerm, Value]) -> List[Feature]:
    """Transform feature data from a Dict to a List form.

    See unit test for example input and output.

    :param coefficient_dict: A dictionary mapping (name, term) -> value.
    :return: A list of Feature(name, term, value).
    """
    return [Feature(name, term, value) for (name, term), value in coefficient_dict.items()]


def feature_list_coefficient_dict(feature_list: List[Feature]) -> Dict[NameTerm, Value]:
    """Transform feature data from a List to a Dict form.

    See unit test for example input and output.

    :param feature_list: A list for Feature(name, term, value)
    :return: A dictionary mapping (name, term) -> value.
    """
    return {f.name_term: f.value for f in feature_list}
