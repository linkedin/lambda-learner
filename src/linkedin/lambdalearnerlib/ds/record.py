from typing import Dict, List, Optional

from linkedin.lambdalearnerlib.ds.feature import Feature
from linkedin.lambdalearnerlib.utils.functions import dedupe_preserve_order


class TrainingRecord(object):
    """
    Data structure to hold a single training instance.
    """

    def __init__(self, label: float, weight: float, offset: float, features: List[Feature], context: Optional[Dict[str, str]] = None):
        self._label: float = label
        self.weight: float = weight
        self.offset: float = offset
        self.features: List[Feature] = features
        self.context: Optional[Dict[str, str]] = context

    @property
    def name_terms(self) -> List[Feature]:
        """
        Set of NameTerm objects found in all features for a given training example
        :return: list[Feature]
        """
        return dedupe_preserve_order(ft.name_term for ft in self.features)

    @property
    def label(self) -> float:
        """
        User response associated with a given training example
        :return: integer response (-1 for negative, 1 for positive)
        """
        return self._label if self._label == 1 else -1

    def __repr__(self):
        features_str = ",".join(str(feature) for feature in self.features)
        return f"[TrainingRecord: label = {self.label}; offset = {self.offset}; weight = {self.weight}; features = {features_str}]"
