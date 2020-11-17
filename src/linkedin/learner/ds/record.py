from typing import Dict, List, Optional

from linkedin.learner.ds.feature import Feature
from linkedin.learner.utils.functions import dedupe_preserve_order

TRAINING_RECORD_REPR_SEPARATOR = ","


class TrainingRecord:
    """Data structure to hold a single training instance."""

    def __init__(self, label: float, weight: float, offset: float, features: List[Feature], context: Optional[Dict[str, str]] = None):
        self._label: float = label
        self.weight: float = weight
        self.offset: float = offset
        self.features: List[Feature] = features
        self.context: Optional[Dict[str, str]] = context

    @property
    def name_terms(self) -> List[Feature]:
        """Get the name-terms found in all features for a given training example.

        :return: A list of the name-terms in this training example.
        """
        return dedupe_preserve_order(ft.name_term for ft in self.features)

    @property
    def label(self) -> float:
        """Get the label for a given training example.

        :return: integer response (-1 for negative, 1 for positive).
        """
        return self._label if self._label == 1 else -1

    def __repr__(self):
        features_str = TRAINING_RECORD_REPR_SEPARATOR.join(str(feature) for feature in self.features)
        return f"[TrainingRecord: label = {self.label}; offset = {self.offset}; weight = {self.weight}; features = {features_str}]"
