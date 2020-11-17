from itertools import chain
from typing import Dict, Iterable, Tuple

from linkedin.learner.ds.feature import Feature
from linkedin.learner.ds.record import TrainingRecord
from linkedin.learner.ds.types import NameTerm
from linkedin.learner.utils.functions import dedupe_preserve_order, flatten

INDEX_MAP_REPR_SEPARATOR = "; "


class IndexMap:
    """A mapping from a human readable feature domain to an indexed numeric feature domain.

    An IndexMap is used to translate human-readable model and data representations
    into the IndexedModel and IndexedDataset representations use for optimization.
    """

    def __init__(self, intercept_name="intercept", intercept_term="intercept"):
        self._intercept_name = intercept_name
        self._intercept_term = intercept_term

        self._size: int = 0
        self._nt_to_index: Dict[NameTerm, int] = {}
        self._index_to_nt: Dict[int, NameTerm] = {}

        # Ensure the intercept/bias feature is already indexed
        self.push(self.intercept_nt)

    @property
    def intercept_nt(self) -> NameTerm:
        return self._intercept_name, self._intercept_term

    @property
    def intercept_index(self):
        return self.get_index(self.intercept_nt)

    def push(self, name_term: NameTerm):
        """Add one term to the index map. No-op if the term is already present.

        NOT THREAD SAFE. This class is not thread safe and may not work correctly
        if shared for the training of multiple models, with features iteratively
        indexed during processing of different stream windows. If sharing and
        iterative indexing is desired in future, `push` and `batch_push` should
        be made atomic (instance-level lock).

        :param name_term: The term to add.
        :return: None
        """
        if name_term not in self._nt_to_index:
            index = self._size
            self._nt_to_index[name_term] = index
            self._index_to_nt[index] = name_term
            self._size = index + 1

    def batch_push(self, name_terms: Iterable[NameTerm]) -> Tuple[int, int]:
        """Add multiple terms to the index map, ignoring those already indexed.

        Also return some metadata about how many were actually added.

        :param name_terms: Multiple terms to try to index.
        :return: Number of terms added; and number of distinct terms it tried to add.
        """
        initial_index_size = len(self)
        distinct_nts = set()
        for nt in name_terms:
            distinct_nts.add(nt)
            self.push(nt)
        new_index_size = len(self)
        num_new_entries = new_index_size - initial_index_size
        num_nts_seen = len(distinct_nts)
        return num_new_entries, num_nts_seen

    def get_nt(self, index: int) -> NameTerm:
        """Look up a term corresponding to the given index.

        This raised an IndexError in the case of an out-of-bounds index.

        :param index: The index for which to look up the corresponding term.
        :return: The term, if is has been indexed.
        """
        if index in self._index_to_nt:
            return self._index_to_nt.get(index)  # type: ignore
        else:
            # User error. User should not ask for an out-of-bounds index.
            raise IndexError(f"Index {index} does not exist in index.")

    def get_index(self, name_term: NameTerm) -> int:
        """Look up the index for the given term.

        This raised an IndexError in the case of an unindexed term.

        :param name_term: The term for which to look up the corresponding index.
        :return: The index, if it exists.
        """
        if name_term in self._nt_to_index:
            return self._nt_to_index.get(name_term)  # type: ignore
        else:
            # User error. User should not ask for an unindexed NameTerm.
            raise IndexError(f"NameTerm {name_term} does not exist i index.")

    def __len__(self) -> int:
        return self._size

    def __repr__(self):
        pairs_str = INDEX_MAP_REPR_SEPARATOR.join(f"{k}:{v}" for k, v in self._nt_to_index.items())
        return f"[{pairs_str}]"

    @staticmethod
    def from_records_means_and_variances(
        training_records: Iterable[TrainingRecord], coefficient_mean: Iterable[Feature], coefficient_vars: Iterable[Feature]
    ) -> Tuple["IndexMap", Dict]:
        """Build an index map for given training data and coefficients.

        :param training_records: Training data.
        :param coefficient_mean: Model coefficient means.
        :param coefficient_vars: Model coefficient variances.
        :return: An index map which can be used for translating the above
                 data and model between the name term and index-domains.
        """
        imap = IndexMap()

        coeff_name_terms = [ft.name_term for ft in chain(coefficient_mean, coefficient_vars)]
        num_coeff_nts_added, _ = imap.batch_push(coeff_name_terms)

        features = dedupe_preserve_order(flatten(training_record.name_terms for training_record in training_records))
        num_record_nts_added, num_record_nts_seen = imap.batch_push(features)

        num_nts_in_both = num_record_nts_seen - num_record_nts_added

        imap_metadata = {
            "num_nts_in_both": num_nts_in_both,
            "num_nts_in_coeffs_only": num_coeff_nts_added - num_nts_in_both,
            "num_nts_in_records_only": num_record_nts_added,
        }

        return imap, imap_metadata
