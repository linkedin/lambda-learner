import unittest

from linkedin.lambdalearnerlib.ds.feature import Feature
from linkedin.lambdalearnerlib.ds.index_map import IndexMap
from linkedin.lambdalearnerlib.ds.record import TrainingRecord


class IndexMapTest(unittest.TestCase):
    def test_default_imap_construction(self):
        imap = IndexMap()
        self.assertEqual(len(imap), 1, "Intercept is already indexed.")
        self.assertEqual(imap.intercept_nt, ("intercept", "intercept"), 'Intercept name and term are both "intercept" by default.')
        self.assertEqual(imap.intercept_index, 0, "Intercept is indexed first.")

    def test_custom_imap_construction(self):
        imap = IndexMap("iname", "iterm")
        self.assertEqual(len(imap), 1, "Intercept is already indexed.")
        self.assertEqual(imap.intercept_nt, ("iname", "iterm"), 'Intercept name and term are both "intercept" by default.')
        self.assertEqual(imap.intercept_index, 0, "Intercept is indexed first.")

    def test_imap_index_errors(self):
        imap = IndexMap()

        with self.assertRaises(IndexError, msg="Trying to get unindexed features results in IndexError"):
            imap.get_nt(1)

        n1t1 = ("n1", "t1")
        imap.push(n1t1)

        self.assertEqual(imap.get_nt(1), n1t1, "IndexError no longer raised once index has been assigned.")

    def test_one_at_a_time_imap_construction(self):
        imap = IndexMap()
        self.assertEqual(len(imap), 1, "Intercept is already indexed.")

        n1t1 = ("n1", "t1")
        imap.push(n1t1)

        self.assertEqual(len(imap), 2, "Pushing a new one feature increases the imap size by 1.")
        self.assertEqual(imap.get_index(n1t1), 1, "The next available index is used.")
        self.assertEqual(imap.get_nt(1), n1t1, "The feature can be retrieved using its index.")

        imap.push(n1t1)

        self.assertEqual(len(imap), 2, "Pushing an already added feature does nothing.")

        n1t2 = ("n1", "t2")
        imap.push(n1t2)

        self.assertEqual(len(imap), 3, "Pushing another new feature increases the imap size by 1.")

    def test_batch_imap_construction(self):
        imap = IndexMap()

        imap.batch_push([("n1", "t1"), ("n1", "t2"), ("n1", "t3"), ("n2", "t1"), ("n2", "t2")])

        self.assertEqual(len(imap), 6, "Pushing 5 distinct features results in 6 indexed features (including the intercept).")

        for i in range(len(imap)):
            self.assertEqual(
                imap.get_index(imap.get_nt(i)),
                i,
                "The imap defines a one-to-one mapping from the index domain to the feature name-term domain.",
            )

        expected_nts_1 = [("intercept", "intercept"), ("n1", "t1"), ("n1", "t2"), ("n1", "t3"), ("n2", "t1"), ("n2", "t2")]

        for nt in expected_nts_1:
            # Success here just means it did not throw. The actual mapping is an implementation detail.
            self.assertGreaterEqual(imap.get_index(nt), 0, f"Can forward-lookup an added feature: {nt}.")

        imap.batch_push(
            [
                # Already indexed
                ("n1", "t1"),
                ("n1", "t2"),
                ("n1", "t3"),
                # Not yet indexed
                ("n3", "t1"),
                ("n3", "t2"),
            ]
        )

        self.assertEqual(len(imap), 8, "Pushing a mix of indexed and un-indexed features, only the un-indexed are added.")

        for i in range(len(imap)):
            self.assertEqual(imap.get_index(imap.get_nt(i)), i, "After multiple batch pushes the index mapping remains correct.")

        expected_nts_2 = [
            ("intercept", "intercept"),
            ("n1", "t1"),
            ("n1", "t2"),
            ("n1", "t3"),
            ("n2", "t1"),
            ("n2", "t2"),
            ("n3", "t1"),
            ("n3", "t2"),
        ]

        for nt in expected_nts_2:
            # Success here just means it did not throw. The actual mapping is an implementation detail.
            self.assertGreaterEqual(imap.get_index(nt), 0, f"Can forward-lookup an added feature: {nt}.")

    def test_from_records_means_and_variances_builder(self):
        # coefficients
        means = [Feature("n1", "t1", 1.0), Feature("n1", "t2", 1.0), Feature("n2", "t3", 1.0)]
        variances = [Feature("n1", "t1", 1.0), Feature("n1", "t2", 1.0), Feature("n2", "t3", 1.0)]

        # training data
        records = [
            TrainingRecord(1, 1, 0, [Feature("n1", "t1", 1.0)]),
            TrainingRecord(1, 1, 0, [Feature("n2", "t3", 1.0)]),
            TrainingRecord(
                1, 1, 0, [Feature("n3", "t1", 1.0), Feature("n3", "t2", 1.0), Feature("n3", "t3", 1.0), Feature("n3", "t4", 1.0)]
            ),
        ]

        imap, meta = IndexMap.from_records_means_and_variances(records, means, variances)

        self.assertEqual(len(imap), 8)

        for i in range(len(imap)):
            self.assertEqual(
                imap.get_index(imap.get_nt(i)),
                i,
                "The imap defines a one-to-one mapping from the index domain to the feature name-term domain.",
            )

        expected_nts = [
            ("intercept", "intercept"),
            ("n1", "t1"),
            ("n1", "t2"),
            ("n2", "t3"),
            ("n3", "t1"),
            ("n3", "t2"),
            ("n3", "t3"),
            ("n3", "t4"),
        ]

        for nt in expected_nts:
            # Success here just means it did not throw. The actual mapping is an implementation detail.
            self.assertGreaterEqual(imap.get_index(nt), 0, f"Can forward-lookup an added feature: {nt}.")

        self.assertEqual(meta["num_nts_in_both"], 2, "Correct metadata: num features in shared by records and coeffs.")
        self.assertEqual(meta["num_nts_in_coeffs_only"], 1, "Correct metadata: num features only in coeffs.")
        self.assertEqual(meta["num_nts_in_records_only"], 4, "Correct metadata: num features only in records.")
