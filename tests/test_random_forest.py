import unittest

import numpy as np
import pandas as pd

from random_forest import make_pipeline, make_split, select_threshold


class RandomForestWorkflowTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_stratified(self):
        df = pd.DataFrame({"feature": range(100), "fraud_reported": [0] * 75 + [1] * 25})

        train_idx, test_idx = make_split(df, seed=42)
        again_train, again_test = make_split(df, seed=42)

        np.testing.assert_array_equal(train_idx, again_train)
        np.testing.assert_array_equal(test_idx, again_test)
        self.assertEqual(len(set(train_idx) & set(test_idx)), 0)
        self.assertEqual(len(test_idx), 20)
        self.assertEqual(df.iloc[test_idx]["fraud_reported"].sum(), 5)

    def test_pipeline_handles_missing_and_unseen_categories(self):
        X_train = pd.DataFrame({
            "amount": [100.0, 200.0, np.nan, 400.0, 500.0, 600.0],
            "kind": ["collision", "theft", None, "collision", "theft", "collision"],
        })
        y_train = [0, 1, 0, 1, 0, 1]
        X_new = pd.DataFrame({"amount": [np.nan], "kind": ["fire"]})

        model = make_pipeline(X_train, n_estimators=10)
        model.fit(X_train, y_train)

        probability = model.predict_proba(X_new)[0, 1]
        self.assertTrue(0 <= probability <= 1)

    def test_threshold_is_selected_from_training_labels_and_probabilities(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.3, 0.45])

        self.assertAlmostEqual(select_threshold(labels, probabilities), 0.3)


if __name__ == "__main__":
    unittest.main()
