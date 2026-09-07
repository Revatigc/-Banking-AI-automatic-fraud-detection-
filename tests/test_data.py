import unittest

import pandas as pd

from src.data import make_demo_data, numeric_feature_frame


class DataContractTests(unittest.TestCase):
    def test_rejects_missing_feature_values(self):
        frame = pd.DataFrame({"amount": [20.0, None], "is_fraud": [0, 1]})
        with self.assertRaisesRegex(ValueError, "Missing values"):
            numeric_feature_frame(frame)

    def test_rejects_non_binary_target(self):
        frame = pd.DataFrame({"amount": [20.0, 30.0], "is_fraud": [0, 2]})
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            numeric_feature_frame(frame)

    def test_rejects_non_numeric_features(self):
        frame = pd.DataFrame({"merchant": ["a", "b"], "is_fraud": [0, 1]})
        with self.assertRaisesRegex(ValueError, "numeric"):
            numeric_feature_frame(frame)

    def test_returns_numeric_features_and_labels(self):
        features, labels = numeric_feature_frame(pd.DataFrame({"amount": [20.0, 30.0], "hour": [2, 14], "is_fraud": [0, 1]}))
        self.assertEqual(list(features.columns), ["amount", "hour"])
        self.assertEqual(labels.tolist(), [0, 1])

    def test_oracle_probability_is_opt_in(self):
        self.assertNotIn("oracle_fraud_probability", make_demo_data(rows=20).columns)
        oracle_frame = make_demo_data(rows=20, include_oracle=True)
        self.assertTrue(oracle_frame["oracle_fraud_probability"].between(0, 1).all())
