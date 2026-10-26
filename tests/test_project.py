"""Checks for feature definitions, decision arithmetic and pipeline isolation."""

import unittest

import numpy as np
import pandas as pd

from telco_churn.data import RAW_FEATURE_COLUMNS, prepare_features
from telco_churn.experiment import make_pipeline, selection_index
from telco_churn.retention import retention_scenario
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


class ProjectChecks(unittest.TestCase):
    def test_retention_uses_comparable_losses(self):
        decisions, totals = retention_scenario([0, 0.2, 1 / 3, 0.8, 1])
        np.testing.assert_allclose(decisions.expected_savings_vs_none, [0, 0, 0, 140, 200], atol=1e-10)
        self.assertEqual(decisions.action.tolist(), ["none", "none", "none", "high", "high"])
        self.assertEqual(int(decisions.action.eq("low").sum()), 0)
        self.assertAlmostEqual(totals.iloc[0].expected_loss - totals.iloc[-1].expected_loss, 340)

    def test_invalid_probabilities_fail(self):
        for p in [[-0.1], [1.1], [np.nan], [[0.5]]]:
            with self.assertRaises(ValueError):
                retention_scenario(p)

    def test_selection_ignores_ineligible_high_recall(self):
        result = {"mean_test_recall": [0.99, 0.8, 0.8], "mean_test_f1": [0.3, 0.6, 0.65]}
        self.assertEqual(selection_index(result), 2)

    def test_feature_definitions_and_input_not_mutated(self):
        raw = pd.DataFrame([{c: "No" for c in RAW_FEATURE_COLUMNS}] * 4)
        raw["SeniorCitizen"] = 0
        raw["tenure"] = [0, 12, 24, 49]
        raw["MonthlyCharges"] = 50
        raw["TotalCharges"] = [" ", "600", "1200", "2450"]
        raw["MultipleLines"] = "Yes"
        original = raw.copy(deep=True)
        features = prepare_features(raw)
        self.assertEqual(features.Tenure_Group.tolist(), ["0-12 months", "0-12 months", "13-24 months", "49+ months"])
        self.assertEqual(features.Service_Count.tolist(), [1, 1, 1, 1])
        self.assertAlmostEqual(features.Avg_Monthly_Spend.iloc[1], 600 / 13)
        self.assertTrue(pd.isna(features.TotalCharges.iloc[0]))
        pd.testing.assert_frame_equal(raw, original)

    def test_smote_runs_inside_folds(self):
        from telco_churn.data import DEFAULT_DATA, load_dataset
        if not DEFAULT_DATA.exists():
            self.skipTest("Download data to run the fold-isolation check.")
        _, X, y, _ = load_dataset()
        X, y = X.iloc[:600], y.iloc[:600]
        pipeline = make_pipeline(LogisticRegression(max_iter=2000), smote=True)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=7)
        splits = list(cv.split(X, y))
        scores = cross_validate(pipeline, X, y, cv=splits, return_estimator=True, error_score="raise")
        self.assertFalse(hasattr(pipeline.named_steps["select"], "support_"))
        for (train, validation), fitted in zip(splits, scores["estimator"]):
            self.assertFalse(set(train) & set(validation))
            expected_mean = X.iloc[train]["tenure"].mean()
            scaler = fitted.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
            self.assertAlmostEqual(scaler.mean_[0], expected_mean)
            self.assertEqual(len(fitted.predict(X.iloc[validation])), len(validation))


if __name__ == "__main__":
    unittest.main()
